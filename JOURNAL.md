# Research journal

Append-only. Newest entries at the bottom. Read on demand, not every session.

## 2026-09-22: repo setup
Project renamed to `nucseg` and pushed to github.com/lkx100/nucseg. Kaggle is the only GPU
backend; Modal dropped. uv project on Python 3.12 to match Kaggle's runtime; `kaggle` and
`jupytext` as uv tools. W&B chosen for live curves (entity `lkx100-kl-university`).
Next: Kaggle env probe kernel to pin dependency versions, then freeze `eval/`.

## 2026-09-22: Kaggle env probe (kernel `luckyx100/nucseg-env-probe`, v2)
First push failed: jupytext writes no `kernelspec`, and Kaggle rejects that
("No kernel name found in notebook"). `launch/kaggle/push.sh` now patches it in.
v2 completed on a T4 (15.6 GB). Image versions: Python 3.12.13, torch 2.10.0+cu128,
torchvision 0.25.0, numpy 2.0.2, albumentations 2.0.8, scikit-image 0.25.2, wandb 0.26.1.
W&B failed with "ConnectionError ... communicate with service": no secret attached yet;
the probe now reports the secret lookup and the `wandb.init` separately to tell them apart.

## 2026-09-23: probe v3: dataset attached, W&B secret unreachable
Dataset `sindhu9642/nuclei-seg-corrected` mounts at
`/kaggle/input/datasets/sindhu9642/nuclei-seg-corrected/` (newer Kaggle path layout, not
`/kaggle/input/<slug>/`). DSB2018 layout: `stage1_train/<id>/images/<id>.png` plus one PNG per
nucleus in `masks/`. `UserSecretsClient().get_secret` fails with ConnectionError even with the
secret attached in the UI: secrets are a known gap for CLI-pushed kernels (kaggle-cli #582).
Probe v4 (2026-09-23): plain `wandb.init` with the secret attached in the UI also fails -
`WANDB_API_KEY` is not in the environment and wandb raises "No API key configured".
The Kaggle runtime does not inject secrets into CLI-pushed runs.
Probe v5 (2026-09-23): Kaggle's own `UserSecretsClient().get_secret("WANDB_API_KEY")` snippet
fails the same way (ConnectionError), so secrets are unusable in CLI-pushed runs. Needs an
alternative: W&B offline + local `wandb sync`, or the key in a private dataset.

## 2026-09-23: W&B switched to offline logging on Kaggle
Probe v6 logged W&B offline to `/kaggle/working/wandb/`, and `launch/kaggle/pull.sh` downloaded
the run folder. The upload with `wandb sync` failed: no W&B login on this machine yet. The run
folder is kept in `runs/env-probe/wandb/` so it can be synced after `wandb login`.
`wandb login` saved to `~/.netrc`, and `wandb sync` uploaded the probe run
(https://wandb.ai/lkx100-kl-university/nucseg/runs/imhsavb4). The offline W&B loop works end to end.

## 2026-09-24: goal set, spec 01 drafted
Goal: foreground Dice ≥ 0.937, with IoU and pixel accuracy reported alongside. The downloaded dataset holds
only `stage1_train` (670 images, no labelled test set), so we hold out about 10% as a test set and run 5-fold CV
on the rest. Image types: 546 dark fluorescence, 108 colour brightfield/H&E, 16 grayscale brightfield.
Nuclei cover 14% of pixels on average. Spec 01 compares a Swin-T U-Net with and without NSL adversarial
regularization. A brute-force search for overlapping crops was too slow on this machine; `eval/make_splits.py` needs a narrower search.

## 2026-09-24: spec 01 approved, splits built
Decisions: per-image mean Dice is the headline, the test set is locked, 256 crops now and 512 later, and early
stopping stays. To keep the validation fold clean, early stopping watches a separate stop set (10% of each fold's
training images). `eval/make_splits.py` found no exact duplicates or overlapping crops, but one near-duplicate pair
(two frames of the same embryo), which now share a split. The test set and folds match the full dataset's mix of
image types and nucleus coverage (0.13 to 0.16 against 0.139 overall).

## 2026-09-25: spec 01 arm A (baseline), run 20260925-1de4ab8-s01-base
CV Dice 0.918 ± 0.005 per image (IoU 0.856, pixel accuracy 0.979). Pooled Dice is 0.938, already above
0.937, but the target uses the per-image mean, so the gap is 1.9 points. Folds stopped at epochs 36 to 57,
with stop-set Dice flat near 0.925 from about epoch 25. 90 minutes and 5.7 GB on a T4.
Weakest types: colour H&E 0.892 and grayscale brightfield 0.898. Fluorescence scores 0.918 (small) and 0.945 (large).
In the H&E grid, most "false positives" are purple nuclei missing from the ground truth, so label noise caps that type.
FGSM at 2/255 drops Dice to 0.800, so the baseline is fragile. That is what arm B's NSL term targets.

## 2026-09-25: spec 01 arm B (NSL), run 20260925-d981a3d-s01-nsl
CV Dice 0.914 ± 0.005, 0.34 points below arm A, and lower on all 5 folds (by 0.04 to 0.56 points). So NSL at
α = 0.2, ε = 2/255 fails the win condition. It did what adversarial training promises: Dice under FGSM 2/255 rose
from 0.800 to 0.918, about equal to its clean Dice. It cost 2.4 times the GPU time (215 minutes).
The prediction grids look the same as arm A's. Arm B's pull failed on a network blip in `pull.sh`'s status loop;
the loop now retries.

## 2026-09-25: spec 02 arm A′ (baseline rerun), run 20260925-b3fe577-s02-base
Identical setup to spec 01 arm A, but CV Dice 0.9143 against 0.9178. Most of the gap is fold 1: a lucky stop-set
spike at epoch 8 ended training at epoch 18 while the learning rate was still high (0.9050 against 0.9165).
The other folds differ by 0.0 to 0.3 points. So early stopping with patience 10 on a 49-image stop set is fragile,
and run-to-run noise is about 0.3 points. That puts spec 01's NSL result (0.9144) level with this rerun; only
its robustness gain stands. TTA adds 0.12 points (0.9155). Next spec should stop early only after a minimum
number of epochs.

## 2026-09-25: spec 02 arm C (full-resolution skip), run 20260925-b3fe577-s02-hires
CV Dice 0.9196 (0.9211 with TTA), 0.53 points above A′ and ahead on 4 of 5 folds, so it passes the win rule.
The gain sits where the edge idea predicts: small fluorescence images rose from 0.914 to 0.921, while H&E and
brightfield barely moved. The margin is close to the noise, though. Against spec 01's arm A it is only +0.18,
and part of the gain over A′ comes from A′'s cut-short fold 1. Fold 1 stopped early in C too (epoch 27).
It costs 5% more time (100 minutes). Still 1.6 points (1.5 with TTA) short of 0.937.

## 2026-09-25: spec 02 arm D (2× input scale), run 20260925-cf38771-s02-scale2
CV Dice 0.9190 (0.9197 with TTA), 0.47 points above A′ and ahead on 4 of 5 folds. It misses the 0.5-point bar
by a hair, and it is level with C (0.9196) at 2.85 times the time (285 minutes, 13.7 GB). C and D help different
images: C gains more on small fluorescence (0.921 against 0.919), while D gains on brightfield (0.912 against
0.893) and H&E (0.894 against 0.889). So combining them may add up. Spec 02 winner: C. Fold 1 stopped early again (epoch 28).

## 2026-09-28: spec 04 arm F (strong augmentation), run 20260928-a6250b2-s04-hires-aug
CV Dice 0.9213 (0.9227 with TTA), +0.16 points against C and ahead on 3 of 5 folds, so no win. The colour and
stain jitter aimed at H&E, but H&E stayed at 0.889. Brightfield rose from 0.893 to 0.901 and fluorescence moved
by 0.1 to 0.2. Fold 1 trained to epoch 53 instead of stopping at 27, and the fold spread shrank. Best TTA Dice
so far, still 1.4 points short. The kernel waited 6 h 20 min for a GPU while `kaggle kernels status` said RUNNING;
the session itself took 2.06 h.

## 2026-09-28: spec 03 arm E (C + NSL at ε 1/255), run 20260928-f8051ad-s03-hires-nsl1
CV Dice 0.9197 (0.9212 with TTA), level with C (+0.01) and ahead on 3 of 5 folds, so no win. FGSM 2/255 Dice
rose from 0.818 to 0.905, the same trade as spec 01's ε 2/255. It took 227 minutes, 2.3 times C. That closes NSL
for the Dice target: it buys robustness, not accuracy. Both spec 03 and 04 kernels waited about 6 hours for a
GPU and started training at the same minute, so the delay was Kaggle's queue, not our code.

## 2026-09-30: spec 05 arm G (C at 2× input scale), run 20260929-e7415b1-s05-hires-scale2
CV Dice 0.9196 (0.9206 with TTA), exactly level with C and ahead on 4 of 5 folds, so no win. Fold 0 lost 1.33
points: its stop-set Dice peaked at epoch 3, during warm-up, and training stopped at epoch 13. The other folds gained
0.33 on average. G kept most of D's brightfield and H&E gains but not C's small-fluorescence gain. It took 343
minutes (3.4 times C) and peaked at 15.2 GB, close to the T4's limit. Early stopping cut a fold short in 4 of the 6 runs
since spec 02 began, so fixing it comes next.

## 2026-10-03: spec 06 arm C* (C + minimum epoch 30), run 20261002-fa7e63a-s06-hires-min30
CV Dice 0.9207 (0.9219 with TTA), +0.10 points against C and ahead on 3 of 5 folds. That is no win, but it is not
below C, so C* is the baseline from here. No fold stopped early: they ran 43 to 60 epochs. Fold 1 trained to
epoch 60 instead of 27 and gained 0.50. H&E rose from 0.889 to 0.895 and brightfield (16 images) from 0.893 to 0.908.
In fold 1 the stop-set Dice fell 3 points near epoch 23 and stayed there, while the fold's validation Dice rose.
So the 49-image stop set is a weak guide for picking the epoch. It took 131 minutes, 1.3 times C.

## 2026-10-03: spec 06 arm F* (F + minimum epoch 30), run 20261002-fa7e63a-s06-hires-aug-min30
CV Dice 0.9201 (0.9214 with TTA), 0.05 points below C* and ahead on 1 of 5 folds, so no win. It is also 0.11
below F. With training length fixed, the strong augmentation adds nothing to Dice; F's +0.16 in spec 04 was noise
and longer training. It still helps under attack (FGSM Dice 0.832 against 0.812). C, F, C* and F* all sit within
0.17 points, inside the run-to-run noise of about 0.3. Still 1.5 points short of 0.937 (best TTA stays F's 0.9227).

## 2026-10-10: spec 07 arms H2 and H4 (averaged predictions), runs 20261003-512e574-s07-ens2 and -s07-ens4
Both kernels finished on 2026-10-03; H4 and spec 08 were pulled today. H2 (C* and F*) scores 0.9217 (0.9223 with
TTA), +0.10 against C* and ahead on 4 of 5 folds. H4 (C*, F*, C, F) scores 0.9230 (0.9235 with TTA, the best so
far), +0.23 and ahead on all 5 folds. Both miss the 0.5-point bar, so no win. Every member alone matched its own
run within 0.001. Edge diagnostic: 83% of the wrong pixels lie within 2 pixels of a nucleus boundary, and Dice
with that band ignored is 0.975. H2 took 9 minutes and H4 24.

## 2026-10-10: spec 08 arm I (edge-weighted loss), run 20261003-e961d17-s08-hires-edge
CV Dice 0.9213 (0.9229 with TTA), +0.06 against C* and ahead on 4 of 5 folds, so no win. Brightfield rose from
0.908 to 0.912 and the other types moved by 0.1 or less. FGSM Dice is 0.808 against 0.812. Weighting the edge
pixels 5 times did not move the edge errors, which fits the label-noise risk the spec named. It took 123 minutes.

## 2026-10-10: where the missing points are (per-image scores of H4, no new run)
The median per-image Dice is 0.938, above the target. The mean is 0.9235 because of a tail: 47 of 603 images
score below 0.85 and cost 1.46 points, more than the whole 1.35-point gap. 44 of the 47 are also below 0.85 in H2.
One cause is a wrong label. Image `7b38c917` shows nuclei on about 16% of its area, but its mask has 26 pixels.
It is one of only three 260×347 images among the 603 CV images (the locked test set holds two more). The other two, `0402a81e` and `df53d0b6`, are validated in fold 4 by
models that trained on the empty mask, and they score 0.008 and 0.364 with errors about the size of their masks.
So those models predict almost nothing on them. The three images cost 0.40 points of CV Dice (0.9275 without them).
The same two images sit in fold 1's stop set (and `0402a81e` in fold 3's). That explains the 3-point drop of fold
1's stop-set Dice near epoch 23 and the early stops of fold 1 since spec 02. No other image scores below 0.6.
Of the 47, 28 are 256×256 fluorescence images and 9 are H&E. The four fluorescence ones among the worst 12 are
dim and out of focus. Nobody has looked at predictions on the rest of the tail yet.

## 2026-10-11: answer-key scan of the 603 CV images (no new run)
A scan without any model (mask size, number of mask files, agreement with an Otsu cut of the image) found no second
blank mask. Four masks have a single file; `7b38c917` is the only one on an image full of nuclei. Gap to 0.937 by
type (H4 with TTA): small fluorescence 0.83 points, colour 0.65, grayscale brightfield 0.05, large fluorescence −0.19.
The low fluorescence images are not the ones with the smallest nuclei: 22 of 37 have a mean nucleus area of 229 to 658 px.
The user says 0.937 comes from a paper that ran plain 5-fold CV with no stop set. The paper is not in the repo yet.
How it averages Dice decides whether the numbers compare: our pooled Dice is already 0.939 (C*) and 0.940 (H4).

## 2026-10-11: protocol changed by the user, spec 09 drafted
Decisions: leave image `7b38c917` out of training and scoring, and drop the stop set, so each fold trains on the
other four and the validation fold picks the kept epoch. Both are config keys (`data.exclude`, `data.stop_set`);
`eval/` is untouched. Scores under the new protocol don't compare with the 13 earlier runs, so spec 09 reruns C*
as the new baseline (arm J). The user knows the paper's author; the questions for them are in the spec.

## 2026-10-11: target and table form set by the user (no new run)
Results are now shown as the paper's table (per fold, mean ± std, highest) for Dice, IoU and pixel accuracy:
`nucseg.foldtable`, also written into `summary.txt`. Target: per-image Dice with TTA, highest fold ≥ 93.68 and mean
≥ 92.81 (the paper's two rows). Best so far under the old protocol: H4 with 92.35 and 93.46.
The paper's Dice equals 2·IoU/(1+IoU) in every fold, so it is very likely pooled over a fold of 512×512 images.
H4 scored that way (every image weighted equally) gives 92.62 and 93.51; our raw-pixel `dice_pooled` (94.03) does
not compare with the paper. Details in spec 09.

## 2026-10-11: spec 09 arm J (C* under plain CV), run 20261011-1a5ff70-s09-plain-cv
Per-image Dice with TTA 92.75 ± 0.40, highest fold 93.36. That misses the paper's two rows by 0.06 on the mean
and 0.32 on the highest, so no win, but J is the baseline from here. The 0.56-point rise over C* is all bad label:
folds 0 and 4, which hold `7b38c917` and its two look-alikes, gained 0.84 and 1.73 points, while the other three
gained 0.04 to 0.17, inside the 0.3 noise. The fold spread fell from 0.70 to 0.40 for the same reason.
The headline target is the highest fold, and that row barely moved (93.33 to 93.36): the strongest fold never
held a bad image, so it had nothing to gain. Lifting the floor does not lift the ceiling, and the next idea has to
make a good fold better. Every fold peaked at epoch 31 to 37 with min_epochs at 30, then ran its full 10 patience
epochs for nothing and stopped at 41 to 47; the curves are flat from about epoch 20, so the kept epoch is a noise
peak on a plateau. 132 minutes against the 160 to 190 estimate, fold 0 28.5 min against the 55-minute limit.
Caveat the spec accepted: the validation fold both picked the epoch and gave the score, so J leans high.
FGSM Dice rose to 0.832 from 0.812 on its own.

## 2026-10-12: spec 10 arm K (J scored photo by photo), run 20261012-c4b6e50-s10-j-per-image
No training; 6 minutes. It reproduces J exactly and saves all 602 masks, so the rest ran on this machine
(`reports/photo_errors.py`). 89% of the wrong pixels are rings along outlines, 4% are missed nuclei and 7% invented
blobs. The rings are one-sided: in 38 of the 46 photos below 0.85 the model draws every nucleus too wide (23) or too
narrow (15), and over all photos it is too wide as often as too narrow. So the hand-drawn answers differ from photo
to photo in how generously the nuclei are outlined, and the model draws a middle width. With the best grow or shrink
of up to 2 pixels per photo, picked with the answer in hand, the mean is 93.41 and the highest fold 93.86, over both
target rows; nearly all of it comes from small fluorescence photos (92.99 to 93.90). That is a ceiling, not a method.
It explains why the edge-weighted loss of spec 08 changed nothing: the model is not unsure where the edge is, the
answers disagree about it. Open: can the width be predicted from the photo? No GPU needed to check.
