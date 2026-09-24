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
