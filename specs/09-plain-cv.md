# 09: C* under plain 5-fold CV, without the wrongly labelled image

Status: done 2026-10-11. Arm J run as `20261011-1a5ff70-s09-plain-cv`. It misses the target on both rows, by 0.32 points on the highest fold and 0.06 on the mean (see Outcome). J is the new baseline.

## Question

What does C* score under plain 5-fold cross-validation, the protocol of the paper the 0.937 target comes from, once image `7b38c917` is left out?

## Why

The user decided two things on 2026-10-11.

1. Leave out image `7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80`. Its mask has 26 pixels on an image full of nuclei. Models that train on it predict almost nothing on its two look-alikes (Dice 0.008 and 0.364).
2. Drop the stop set. The paper trained and validated on each fold with no third set, and our stop set made the protocol harder to follow and to compare.

## Arm

| Arm | Config | What changes against C* |
|---|---|---|
| J, C* under plain CV | `09-hires-plain-cv.yaml` | `data.stop_set: false`, `data.exclude: [7b38c917…]` |

In practice:

- Each fold trains on 481 or 482 images, up from 433 or 434. The 49 stop-set images go back into training.
- The validation fold is scored after every epoch. From epoch 30 on, the weights of the epoch with the best validation Dice are kept, and training stops after 10 epochs without a gain. This is the old rule, now watching the validation fold.
- 602 images are scored, not 603. Fold 0 has 120.
- `eval/` is not edited. Both changes live in the config. `eval/splits.json` still lists the image and the stop sets, and this config does not use them.

Everything else matches C*: folds, seeds, schedule, model, augmentation, minimum epoch and TTA. Configs without the two new keys behave exactly as before.

## What this does to comparisons

- The validation fold now picks the epoch and also gives the score, so the score leans high by an amount this run can't measure. The 13 earlier runs kept the validation fold out of that choice. J's score does not compare with theirs. J becomes the baseline, and later runs compare with J.
- From the per-image scores of spec 07: leaving the image out of scoring adds about 0.15 points, and leaving it out of training should add about 0.25 (an estimate). Any gain beyond that comes from the 49 extra training images and from the epoch choice, and this run can't separate the two.
- The locked test set is now the only data that no decision has touched. It stays locked.

## Metrics and target

The user set these on 2026-10-11.

Every run is reported as the paper's table: one row per fold, then mean ± std and the highest fold, for Dice, IoU and pixel accuracy. `summary.txt` now holds this table under the `RESULT` line, and `uv run python -m nucseg.foldtable runs/<run_id> ...` prints it for any run. The std in the table is the population std, as in the paper. `eval/metrics.py` is unchanged.

The target is the paper's table, read on per-image Dice with TTA:

| Row | Paper | Target |
|---|---|---|
| Highest fold | 93.68 | at least 93.68 |
| Mean of five folds | 92.81 | at least 92.81 |

The user chose the highest fold. The mean row is the agent's addition, so that no row of the paper's table stays above ours. If J reaches both, we ask whether to score the locked test set.

The user left the other open points to the agent. They are settled like this:

- Which Dice: per image, as before. The paper's table points to Dice over all pixels of a fold (see below). On our predictions the per-image Dice is the lower of the two, so a target reached per image is also reached the paper's way. Pooled Dice stays in the table.
- TTA counts, because it is part of our method. Dice without TTA is shown next to it.
- The validation fold picks the kept epoch, as in the paper's plain cross-validation with early stopping.
- Images keep their original size, with 256×256 training crops (the user's call).
- The 67 test images stay locked. The paper has no test set and trains on about 536 images per fold; we train on 481 or 482.

## What the reference paper says

The paper is "NuRAFormer: A Hybrid UNet-Transformer with Reverse Attention and Boundary-Aware Learning for Nuclei Segmentation" (J. Sindhu Manogna et al.), read on 2026-10-11 from the user's PDF.

| Fold | Dice | IoU | Pixel accuracy |
|---|---|---|---|
| 1 | 91.66 | 84.61 | 97.63 |
| 2 | 93.70 | 88.14 | 98.29 |
| 3 | 91.84 | 84.90 | 97.71 |
| 4 | 93.41 | 87.64 | 98.16 |
| 5 | 93.44 | 87.69 | 98.16 |
| Mean ± std | 92.81 ± 0.87 | 86.60 ± 1.52 | 97.99 ± 0.27 |
| Highest | 93.68 | 88.12 | 98.26 |

- The headline 93.68 is the row "Highest". It is not a fold: in all three columns it equals mean + std (92.81 + 0.87, 86.60 + 1.52, 97.99 + 0.27). The best single fold is fold 2 at 93.70. Found by the user on 2026-10-17. `nucseg.foldtable` prints a "Mean + std" row for ours; J's is 93.16, lower than its best fold, because its folds are closer together (std 0.40 against 0.87).
- Five-fold stratified cross-validation on all 670 images, with no separate test set. Early stopping is used and no third set is mentioned.
- All images and masks are resized to 512×512 before training.
- Dice is 2|P∩G| / (|P| + |G|) on the nucleus mask. The paper doesn't mention test-time augmentation.
- In every row, Dice equals 2·IoU / (1 + IoU) to within 0.01. That holds when both come from one count of pixels over the whole fold (or over large batches). With per-image averaging the two drift apart: for C* the same formula gives 92.54 from our IoU, and our per-image Dice is 92.07. So the paper's Dice is very likely pooled over a fold, with every image weighing the same because all are 512×512.

Our earlier runs in the same form are printed by `nucseg.foldtable`. Under the old protocol C* has a per-image mean of 92.07 ± 0.73 and a highest fold of 93.20 (with TTA 92.19 and 93.33). The four-model average H4 has 92.30 and 93.37 (with TTA 92.35 and 93.46).

A one-off check of the paper's kind of Dice on H4 without TTA, pooled over a fold with every image weighted equally: mean 92.62 ± 0.51, highest 93.51. Without image `7b38c917` the mean is 92.72. Our `dice_pooled` (94.03 for H4) is higher because it counts raw pixels, so the large photos with large nuclei dominate it. It doesn't compare with the paper.

One question is left for the author: are Dice and IoU computed once from the pixel counts of the whole validation fold, per batch, or per image and then averaged?

## Budget

C* took 131 minutes. Scoring about 120 images per epoch in place of 49, and training on 11% more images, should bring J to about 160 to 190 minutes, or about 3 GPU-hours. If fold 0 takes more than 55 minutes, we stop and ask.

## Outcome

Run `20261011-1a5ff70-s09-plain-cv`, 132 minutes, 6.1 GB, folds ran 41 to 47 epochs.

| Row | Paper | Target | Arm J, per image with TTA | Short by |
|---|---|---|---|---|
| Highest fold | 93.68 | ≥ 93.68 | 93.36 | 0.32 |
| Mean of five folds | 92.81 | ≥ 92.81 | 92.75 ± 0.40 | 0.06 |

The full table, per image with TTA next to the plain score:

| Fold | Dice | Dice TTA | Dice pooled | IoU | Pixel accuracy |
|---|---|---|---|---|---|
| 1 | 92.66 | 92.80 | 94.53 | 86.73 | 98.02 |
| 2 | 92.01 | 92.12 | 94.05 | 85.81 | 98.11 |
| 3 | 92.43 | 92.59 | 93.38 | 86.31 | 97.53 |
| 4 | 93.20 | 93.36 | 94.24 | 87.61 | 98.17 |
| 5 | 92.66 | 92.89 | 94.22 | 86.74 | 97.93 |
| Mean ± std | 92.59 ± 0.39 | 92.75 ± 0.40 | 94.09 ± 0.38 | 86.64 ± 0.59 | 97.95 ± 0.23 |
| Highest | 93.20 | 93.36 | 94.53 | 87.61 | 98.17 |

### The gain is where the bad label was, and only there

Against C* under the old protocol, per fold, with TTA:

| Fold | C* | J | Change |
|---|---|---|---|
| 1 | 91.97 | 92.80 | +0.84 |
| 2 | 92.07 | 92.12 | +0.05 |
| 3 | 92.41 | 92.59 | +0.17 |
| 4 | 93.33 | 93.36 | +0.04 |
| 5 | 91.16 | 92.89 | +1.73 |

Folds 1 and 5 of the table are fold 0 and fold 4 in the code. `7b38c917` is validated in fold 0, and its two
look-alikes `0402a81e` and `df53d0b6` in fold 4. Those are the two folds that moved. The other three gained
0.04 to 0.17 points, which is inside the run-to-run noise of about 0.3. So the whole 0.56-point rise of the mean
comes from the wrong label, as predicted, and not from the 49 extra training images or the epoch choice.

The fold spread fell from 0.70 to 0.40, because the two weak folds were weak for that one reason.

### Why the highest fold barely moved

The headline target is the highest fold, and that is the row the change did not touch: 93.33 for C*, 93.36 for J.
The strongest fold never held a bad-label image, so it had nothing to gain. Lifting the floor does not lift the
ceiling. Anything aimed at the 93.68 row has to make a good fold better, not fix a broken one.

### Every fold peaked just after the minimum epoch

Best epochs were 35, 33, 37, 31 and 33, with `min_epochs` at 30, and every fold then ran its full 10 patience
epochs without a gain and stopped at 41 to 47. The curves plateau from about epoch 20. So the kept epoch is a
noise peak on a flat plateau, not a real optimum, and the last 10 to 15 epochs of every fold buy nothing.

### Caveats

- The validation fold both picked the epoch and gave the score, so J leans high by an amount it cannot measure.
  `val_epoch_dice` equals the reported fold Dice exactly. The spec accepted this to match the paper.
- J does not compare with the 13 earlier runs. It is the baseline from here.
- Robustness improved on its own: FGSM Dice 0.832 against C*'s 0.812.
- Budget: fold 0 took 28.5 minutes against the 55-minute limit, and the run came in at 132 minutes against the
  160 to 190 estimate, because scoring 120 images an epoch cost less than feared.

Verdict: no win on either target row. The protocol change did what the diagnosis said it would and no more.
The locked test set stays untouched.
