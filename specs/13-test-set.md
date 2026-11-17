# 13: How does arm M score on the 67 locked test images?

Status: done 2026-11-17. M scores 92.99 ± 0.18 with TTA on all 67 test images, and 93.58 ± 0.18 without photo `adc315bd`, whose hand-drawn answer is missing 10 of its 15 nuclei (see Outcome).

## Question

Arm M (Swin-T U-Net with scSE in the decoder, spec 11) scored 92.86 ± 0.37 with TTA in cross-validation. What does it score on the 67 test images that no training run or decision has touched?

## Why

The user asked for it on 2026-11-17. Specs 01 to 12 kept the test set locked; this is the first time it is opened. After this run it no longer counts as unseen for any later choice between models.

## Arm

| Arm | Config | What it does |
|---|---|---|
| M on test | `13-scse-test.yaml` | Loads M's five saved models (kernel `s11-scse`). Each one scores all 67 test images, with and without TTA, as 256 px tiles at half overlap |

No training and no new weights. `nucseg.ensemble` gets one option, `eval.split: test`, which swaps each fold's validation images for the test images. `eval/` is not edited. The five models are scored one by one, not averaged (no ensembles).

Check: the test set's make-up is 43 dark-gray/small, 11 bright-colour/small (H&E), 11 dark-gray/large and 2 bright-gray/large.

## Reading

Per-image Dice with TTA, in the paper's fold table: one row per fold model, mean ± std across the five, highest, mean + std. The spread here comes only from the models, since all five see the same images, so it is not the same thing as the CV spread.

## Budget

67 images × 5 models × 2 (with and without TTA): about 15 GPU-minutes. If the kernel runs longer than 45 minutes, we stop and ask.

## Outcome

Run `20261117-769438f-s13-scse-test`, 6.6 minutes, 1.0 GB. All 67 test images were scored by all five models.

Per-image Dice with TTA:

| | Test (this run) | CV (spec 11) | Target |
|---|---|---|---|
| Mean ± std | 92.99 ± 0.18 | 92.86 ± 0.37 | 92.81 |
| Highest fold | 93.22 | 93.43 | |
| Mean + std | 93.17 | 93.23 | 93.68 |

Per fold with TTA: 93.02, 93.22, 92.67, 93.05, 92.98. IoU 87.22 ± 0.25, pixel accuracy 98.08 ± 0.05. Without TTA: 92.84 ± 0.17.

- M holds up on unseen photos: the test mean is 0.13 above its CV mean, so the CV score was not tuned to the folds.
- The mean clears 92.81; mean + std (93.17) misses 93.68. The std is half the CV one because all five models see the same images, so mean + std on the test set is not the paper's number and should not be compared with it.
- One photo, `adc315bd` (dark-gray, 360×360), scores 53.9 with every model. Without it the mean would be 93.58. The other 66 are all above 80.
- By type with TTA: dark-gray/small 93.51, dark-gray/large 93.67, H&E 90.76, bright-gray/large 90.31 (only 2 photos).

### Second run, without `adc315bd`

A figure of the photo (photo, hand-drawn answer, model, overlap) shows about 15 clear nuclei, of which the answer draws 5. The model finds all 15, so its 10 correct finds count as errors. The user asked to score again without it: run `20261117-538bcb3-s13-scse-test-no-adc`, config `13-scse-test-no-adc.yaml`, 66 photos, 6.0 minutes.

| Per-image Dice with TTA | 67 photos | 66 photos | Target |
|---|---|---|---|
| Mean ± std | 92.99 ± 0.18 | 93.58 ± 0.18 | 92.81 |
| Highest fold | 93.22 | 93.82 | |
| Mean + std | 93.17 | 93.76 | 93.68 |

Per fold with TTA: 93.61, 93.82, 93.26, 93.65, 93.57. IoU 87.98 ± 0.25, pixel accuracy 98.27 ± 0.05. Without TTA: 93.43 ± 0.17.

Both numbers clear the target on paper, but this is not a win over the paper:
- The photo was dropped after we saw its score. Other photos have wrong answers too (about 2% of nuclei, spec 10 label check) and stay in.
- The std is between five models on the same photos, not between folds, so mean + std is not the paper's measure. The comparable result is still M's CV score, 92.86 mean and 93.23 mean + std.
