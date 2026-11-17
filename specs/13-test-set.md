# 13: How does arm M score on the 67 locked test images?

Status: running.

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
