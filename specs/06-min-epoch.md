# 06: No early stop before epoch 30, with and without strong augmentation

Status: approved 2026-10-02, running.

## Question

1. Does a minimum epoch raise per-image CV Dice of arm C (0.9196)?
2. With training length fixed that way, does spec 04's strong augmentation beat it?

## Why

Training runs up to 60 epochs on a cosine schedule. Early stopping watches the 49-image stop set, keeps the best epoch's weights, and stops after 10 epochs without a new best. Since spec 02 it has cut a fold short before epoch 30 in 4 of 6 runs (A′, C, D and G). In G it kept weights from epoch 3, still in warm-up, and that fold lost 1.33 points. A fold that stops at epoch 27 never trains at the low learning rate at the end of the schedule.

This noise also blurs every comparison. Spec 04's arm F gained 0.16 points, but its fold 1 trained to epoch 53 where C's stopped at 27. So we can't tell how much of that gain came from the augmentation.

## Arms

| Arm | Config | What changes |
|---|---|---|
| C*, C + minimum epoch | `06-hires-min30.yaml` | `train.min_epochs: 30` against arm C |
| F*, F + minimum epoch | `06-hires-aug-min30.yaml` | `train.min_epochs: 30` against arm F (spec 04) |

`min_epochs: 30` means that before epoch 30 no weights are kept and the patience counter doesn't run. From epoch 30 on, early stopping works as before: keep the best stop-set epoch, stop after 10 epochs without a gain. So every fold trains at least 40 epochs and at most 60. Without `min_epochs` training behaves exactly as before.

F* uses spec 04's augmentation unchanged. Everything else matches arm C: folds, stop sets, seeds, schedule, model and TTA.

## Metrics and win conditions

Metrics are unchanged (frozen `eval/metrics.py`). The headline is per-image CV Dice without TTA. TTA Dice, Dice by image type and the epochs per fold are reported next to it.

Both comparisons use the usual rule: the mean CV Dice without TTA is at least 0.005 higher, and the arm scores at least as high on at least 4 of the 5 folds.

- **C* against C** (run `20260925-b3fe577-s02-hires`, 0.9196). C* fixes the protocol, not the model. It becomes the baseline for later specs if its mean is not below C's, even without a win.
- **F* against C***. This is the augmentation question. F* against F (0.9213) shows how much of F's gain came from training longer.

If either arm's TTA Dice reaches 0.937, we ask whether to score the locked test set.

## Budget

C ran 0.51 minutes per epoch and F 0.54. Each fold now runs 40 to 60 epochs, so C* should take 1.7 to 2.6 hours and F* 1.8 to 2.7 hours, about 4 to 5 GPU-hours together. The two kernels run in parallel. If fold 0 of either arm takes more than 40 minutes, we stop and ask.
