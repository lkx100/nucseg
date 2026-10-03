# 06: No early stop before epoch 30, with and without strong augmentation

Status: done 2026-10-03. No win for either arm: C* +0.10 points against C, F* −0.05 against C*. C* is the new baseline (see Outcome).

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

## Outcome

| Arm | Run | CV Dice | Dice with TTA | IoU | Pooled Dice | FGSM Dice | Epochs per fold | Minutes |
|---|---|---|---|---|---|---|---|---|
| C | 20260925-b3fe577-s02-hires | 0.9196 ± 0.0088 | 0.9211 | 0.8588 | 0.9378 | 0.818 | 48/27/36/47/39 | 100 |
| F | 20260928-a6250b2-s04-hires-aug | 0.9213 ± 0.0059 | 0.9227 | 0.8608 | 0.9387 | 0.833 | 60/53/34/34/46 | 123 |
| C* | 20261002-fa7e63a-s06-hires-min30 | 0.9207 ± 0.0081 | 0.9219 | 0.8611 | 0.9390 | 0.812 | 43/60/60/49/52 | 131 |
| F* | 20261002-fa7e63a-s06-hires-aug-min30 | 0.9201 ± 0.0065 | 0.9214 | 0.8594 | 0.9383 | 0.832 | 52/44/60/54/46 | 129 |

Per fold:

| Arm | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| C | 0.9202 | 0.9146 | 0.9219 | 0.9325 | 0.9090 |
| F | 0.9204 | 0.9177 | 0.9201 | 0.9314 | 0.9167 |
| C* | 0.9188 | 0.9197 | 0.9233 | 0.9320 | 0.9095 |
| F* | 0.9156 | 0.9164 | 0.9229 | 0.9303 | 0.9155 |

- **C* against C:** +0.10 points, ahead on 3 of 5 folds (−0.14, +0.50, +0.14, −0.05, +0.06). No win. The mean is
  not below C's, so C* is the baseline from here. The gain is fold 1, which trained to epoch 60 instead of 27.
- **F* against C*:** −0.05 points, ahead on 1 of 5 folds (−0.32, −0.32, −0.04, −0.17, +0.60). No win.
- **F* against F:** −0.11 points. With training length fixed, the strong augmentation adds nothing to Dice. F's
  +0.16 in spec 04 came from noise and from its fold 1 training longer, not from the augmentation.

All four arms sit within 0.17 points, inside the run-to-run noise of about 0.3. Neither TTA Dice reaches 0.937,
so the locked test set stays untouched.

Dice by image type:

| Type | Images | C | F | C* | F* |
|---|---|---|---|---|---|
| Fluorescence, small | 434 | 0.921 | 0.924 | 0.921 | 0.921 |
| Fluorescence, large | 112 | 0.946 | 0.947 | 0.947 | 0.946 |
| H&E, colour | 108 | 0.889 | 0.889 | 0.895 | 0.892 |
| Brightfield, grayscale | 16 | 0.893 | 0.901 | 0.908 | 0.908 |

Longer training helped the two weak types a little: H&E +0.6 points and brightfield +1.4 (16 images, so noisy).
That matches the brightfield gain the 2× scale gave in specs 02 and 05, at a third of the cost.

The minimum epoch did its job: no fold stopped before epoch 43. But the stop set is still a weak guide. In C*'s
fold 1 the stop-set Dice fell from about 0.925 to 0.895 near epoch 23 and stayed there to epoch 60, while that
fold's validation Dice rose 0.50 against C. Spec 05's arm G showed the same drop on the same fold. With 49 images,
one or two images that flip are enough to move the stop-set mean by 3 points. This also explains why C's fold 1 stopped at 27.

The augmentation still helps under attack (FGSM Dice 0.832 against 0.812), as in spec 04.

Fold 0 took 24 minutes in C* and 29 in F*, inside the 40-minute limit. C* took 131 minutes, 1.3 times C.

Verdict: C* is the baseline. Strong augmentation is closed for the Dice target. Training-recipe changes (NSL,
2× scale, augmentation, training length) have all landed between 0.919 and 0.921, so the remaining 1.5 points
need a different kind of change.
