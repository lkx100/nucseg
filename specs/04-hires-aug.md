# 04: Strong augmentation on the full-resolution-skip model

Status: done 2026-09-28. No win: +0.16 points, 3 of 5 folds (see Outcome).

## Question

Does stronger data augmentation raise per-image CV Dice of arm C, the spec 02 winner (0.9196)?

## Why

Arm C trains with random crops, scaling between 0.75 and 1.25, flips, 90° rotations and small brightness and contrast changes. The weakest image types are H&E (0.889) and grayscale brightfield (0.893). They are 124 of the 670 images (18%), so each fold trains on only about 80 of them. Colour and stain changes give the model more varied examples of exactly those images. Warps, blur and noise add shape, focus and camera variation to all types.

This runs on arm C, not on spec 03's C + NSL, so it changes one thing against a finished baseline and can run while spec 03 is still going. If both win, a later run combines them.

## Arm

| Arm | Config | What changes against arm C |
|---|---|---|
| F, C + strong aug | `04-hires-aug.yaml` | adds `data.aug` |

What `data.aug` adds to each training crop, after the existing crop, flips and rotation:

| Step | Setting |
|---|---|
| Elastic warp | 30% of crops; random shifts with a 3-pixel std on an 8×8 lattice, smoothed in between. The mask gets the same warp. |
| Hue and saturation | every crop; hue shifted by up to 5% of the colour wheel, saturation scaled by 0.7 to 1.3. Grayscale images have no saturation, so they don't change. |
| Gamma | every crop; 0.7 to 1.4, log-uniform |
| Gaussian blur | 20% of crops; sigma 0.3 to 1.5 pixels |
| Gaussian noise | 20% of crops; std up to 0.03 |

All settings were fixed before the run. Validation, stop-set scoring and TTA are unchanged. Everything else matches arm C, including early stopping (patience 10, no minimum epoch). Without `data.aug` the loader draws exactly the same random numbers as before, so older configs give the same crops (checked on 20 crops).

## Metrics and win condition

Metrics are unchanged (frozen `eval/metrics.py`). The headline is per-image CV Dice without TTA, with TTA and Dice by image type reported next to it.

Arm F beats arm C (run `20260925-b3fe577-s02-hires`) if both hold:

- Its mean CV Dice without TTA is at least 0.005 above C's 0.9196.
- It scores at least as high as C on at least 4 of the 5 folds.

If its TTA Dice reaches 0.937, we ask whether to score the locked test set.

## Budget

Augmentation costs about 16 ms per crop on the CPU, which 2 loader workers hide behind the GPU step. Harder training data may push early stopping later, so arm F should take 1.7 to 2.5 hours. If fold 0 takes more than 45 minutes, we stop and ask.

## Outcome

| Arm | Run | CV Dice | Dice with TTA | IoU | Pooled Dice | FGSM Dice | Epochs per fold | Minutes |
|---|---|---|---|---|---|---|---|---|
| C | 20260925-b3fe577-s02-hires | 0.9196 ± 0.0088 | 0.9211 | 0.8588 | 0.9378 | 0.818 | 48/27/36/47/39 | 100 |
| F | 20260928-a6250b2-s04-hires-aug | 0.9213 ± 0.0059 | 0.9227 | 0.8608 | 0.9387 | 0.833 | 60/53/34/34/46 | 123 |

F per fold: 0.9204, 0.9177, 0.9201, 0.9314, 0.9167, which is +0.02, +0.31, −0.18, −0.11 and +0.77 against C.
The mean gain is +0.16 points and F is ahead on 3 of 5 folds, so it fails both parts of the win rule.

Dice by image type:

| Type | C | F |
|---|---|---|
| Fluorescence, small | 0.921 | 0.924 |
| Fluorescence, large | 0.946 | 0.947 |
| Brightfield, grayscale | 0.893 | 0.901 |
| H&E, colour | 0.889 | 0.889 |

The colour and stain changes were meant for H&E, and H&E did not move. The small gains are on fluorescence and
brightfield. Fold 1 no longer stopped early (epoch 53 against 27), and the spread across folds shrank
(std 0.0059 against 0.0088). The prediction grids look like C's: the errors are still thin rims at nucleus edges.

Verdict: within run-to-run noise of C, so C stays the baseline. F has the best TTA Dice so far (0.9227), 1.4
points short of 0.937.
