# 08: Edge-weighted loss on C*

Status: done 2026-10-10. No win: +0.06 points against C* (see Outcome). Spec 07 found 83% of the wrong pixels within 2 pixels of a nucleus boundary.

## Question

Does counting nucleus-edge pixels more in the loss raise per-image CV Dice of C* (0.9207)?

## Why

The prediction grids show the remaining errors as thin rims at nucleus edges. The loss treats every pixel alike,
so the few edge pixels of a small nucleus weigh little against its interior and the background. Per-image Dice
is the opposite: on an image of small nuclei, a one-pixel rim is a large share of the mask.

The risk is that the rims are label noise. If the annotators' outlines are off by a pixel either way, more
weight on those pixels teaches nothing, and the score stays level or drops.

## Arm

| Arm | Config | What changes against C* |
|---|---|---|
| I, C* + edge-weighted loss | `08-hires-edge.yaml` | adds `loss.edge_weight: 4`, `loss.edge_px: 2` |

The loss stays BCE plus soft Dice. Only the BCE part changes: pixels within 2 pixels of a mask boundary, on
either side, count 5 times as much as the others. The weights are then scaled so their mean is 1, which keeps
the balance between BCE and Dice. Where a mask touches the image border there is no boundary and no extra weight.

Both settings were fixed before the run. The band is the one spec 07 measures. Everything else matches C*:
folds, stop sets, seeds, schedule, model, minimum epoch and TTA. Without `loss` in the config the loss is exactly
as before. The robustness score still attacks with the plain loss.

## Metrics and win condition

Metrics are unchanged (frozen `eval/metrics.py`). The headline is per-image CV Dice without TTA. TTA Dice,
Dice by image type and the epochs per fold are reported next to it.

Arm I beats C* (run `20261002-fa7e63a-s06-hires-min30`) if both hold:

- Its mean CV Dice without TTA is at least 0.005 above C*'s 0.9207.
- It scores at least as high as C* on at least 4 of the 5 folds.

If its TTA Dice reaches 0.937, we ask whether to score the locked test set.

## Budget

The extra weight map costs two pooling steps per batch, so the time per epoch should match C*: about 2.2
GPU-hours for 5 folds. If fold 0 takes more than 40 minutes, we stop and ask.

## Outcome

| Arm | Run | CV Dice | Dice with TTA | IoU | Pooled Dice | FGSM Dice | Epochs per fold | Minutes |
|---|---|---|---|---|---|---|---|---|
| C* | 20261002-fa7e63a-s06-hires-min30 | 0.9207 ± 0.0081 | 0.9219 | 0.8611 | 0.9390 | 0.812 | 43/60/60/49/52 | 131 |
| I | 20261003-e961d17-s08-hires-edge | 0.9213 ± 0.0077 | 0.9229 | 0.8621 | 0.9390 | 0.808 | 43/53/50/51/52 | 123 |

Per fold:

| Arm | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| C* | 0.9188 | 0.9197 | 0.9233 | 0.9320 | 0.9095 |
| I | 0.9204 | 0.9199 | 0.9242 | 0.9315 | 0.9103 |

**I against C*:** +0.06 points, ahead on 4 of 5 folds (+0.16, +0.02, +0.09, −0.05, +0.08). No win. The TTA Dice
of 0.9229 does not reach 0.937, so the locked test set stays untouched.

By image type, brightfield rose from 0.908 to 0.912 (14 images) and the other three types moved by 0.1 points or
less. Fold 0 took 24 minutes, inside the 40-minute limit.

Verdict: no win, and C* stays the baseline. Five times the weight on edge pixels did not move the edge errors.
That fits the risk named above, that the outlines are uncertain by a pixel or two. This run does not prove it.
