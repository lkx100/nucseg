# 08: Edge-weighted loss on C*

Status: approved 2026-10-03, running. Spec 07 found 83% of the wrong pixels within 2 pixels of a nucleus boundary.

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
