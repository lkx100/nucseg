# 12: Does arm J score better if it also sees the whole photo at once?

Status: running.

## Question

Arm J predicts a photo from 256 px tiles at half overlap. Does one pass over the whole photo, alone or averaged with the tiles, change its Dice?

## Why

The user asked whether SAHI tiled inference (Ultralytics guide) suits us. SAHI cuts large photos into tiles so a detector doesn't shrink them, then also runs the detector once on the whole photo by default. We already tile at full size; the whole-photo pass is the only part we don't do. The run answers "did you try SAHI-style inference?" with a number. SAHI's own model (YOLO) answers a different question, boxes per nucleus, so it is out of scope.

334 of the 670 photos are exactly 256×256. For those, the whole photo is the one tile, and the three ways must give the same answer. Only the larger photos can change.

## Arm

| Arm | Config | What it does |
|---|---|---|
| P, J scored three ways | `12-whole-photo.yaml` | Loads J's five saved models and scores each fold's validation photos as tiles, as the whole photo, and as the mean of the two logit maps, each with and without TTA |

No training and no new weights. The Swin encoder is built with `strict_img_size: false`, so timm rebuilds the shifted-window attention mask for each input size; the mask is not stored in the checkpoint, so J's weights load as they are. The whole photo is reflect-padded at the bottom and right to a multiple of 256, so no attention window holds zero padding. `eval/` is not edited.

Check: the tiles view must reproduce J, 92.59 and 92.75 with TTA. If it doesn't, the run is void.

## Win rule

Per-image Dice with TTA, in the paper's fold table.

- **Better than tiles:** "whole" or "both" at least 0.35 above 92.75 on the mean, the luck band of a training run. The models are the same here, so a smaller gain on all 5 folds is real but small, and we report it as such.
- **Reaches the target:** mean at least 92.81 and mean + std at least 93.68.

Expectation, written before the run: "both" within +0.00 to +0.05 of tiles, "whole" level or a little lower, since J only saw 256 px crops while training.

## Budget

Spec 10 scored one model with and without TTA in 6 minutes. Three views is about 3 times that, plus the larger whole-photo passes: about 15 to 25 GPU-minutes. If the kernel runs longer than 45 minutes, we stop and ask.
