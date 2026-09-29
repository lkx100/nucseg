# 05: The full-resolution-skip model at 2× input scale

Status: running.

## Question

Does 2× input scale raise per-image CV Dice of arm C, the spec 02 winner (0.9196)?

## Why

In spec 02, the full-resolution skip (arm C) and 2× input scale (arm D) scored about the same (0.9196 and 0.9190), but on different images. C gained most on small fluorescence (0.921 against D's 0.919). D gained on brightfield (0.912 against C's 0.893) and H&E (0.894 against 0.889). Brightfield and H&E are the weakest types, and spec 04's augmentation didn't move H&E. If the two gains come from different errors, putting both in one model should add up.

## Arm

| Arm | Config | What changes against arm C |
|---|---|---|
| G, C at 2× scale | `05-hires-scale2.yaml` | `model.input_scale: 2` |

The conv stem now reads the upsampled 512×512 input, so its "full resolution" is twice the image's own resolution.

Arm D trained in batches of 8 with 2 accumulation steps, because we thought 16 crops at 512 wouldn't fit on a T4. A CPU count of stored activations shows that was wrong. Training needs about 0.6 GB per crop, so batch 16 comes to about 10 GB. D's 13.7 GB peak came from the FGSM robustness step, which runs 16 tiles at full precision with gradients. So arm G trains at batch 16 with no accumulation, exactly like C, and the input scale is the only change.

The FGSM step for G would need about 15.6 GB at 16 tiles, so it runs 8 tiles at a time (`eval.robust_batch: 8`, new, default 16). Its loss is averaged per group of tiles, so images with 8 tiles or fewer get exactly the same attack as before. Only large images weight their tiles a little differently. FGSM Dice is a side score, so this doesn't touch the win rule.

Everything else matches arm C: folds, stop sets, seeds, schedule, augmentation (no `data.aug`), early stopping (patience 10, no minimum epoch) and TTA.

## Metrics and win condition

Metrics are unchanged (frozen `eval/metrics.py`). The headline is per-image CV Dice without TTA. TTA Dice and Dice by image type are reported next to it.

Arm G beats arm C (run `20260925-b3fe577-s02-hires`) if both hold:

- Its mean CV Dice without TTA is at least 0.005 above C's 0.9196.
- It scores at least as high as C on at least 4 of the 5 folds.

If its TTA Dice reaches 0.937, we ask whether to score the locked test set.

## Budget

Arm D took 285 minutes (fold 0: 73 minutes for 51 epochs). The stem at 512 adds some compute, so arm G should take 5 to 6 hours, and at most about 7.5 hours if every fold runs all 60 epochs. That fits the 12-hour cap. If fold 0 takes more than 100 minutes, we stop and ask.
