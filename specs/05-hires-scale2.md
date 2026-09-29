# 05: The full-resolution-skip model at 2× input scale

Status: done 2026-09-30. No win: +0.00 points, 4 of 5 folds (see Outcome).

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

## Outcome

| Arm | Run | CV Dice | Dice with TTA | IoU | Pooled Dice | FGSM Dice | Epochs per fold | Minutes |
|---|---|---|---|---|---|---|---|---|
| C | 20260925-b3fe577-s02-hires | 0.9196 ± 0.0088 | 0.9211 | 0.8588 | 0.9378 | 0.818 | 48/27/36/47/39 | 100 |
| D | 20260925-cf38771-s02-scale2 | 0.9190 ± 0.0082 | 0.9197 | 0.8579 | 0.9380 | 0.833 | 51/28/46/36/41 | 285 |
| G | 20260929-e7415b1-s05-hires-scale2 | 0.9196 ± 0.0109 | 0.9206 | 0.8588 | 0.9376 | 0.824 | 13/35/55/42/47 | 343 |

G per fold: 0.9070, 0.9221, 0.9256, 0.9334, 0.9102, which is −1.33, +0.74, +0.37, +0.09 and +0.13 against C.
G is ahead on 4 of 5 folds, but its mean is level with C (+0.00), so it fails the 0.5-point part of the win rule.

Fold 0 decides the result. Its stop-set Dice peaked at 0.913 in epoch 3, still in warm-up, and no later epoch beat
that peak within the 10-epoch patience, so it stopped at epoch 13 and kept the epoch-3 weights. On the other four folds G averages
+0.33 against C. Fold 1 went the other way: its stop-set Dice dropped from 0.935 to about 0.90 after epoch 25 and
stayed there, and the fold kept its epoch-25 weights.

Dice by image type:

| Type | Images | C | D | G |
|---|---|---|---|---|
| Fluorescence, small | 434 | 0.921 | 0.919 | 0.920 |
| Fluorescence, large | 112 | 0.946 | 0.946 | 0.947 |
| H&E, colour | 108 | 0.889 | 0.894 | 0.892 |
| Brightfield, grayscale | 16 | 0.893 | 0.912 | 0.908 |

G keeps most of D's brightfield and H&E gains, but not C's small-fluorescence gain. Brightfield is only 16 images,
so its swings are noisy. The prediction grids look like C's: the errors are thin rims at nucleus edges.

Peak GPU memory was 15.2 GB, just inside the T4's limit, well above the 10 GB estimated for training. The peak
is logged for the whole run, so it doesn't show which step set it. The run took 343 minutes, 3.4 times C.

Verdict: C stays the baseline. The 2× scale is too costly for what it gives. Early stopping has now cut one fold
short before epoch 30 in 4 of the 6 runs since spec 02 began (A′, C, D and G), so the next spec should fix it: stop early only after a minimum
number of epochs, or train a fixed number of epochs.
