# 03: NSL at ε 1/255 on the full-resolution-skip model

Status: done 2026-09-28. No win: +0.01 points, 3 of 5 folds. NSL stops here for the Dice target (see Outcome).

## Question

Does a gentler NSL adversarial term (α 0.2, ε 1/255) raise per-image CV Dice of arm C, the spec 02 winner (0.9196)?

## Why a gentler ε

In spec 01, NSL at ε 2/255 made the model far more robust (FGSM Dice 0.800 to 0.918) but did not raise clean Dice. It scored 0.34 points below arm A, which is inside the 0.3-point run-to-run noise that arm A′ later measured. A smaller step keeps the adversarial neighbours closer to real images, so the extra loss term acts more like a regularizer and less like a shift in the training data. If NSL helps clean Dice at all, a smaller ε is where it should show.

This goes straight to ε 1/255 and skips the ε 2/255 arm on top of C. If this arm also fails, we record "NSL gives robustness, not accuracy" and stop spending GPU hours on NSL for the Dice target.

## Arm

| Arm | Config | What changes against arm C |
|---|---|---|
| E, C + NSL gentle | `03-hires-nsl1.yaml` | `nsl.alpha: 0.2`, `nsl.eps_255: 1` |

The two NSL values count as one change, "turn on NSL at ε 1/255". Everything else matches arm C: model, folds, stop sets, seeds, schedule, augmentation, early stopping and TTA. Early stopping keeps its spec 02 settings (patience 10, no minimum epoch), so the comparison with C stays clean, even though it cut fold 1 short in every spec 02 arm.

The robustness score stays FGSM at 2/255, so it compares with spec 01 and 02.

## Metrics and win condition

Metrics are unchanged (frozen `eval/metrics.py`). The headline is per-image CV Dice without TTA, with TTA reported next to it.

Arm E beats arm C (run `20260925-b3fe577-s02-hires`) if both hold:

- Its mean CV Dice without TTA is at least 0.005 above C's 0.9196.
- It scores at least as high as C on at least 4 of the 5 folds.

If its TTA Dice reaches 0.937, we ask whether to score the locked test set.

## Budget

Spec 01's NSL arm took 2.4 times its baseline (215 against 90 minutes). Arm C took 100 minutes, so arm E should take about 4 hours on a T4. That is one kernel, well under the 12-hour cap. If fold 0 takes more than 75 minutes, we stop and ask.

## Outcome

| Arm | Run | CV Dice | Dice with TTA | IoU | Pooled Dice | FGSM Dice | Epochs per fold | Minutes |
|---|---|---|---|---|---|---|---|---|
| C | 20260925-b3fe577-s02-hires | 0.9196 ± 0.0088 | 0.9211 | 0.8588 | 0.9378 | 0.818 | 48/27/36/47/39 | 100 |
| E | 20260928-f8051ad-s03-hires-nsl1 | 0.9197 ± 0.0076 | 0.9212 | 0.8593 | 0.9379 | 0.905 | 39/30/53/31/53 | 227 |

E per fold: 0.9188, 0.9171, 0.9240, 0.9295, 0.9092, which is −0.14, +0.25, +0.21, −0.30 and +0.02 against C.
The mean is level with C (+0.01) and E is ahead on 3 of 5 folds, so it fails both parts of the win rule.

Dice by image type moved by at most 0.3 points on fluorescence and 1.0 on brightfield (C against E: small
fluorescence 0.921 and 0.920, large fluorescence 0.946 and 0.946, brightfield 0.893 and 0.903, H&E 0.889 and 0.892).
Robustness rose as in spec 01: Dice under FGSM 2/255 is 0.905 against C's 0.818. Fold 1 stopped early again
(epoch 30, best 20). The prediction grids look like C's.

Verdict: NSL gives robustness, not accuracy. Both ε 2/255 (spec 01) and ε 1/255 (here) leave clean Dice within
run-to-run noise and cost 2.3 to 2.4 times the GPU time. Per the plan, we stop spending GPU hours on NSL for
the Dice target.
