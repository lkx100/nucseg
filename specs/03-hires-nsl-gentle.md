# 03: NSL at ε 1/255 on the full-resolution-skip model

Status: approved 2026-09-28, running.

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

Pending.
