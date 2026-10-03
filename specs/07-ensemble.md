# 07: Average the predictions of finished runs

Status: approved 2026-10-03, running.

## Question

Does averaging the predictions of models we already trained raise per-image CV Dice above C* (0.9207)?

## Why

Every training change since arm C has landed between 0.919 and 0.921. Two models trained differently make
different mistakes at nucleus edges, and averaging their probabilities cancels part of that. It needs no
training: every finished kernel still holds its five fold checkpoints on Kaggle.

The members of an arm were all trained on the same folds. For fold k each member's fold-k model predicts the
fold's validation images, which none of them saw in training or in early stopping. So the score is as leak-free as
a single model's.

## Arms

| Arm | Config | Members |
|---|---|---|
| H2 | `07-ens2.yaml` | C*, F* (spec 06) |
| H4 | `07-ens4.yaml` | C*, F*, C (spec 02), F (spec 04) |

Each member predicts probabilities as before (overlapping 256 tiles; with TTA, 8 orientations). The members'
probabilities are averaged with equal weights and then cut at 0.5. Nothing is tuned on the validation folds.
H4 adds the two older runs of the same model to see whether more members keep helping.

The code is `src/nucseg/ensemble.py`. Checkpoints are stored in half precision, so a member alone may differ from
its recorded score in the fourth decimal.

## Check before reading the result

Each member is also scored alone. Its mean CV Dice must match its own run within 0.001 (C* 0.9207, F* 0.9201,
C 0.9196, F 0.9213). If it doesn't, the weights or the fold pairing are wrong and the result is void.

## Edge diagnostic

The same run measures how much of the remaining error sits at nucleus edges, for the planned edge-weighted loss:

- the share of wrong pixels that lie within 2 pixels of the ground-truth boundary;
- per-image Dice with that 2-pixel band ignored.

These are diagnostics only. They are computed outside `eval/` and decide nothing here.

## Metrics and win conditions

Metrics are unchanged (frozen `eval/metrics.py`). The headline is per-image CV Dice without TTA. TTA Dice and
Dice by image type are reported next to it. The robustness score is skipped.

Each arm is compared with C* (run `20261002-fa7e63a-s06-hires-min30`) by the usual rule: its mean CV Dice
without TTA is at least 0.005 higher, and it scores at least as high on at least 4 of the 5 folds.

A winning arm does not replace C* as the single-model baseline. Later specs still change one thing against C*,
and a better single model can then join the ensemble.

If either arm's TTA Dice reaches 0.937, we ask whether to score the locked test set.

## Budget

No training. Each kernel loads the data and runs 2 or 4 models over 603 images, with and without TTA: about 10
to 20 GPU-minutes each, 0.5 GPU-hours together at most. If a kernel runs longer than 45 minutes, we stop and ask.
