# 07: Average the predictions of finished runs

Status: done 2026-10-10. No win for either arm: H2 +0.10 points against C*, H4 +0.23. H4's TTA Dice of 0.9235 is the best so far (see Outcome).

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

## Outcome

| Arm | Run | CV Dice | Dice with TTA | IoU | Pooled Dice | Minutes |
|---|---|---|---|---|---|---|
| C* | 20261002-fa7e63a-s06-hires-min30 | 0.9207 ± 0.0081 | 0.9219 | 0.8611 | 0.9390 | 131 |
| H2 | 20261003-512e574-s07-ens2 | 0.9217 ± 0.0076 | 0.9223 | 0.8626 | 0.9396 | 9 |
| H4 | 20261003-512e574-s07-ens4 | 0.9230 ± 0.0069 | 0.9235 | 0.8642 | 0.9403 | 24 |

Per fold:

| Arm | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| C* | 0.9188 | 0.9197 | 0.9233 | 0.9320 | 0.9095 |
| H2 | 0.9184 | 0.9198 | 0.9242 | 0.9331 | 0.9127 |
| H4 | 0.9202 | 0.9202 | 0.9255 | 0.9337 | 0.9155 |

- **Check:** every member alone matched its own run within 0.001 (C* 0.9207, F* 0.9202, C 0.9196, F 0.9213).
- **H2 against C*:** +0.10 points, ahead on 4 of 5 folds (−0.04, +0.01, +0.09, +0.11, +0.32). No win.
- **H4 against C*:** +0.23 points, ahead on all 5 folds (+0.14, +0.05, +0.22, +0.17, +0.60). It passes the fold
  rule but not the 0.5-point bar, so no win.

Averaging helps a little and more members help more, but four runs of one model add 0.23 points, not the 1.5 we
need. With TTA the gain over C* is 0.16. Neither TTA Dice reaches 0.937, so the locked test set stays untouched.

Dice by image type (CV folds only):

| Type | Images | C* | H2 | H4 |
|---|---|---|---|---|
| Fluorescence, small | 391 | 0.921 | 0.922 | 0.924 |
| Fluorescence, large | 101 | 0.947 | 0.947 | 0.948 |
| H&E, colour | 97 | 0.895 | 0.896 | 0.896 |
| Brightfield, grayscale | 14 | 0.908 | 0.912 | 0.912 |

Edge diagnostic (H4): 83% of the wrong pixels lie within 2 pixels of a ground-truth boundary. With that band
ignored, per-image Dice is 0.976 (H&E 0.955, brightfield 0.968, small fluorescence 0.978, large 0.987).

Per-image scores (H4 with TTA, from `per_image.json`): the median is 0.938 and the mean 0.9235. 47 of the 603
images score below 0.85 and cost 1.46 points against 0.937. Three of them are the only 260×347 images among the 603 (the locked test set holds two more).
`7b38c917` has a 26-pixel mask on an image full of nuclei and scores 0.000. `0402a81e` and `df53d0b6` are validated
in fold 4 by models trained on that mask and score 0.008 and 0.364. Without the three, CV Dice with TTA is 0.9275.

Verdict: no win. C* stays the single-model baseline. H4 is the best score to report.
