# 10: Where arm J still loses points, photo by photo

Status: done 2026-10-12. The gap sits in how wide the model draws the nuclei of a photo, mostly on small fluorescence photos (see Outcome).

## Question

What does J's best fold still get wrong: a few photos that fail, or a thin ring around almost every nucleus?

## Why

J is 0.32 points short on the highest fold (93.36 against 93.68) and 0.06 short on the mean. Spec 09 showed that the fold with the highest score gained nothing from dropping the wrong label, so the next idea has to make a good fold better. The last five training ideas were chosen without looking at single photos, and all landed inside the run-to-run noise. The one look at single photos (spec 07) found the wrong label.

J has no per-photo scores. `nucseg.train` writes fold means only.

## Arm

| Arm | Config | What it does |
|---|---|---|
| K, J scored photo by photo | `10-j-per-image.yaml` | Loads J's five saved models, scores each fold's validation photos with and without TTA, and saves the TTA mask of every photo |

No training and no new model. It reuses `nucseg.ensemble` with J as the only member, so nothing is averaged. Two small additions to that entry point: `data.exclude` (so `7b38c917` stays out, as in J) and `eval.save_masks`. `eval/` is not edited.

Check that the kernel reproduces J: Dice 92.59 and 92.75 with TTA, within 0.01 points (the saved weights are half precision).

## What we read from it

The kernel gives `per_image.json` (Dice, Dice with TTA, wrong pixels, wrong pixels within 2 px of an outline) and 602 masks. The rest is done on this machine from the masks and the hand-drawn nuclei, with no GPU:

1. The fold table of the photo scores: median, number of photos below 0.85, and what they cost, per fold and per photo type.
2. For every photo, three kinds of wrong pixel: whole nuclei the model missed, blobs it invented, and the ring along outlines it did find. A nucleus counts as missed when less than half of it is marked.
3. A picture sheet of fold 3's lowest photos and of the lowest photos overall: photo, hand-drawn answer, model's answer.

## How the answer is used

This is a diagnostic, so there is no win rule. The reading decides the next spec:

- **Missed or invented nuclei carry the gap.** If fold 3 would reach 93.68 with its missed and invented nuclei fixed and the rings left alone, the next run aims at those photos (by type), or tries a larger encoder (Swin-S).
- **Rings carry the gap.** If fold 3 stays under 93.68 even then, a single model is close to what the hand-drawn outlines allow, and the next run averages J with models that differ from it.

Either way the picture sheet goes to the user before the next spec is written.

## Budget

One model over 602 photos, twice. H2 (two models, 603 photos) took 9 minutes, so about 5 to 10 GPU-minutes. If the kernel runs longer than 30 minutes, we stop and ask.

## Outcome

Run `20261012-c4b6e50-s10-j-per-image`, 6 minutes. It reproduces J exactly: 92.59, 92.75 with TTA, highest fold 93.36. The numbers below come from `uv run python reports/photo_errors.py runs/20261012-c4b6e50-s10-j-per-image`, all with TTA.

### Three kinds of wrong pixel

89% of the wrong pixels are rings along outlines the model did find. 4% belong to nuclei it missed (813 of 25,983) and 7% to blobs it invented (1,000).

| | Mean of five folds | Highest fold |
|---|---|---|
| Target | 92.81 | 93.68 |
| J as it is | 92.75 | 93.36 |
| Missed and invented nuclei repaired, rings left | 93.76 | 94.12 |
| Rings repaired, missed and invented left | 98.96 | 99.28 |

By the rule set above, fold 3 with its nuclei repaired reaches 93.68, so the first branch applies on paper. The rule was too coarse, though. Repairing every missed and invented nucleus in fold 3 gives 0.75 points and we need 0.32, so 43% of them would have to go. The rings cost 5.9 points in that fold, so 5% of them would do. Both routes are open, and the pictures point to the rings.

### The rings are one-sided

In the low photos the ring has one colour: the model draws every nucleus of the photo wider than the hand-drawn answer, or every one narrower.

- Of the 46 photos below 0.85, 38 have three quarters or more of their wrong pixels on one side: 23 too wide, 15 too narrow.
- Over all 602 photos it is 208: 112 too wide, 96 too narrow. On average the model is neither. The answers differ from photo to photo in how generously the nuclei were outlined, and the model draws a middle width.
- A ceiling: grow or shrink each photo's whole mask by the best of −2 to 2 pixels, chosen with the answer in hand. 104 photos want a change. The mean rises to 93.41 and the highest fold to 93.86, both over the target. This is not a method, because it reads the answer. It says how much the right width per photo is worth: 0.66 points on the mean, 0.50 on the best fold.
- Almost all of that is in the small fluorescence photos (390 of 602): 92.99 as it is, 93.90 at the best width. H&E gains 0.16 and stays the lowest type at 89.85.

| Fold | As it is | Best width per photo | Nuclei repaired |
|---|---|---|---|
| 0 | 92.81 | 93.52 | 93.91 |
| 1 | 92.12 | 92.84 | 93.49 |
| 2 | 92.58 | 93.38 | 93.35 |
| 3 | 93.36 | 93.86 | 94.12 |
| 4 | 92.89 | 93.48 | 93.93 |

Picture sheets: `reports/6-lowest-photos-best-fold.png` and `reports/6-lowest-photos-all.png`.

### What it does not tell us

Whether the width can be predicted. If photos with wide answers look different from photos with narrow ones (brightness, blur, nucleus size, source batch), a model can learn it. If look-alike photos carry different widths, no model can, and the last half point is the people who drew the outlines. That check needs no GPU: compare each photo's best width with what the photo looks like and with its neighbours in the training folds.

Verdict: a diagnostic, so no win or loss. J stays the baseline. The next spec starts from the width question.
