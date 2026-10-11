# 10: Where arm J still loses points, photo by photo

Status: approved 2026-10-12, running.

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

Not run yet.
