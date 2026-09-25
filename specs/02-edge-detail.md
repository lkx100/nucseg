# 02: Finer nucleus edges for the Swin-T U-Net

Status: approved 2026-09-25. Running A′ and C.

## Question

Does giving the model finer detail at nucleus edges raise per-image Dice above spec 01's baseline (0.9178)?

## Why edges

The baseline is 1.9 points short of the 0.937 target. Most of its errors on fluorescence images, which are 81% of the data, are thin rims one or two pixels wide at nucleus edges. These images are small, 256×256 to 360×360, with nuclei 10 to 40 pixels across. On a nucleus that size, a one-pixel rim costs several points of Dice.

The model is coarse at exactly that scale. Swin's first feature map is at 1/4 resolution, and the decoder's last two stages upsample from there with no finer input. So the model sees edges through 4×4-pixel patches and has to guess them back.

This spec tests two separate fixes. Each changes one thing against the baseline:

- **Arm C, full-resolution skip.** A small convolution stem reads the image at full and half resolution and feeds those features into the decoder's last two stages. This is the standard fix in Swin-UNETR-style models. It adds about 10% compute.
- **Arm D, 2× input scale.** The model upsamples each 256×256 crop to 512×512 inside its forward pass, predicts at that size, and averages the logits back down to 256. Nuclei look twice as large to the whole network, so each 4×4 patch covers 2×2 original pixels. This is the 512 step you planned. The model sees 512×512 inputs that cover the same 256×256 of original image, which matters because most images are only 256 to 360 pixels wide. It costs about 4 times the compute.

## Arms

| Arm | Config | What changes against arm A |
|---|---|---|
| A′, baseline rerun | `02-base.yaml` | nothing; spec 01 arm A on the new code |
| C, full-res skip | `02-hires.yaml` | `model.hires_skip: true` |
| D, 2× scale | `02-scale2.yaml` | `model.input_scale: 2`, and batch 8 × 2 accumulation steps |

Why rerun the baseline (A′): it confirms the new code didn't change the baseline, and it measures run-to-run noise. The difference between A and A′ shows how much two identical runs differ. It also gives the baseline's test-time augmentation score (see Metrics).

Arm D can't fit 16 crops at 512×512 on a T4, since arm A already used 5.7 GB at 256. So it runs batches of 8 and adds the gradients of two batches before each update. Each update still sees 16 crops. BatchNorm sees 8 at a time, which is a small unavoidable difference.

Everything else is identical to spec 01 arm A: folds, stop sets, seeds, schedule, augmentation, early stopping, loss, and no NSL.

## Metrics

These are unchanged from spec 01, using the frozen `eval/metrics.py`. Each arm also reports test-time augmentation (TTA) scores. TTA predicts each validation image in 8 orientations (4 rotations, each with and without a flip), turns the predictions back, and averages them. TTA only changes how predictions are made, not the metric. We fix it now, before any run, so choosing it later isn't tuning on the validation folds.

- The headline, used for the win condition, is per-image Dice without TTA, so it compares directly with spec 01.
- The same Dice with TTA is reported next to it for every arm. For the 0.937 target, the best arm's TTA Dice counts, because TTA is part of how that model predicts.
- Stop-set scoring during training stays without TTA, to keep epochs fast.

## Win condition

Arm C or D beats the baseline if both hold:

- Its mean CV Dice without TTA is at least 0.005 above arm A′.
- It scores at least as high as A′ on at least 4 of the 5 folds.

If both arms win, the higher mean wins. Their changes can be combined in spec 03.

If any arm's CV Dice with TTA reaches 0.937, that's a milestone. We then ask you whether to score the locked test set.

## Budget

These are estimates for the full 60 epochs. Early stopping can only lower them.

| Arm | Estimate |
|---|---|
| A′ | about 1.5 h (arm A took 90 minutes) |
| C | about 1.7 h |
| D | about 5 to 7 h, from 4 times the pixels per step |
| TTA scoring | about 1 to 5 minutes per fold on top |

The total is about 8 to 10 GPU-hours. Spec 01 used about 5, so the week's total stays under 30.

Order: A′ and C first, then D. If D's first fold takes more than 90 minutes, we stop and ask. D needs the whole kernel to fit in the 12-hour cap.

## Steps

1. Model options: `model.hires_skip` (convolution stem feeding the last two decoder stages) and `model.input_scale` (upsample inside the model, average the logits back down). Both default to off, so spec 01 configs behave exactly as before.
2. `train.accum_steps` (default 1) for gradient accumulation.
3. `eval.tta` (default false). When it's on, the validation fold is also scored with 8-orientation TTA, and `metrics.json` and `summary.txt` carry the TTA numbers.
4. Three configs, then smoke-test all three on CPU, commit, and launch in the order above.

## Outcome

Not run yet.
