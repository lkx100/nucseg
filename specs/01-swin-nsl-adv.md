# 01: Swin-T U-Net with NSL adversarial regularization, 5-fold CV

Status: approved 2026-09-24. Nothing has run.

## Question

Does NSL adversarial regularization raise the Dice score of a Swin-T U-Net on DSB2018, measured with 5-fold cross-validation?

This spec also sets up the method every later spec reuses. That covers the frozen splits, the frozen metrics, the model and the training loop.

## Why adversarial regularization is the NSL variant to use

NSL (Neural Structured Learning) adds a second loss term that asks the model to give the same answer for an input and its "neighbours". It offers two kinds of neighbour:

1. Graph regularization. The neighbours are other samples, joined by an explicit graph. DSB2018 has no such graph. We would have to build one from image embeddings, which adds a second model and several choices we can't check yet. For per-pixel masks it is also unclear what the neighbour loss should compare.
2. Adversarial regularization. Each neighbour is the same image with a small change that raises the loss as much as possible. This works for any differentiable model and needs no graph. It is what `nsl.keras.AdversarialRegularization` does.

Adversarial regularization is the right choice. It takes about 20 lines of PyTorch, so we don't need a library. Two caveats shape the design:

- Its main effect is robustness to small input changes, such as staining and noise, plus some regularization. On clean data from the same source, it often gives a small gain or a small loss. With about 430 training images per fold, the regularization may help, but that is the hypothesis, not a given.
- Each step costs about twice as much, because it runs one extra forward and backward pass.

So the spec runs a baseline with the same model, folds and schedule and α = 0. Without it, we can't credit any Dice change to NSL.

### Loss

```
L     = L_sup(f(x), y) + α · L_sup(f(x_adv), y)
x_adv = clip(x + ε · sign(∇_x L_sup(f(x), y)), 0, 1)
L_sup = BCE + soft Dice (equal weights)
```

- ε is measured in raw pixel units in [0, 1], before normalization. The perturbation uses the L∞ norm.
- The gradient comes from the clean loss. `x_adv` is detached, so there are no second-order terms. This matches NSL.
- α = 0.2 is NSL's default multiplier. ε = 2/255 is a small, visually invisible change.

A later spec can try a label-free neighbour loss, the VAT form. It compares f(x) with f(x_adv) instead of comparing f(x_adv) with y, so it could also use unlabelled images.

## Arms

| Arm | α | ε | What changes |
|---|---|---|---|
| A, baseline | 0 | none | nothing, no adversarial pass |
| B, NSL-adv | 0.2 | 2/255 (L∞) | adds the adversarial term |

Everything else is identical, including folds, seeds, schedule and augmentation.

## Data

Source: Kaggle `sindhu9642/nuclei-seg-corrected`. It holds DSB2018 `stage1_train` only, with 670 images and 29,461 nucleus masks. It has no labelled test set.

Target: a binary mask per image, nucleus vs background, the union of all its nucleus masks. Individual nuclei are not scored.

Measured locally on 2026-09-24:

- There are three image types, split by colour and background brightness. 546 are grayscale on a dark background (fluorescence), 108 are colour on a bright background (brightfield and H&E), and 16 are grayscale on a bright background.
- There are 9 image sizes. The smallest is 256×256, which covers 334 images. The largest is 1040×1388.
- Nuclei cover 14% of pixels on average, with a median of 11% and a range of 0.03% to 59%. No image is empty.

Preprocessing converts images to RGB and drops alpha. It normalizes with the fixed ImageNet mean and std, so nothing is fitted on the data.

Training augmentation uses random 256×256 crops, padding smaller images. The crop is taken at the image's original scale. Augmentation also adds random scaling between 0.75 and 1.25, flips, 90° rotations and mild brightness and contrast changes. Validation and test use no augmentation. A later spec moves to 512×512 crops once the pipeline is mature.

## Splits (to be frozen in `eval/`)

1. Leak groups come first. DSB2018 may contain exact duplicates and images cropped from the same larger field of view. If one crop sits in train and another in validation, the model has already seen those nuclei. `eval/make_splits.py` finds exact duplicates by hash, near-duplicates by comparing 32×32 thumbnails, and overlapping crops by finding textured 16×16 patches that appear pixel for pixel in another image. It merges each connected set into a group. On this dataset it found no duplicates and no overlapping crops, but one near-duplicate pair: two frames of the same embryo (thumbnail correlation 0.98, with the next-closest pair below 0.7). A planted crop test confirmed that the overlap check works.
2. The test hold-out takes about 10% of images (about 67), with whole groups only. To keep it fair, it is stratified by image type and by size (small means the longest side is at most 360 px). The script prints its make-up next to the full dataset's. We score it only at milestones and only when asked. Day-to-day decisions use the CV folds.
3. The remaining images (about 603) go into 5 folds with `StratifiedGroupKFold`, using the same strata and groups and seed 0. Each image is in exactly one validation fold.
4. Inside each fold, about 10% of the training images form a stop set, with the same strata and groups. Early stopping watches only this set (see Training).
5. The output is `eval/splits.json`, listing the test ids, the fold of every other id and the stop set of each fold. The script runs once. You review the result, we commit it, and it is then read-only.

Other leak rules:

- Normalization constants are fixed, not computed from data.
- The validation fold never steers training. Early stopping and checkpoint picking use the stop set only. If we picked the best epoch by validation Dice and then reported that same Dice, the CV score would be inflated.
- All hyperparameters are fixed in this spec before the first run.

## Metrics (to be frozen in `eval/metrics.py`)

For each validation image, at its original resolution, we threshold the sigmoid output at 0.5 and compute:

- Dice = 2|P∩G| / (|P| + |G|)
- IoU = |P∩G| / |P∪G|
- Pixel accuracy = correct pixels / all pixels

If both P and G are empty, the score is 1.0.

Reported numbers:

- The headline number averages each metric over the images in a fold, then gives the mean and std over the 5 folds.
- The secondary number is pooled Dice per fold, which sums intersections and areas over all images before dividing. Large images count for more in it, so it can differ from the per-image mean by a point or more.
- Robustness is Dice on the validation fold after an FGSM attack with ε = 2/255 against the model being scored. It checks whether arm B gained the robustness it trained for.

How to read them:

- Dice decides.
- Per image, IoU = Dice / (2 − Dice), so it moves with Dice and adds little.
- Background is about 86% of pixels, so a model that predicts only background already reaches about 0.86 pixel accuracy.

## Model

- The encoder is Swin-T, `swin_tiny_patch4_window7_224` from `timm`, pretrained on ImageNet-1k, with 28M parameters. Pretraining matters here because each fold trains on about 430 images, and transformers trained from scratch on so little data do badly. Later specs can move up to Swin-S or Swin-B if the model's capacity is the limit.
- The decoder is U-Net style. Swin-T outputs feature maps at 1/4, 1/8, 1/16 and 1/32 scale. Each decoder stage upsamples by 2, concatenates the matching encoder map, and applies conv3×3, BN and ReLU twice. A last 4× upsample returns to full resolution, followed by a 1-channel output.
- The input size is 256×256. The pretrained window size of 7 doesn't divide 256 / 4 = 64, so the model uses windows of 8, which make an 8×8 grid of windows. It still loads the pretrained weights, because `timm` resizes the relative-position tables. If that fails on the Kaggle `timm` version, we fall back to `swinv2_tiny_window8_256`, which was pretrained at 256. Windows of 8 also fit 512 crops later.
- Inference slides a window over the full image with 256 tiles and 50% overlap, averaging the logits. No image is resized, so the metrics come from the original pixels.
- For `--smoke`, the same code builds a tiny Swin (embed dim 24, one block per stage, no pretrained weights) that runs on CPU.

## Training

The two arms use identical settings:

- AdamW with lr 1e-4 and weight decay 0.01. The schedule warms up for 3 epochs, then follows cosine decay.
- Batch size is 16, with fp16 AMP. For arm B, the input gradient is normalized by its sign, so the AMP loss scale doesn't change ε.
- One epoch is 4 random crops per training image, about 110 steps.
- Training runs for at most 60 epochs and scores Dice on the stop set after each one. It stops after 10 epochs without improvement and keeps the weights with the best stop-set Dice. The validation fold scores those weights once, at the end.
- Seed 0 for every fold.

## Win condition

Arm B beats arm A if both hold:

- Mean CV Dice improves by at least 0.005.
- B scores at least as high as A on at least 4 of the 5 folds. This is a paired comparison on the same folds.

If B fails, we record "no gain at ε = 2/255, α = 0.2". The next step is then an ε sweep (1, 2 and 4 /255) on fold 0 only, not another full CV.

The project target of Dice ≥ 0.937 is tracked in every summary but is not this spec's win condition. The gap between arm A and 0.937 tells us where the next specs should spend effort.

## Budget

- These are estimates for the full 60 epochs, to be calibrated on arm A's first fold. Early stopping can only lower them. Arm A should take about 20 to 25 minutes per fold on a T4, or about 2 hours for 5 folds. Arm B should take about twice that, or about 4 hours. The total is about 6 of the 30 GPU-hours per week.
- Each arm runs as one Kaggle kernel with its 5 folds in sequence, well under the 12-hour cap.
- Arm A runs first. Arm B launches only after arm A's fold 0 metrics and prediction grid look sane.
- If arm A's first fold takes more than 40 minutes, we stop and ask.

## Steps

1. Write `eval/make_splits.py`, `eval/splits.json` and `eval/metrics.py`. You review them, and then they are frozen.
2. Pin dependencies. `timm` is needed, so check its version on the Kaggle image.
3. Write `src/nucseg/` as data, model, losses (including the adversarial term) and `train.py`, following the job contract. It takes `--fold`, and by default runs all 5 folds in turn. It writes per-fold metrics and the CV mean and std to `metrics.json`.
4. Write `configs/01-swin-t-base.yaml` and `configs/01-swin-t-nsl.yaml`. They differ only in α.
5. Run the smoke test, commit, and launch arm A.

## Outcome

Not run yet.
