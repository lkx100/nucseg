# 11: Does a different kind of attention help the Swin-T U-Net?

Status: running since 2026-10-17. Four arms, one Kaggle kernel each, two at a time (Kaggle allows two GPU sessions per account).

## The model is settled

The user decided on 2026-10-17: arm J (`configs/09-hires-plain-cv.yaml`, run `20261011-1a5ff70-s09-plain-cv`) is our model. It is a Swin-T encoder with a U-Net decoder and the full-resolution skip, trained under plain 5-fold CV without image `7b38c917`. Every arm below changes one attention piece of it and nothing else.

J, per-image Dice with TTA: mean 92.75 ± 0.40, highest fold 93.36. The paper: mean 92.81 ± 0.87, row "Highest" 93.68 (that row is mean + std, see `09-plain-cv.md`).

## Question

Does any of four other kinds of attention, in the encoder or the decoder, lift J's score by more than the luck band?

## Why these four

Spec 10 found where J still loses points. About half a point is the hand-drawn outlines being wider or narrower from photo to photo, and no model that reads only the photo can recover it. What is left is whole nuclei the model misses or invents (worth up to 1.0 point on the mean, 0.75 in the best fold) and H&E photos, the lowest type (89.85). Both need the model to judge "is this blob a nucleus" from its surroundings, which is what attention does. The two decoder arms change how skip features are let through; the two encoder arms change how far each layer can look.

| Arm | Config | Slug | The one change against J |
|---|---|---|---|
| L, attention gates | `11-gate.yaml` | `s11-gate` | An additive attention gate (Oktay et al. 2018, as in the reference paper's decoder) on each of the five skip connections. The decoder decides pixel by pixel how much of each skip passes. |
| M, scSE | `11-scse.yaml` | `s11-scse` | Channel and spatial squeeze-excitation (Roy et al. 2018) after each of the five decoder blocks. |
| N, Swin V2 | `11-swinv2.yaml` | `s11-swinv2` | Encoder `swinv2_tiny_window8_256`: cosine attention with a learned temperature and a log-spaced position bias, in place of Swin V1's dot product. Pretrained at 256 px with window 8, our exact crop, where J's Swin V1 was pretrained at 224 with window 7 and resized. |
| O, MaxViT | `11-maxvit.yaml` | `s11-maxvit` | Encoder `maxvit_rmlp_tiny_rw_256`: each layer attends inside an 8×8 block and also across a sparse 8×8 grid that spans the whole crop, so every layer sees global context. It also has conv blocks (MBConv) that Swin lacks. |

All four encoders are ImageNet-1k weights and 31.6 to 31.9 M parameters with the decoder, so model size does not explain a difference. Arms N and O change more than the attention type (pretraining size; conv blocks), as noted above. Without those changes they could not run at all, so they are accepted.

Everything else is J's: folds, seed, schedule, augmentation, minimum epoch 30, patience 10, TTA. Configs without the new keys build the same model as before; J's saved weights still load.

## Win rule

Per-image Dice with TTA, the same table as J (`uv run python -m nucseg.foldtable runs/<run_id>`).

- **Better than J:** five-fold mean at least 93.10 (J + 0.35, the luck band). One seed per arm, so anything less is not evidence.
- **Reaches the target:** mean at least 92.81 and highest fold at least 93.68. An arm that reaches it but stays under 93.10 gets a second seed before we claim it.
- The best arm becomes the model. If two decoder arms both gain, the next spec combines them with the best encoder.

Expectation, written before the runs: small gains at most. Eight single-model runs since spec 02 sit inside one luck band.

## Budget

J took 132 minutes on one Kaggle GPU. L and M add a few small layers: about 2.5 hours each. N's attention is a little heavier: about 3 hours. O is the heaviest: 3 to 4.5 hours. About 12 GPU-hours in total, of the 30 a week. If an arm has not finished after 8 hours, we stop it and ask.

## Outcome

The first push of L and M (2026-10-17) waited about three days for a GPU while Kaggle reported it as RUNNING, then ended in ERROR with an empty log. Nothing of ours ran. Pushed again 2026-10-20; they waited 80 minutes in the queue and then ran.

| Arm | Run | TTA mean ± std | Highest | Mean + std | Verdict |
|---|---|---|---|---|---|
| J (baseline) | `20261011-1a5ff70-s09-plain-cv` | 92.75 ± 0.40 | 93.36 | 93.16 | |
| L, attention gates | `20261020-90abcd5-s11-gate` | 92.71 ± 0.50 | 93.56 | 93.20 | no change: −0.04 |
| M, scSE | `20261020-90abcd5-s11-scse` | 92.86 ± 0.37 | 93.43 | 93.23 | no change: +0.11, inside the luck band |

- **M, scSE:** ahead of J on all five folds, by 0.05 to 0.19 points. That is too small to call, since a new seed alone moves J by up to 0.35. H&E photos, which the attention was meant to help, stayed where they were (89.76 against J's 89.85).
- **L, attention gates:** behind J on four folds by about 0.1 and ahead on fold 4 by 0.20, which lifts the highest fold to 93.56, still 0.12 under the paper's 93.68. The mean does not move. The reference paper's decoder uses these gates; on our model they add nothing.
- N, O: pending.
