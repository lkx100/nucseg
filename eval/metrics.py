"""Frozen metrics (spec 01 onward). Agents must not edit this file; see AGENTS.md.

Ground truth is the union of an image's nucleus masks. Predictions are binarized at
probability 0.5 and scored at the image's original resolution.
"""
from pathlib import Path

import numpy as np
from PIL import Image

THRESHOLD = 0.5


def gt_mask(sample_dir):
    """Union of all nucleus masks in `<sample_dir>/masks/` as a boolean array."""
    mask = None
    for f in sorted(Path(sample_dir, "masks").glob("*.png")):
        m = np.asarray(Image.open(f)) > 0
        mask = m if mask is None else mask | m
    return mask


def binarize(probs):
    """Probabilities in [0, 1] to a boolean mask."""
    return np.asarray(probs) > THRESHOLD


def image_scores(pred, gt):
    """Dice, IoU and pixel accuracy for one image. Both inputs are boolean masks of equal shape."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    inter = int(np.logical_and(pred, gt).sum())
    area = int(pred.sum() + gt.sum())
    union = area - inter
    return {
        "dice": 1.0 if area == 0 else 2 * inter / area,
        "iou": 1.0 if union == 0 else inter / union,
        "pixacc": float((pred == gt).mean()),
        "inter": inter,
        "area": area,
    }


def fold_summary(scores):
    """Per-image means over one fold, plus pooled Dice (intersections and areas summed first)."""
    area = sum(s["area"] for s in scores)
    return {
        "dice": float(np.mean([s["dice"] for s in scores])),
        "iou": float(np.mean([s["iou"] for s in scores])),
        "pixacc": float(np.mean([s["pixacc"] for s in scores])),
        "dice_pooled": 1.0 if area == 0 else 2 * sum(s["inter"] for s in scores) / area,
        "n_images": len(scores),
    }


def cv_summary(folds):
    """Mean and sample std (ddof=1) over the fold summaries. The mean of `dice` is the headline number."""
    out = {}
    for key in ("dice", "iou", "pixacc", "dice_pooled"):
        vals = np.array([f[key] for f in folds])
        out[key] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out
