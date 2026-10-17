"""One picture for a non-specialist audience: why the hand-drawn answers limit the score. Needs no GPU.

    uv run python reports/answer_key.py runs/<run_id with saved masks>

Writes reports/8-answer-key.png. Four panels: what one pixel of tracing costs, two look-alike photos traced with
different habits, a plainly wrong answer, and the score if each photo's outline width matched its answer (a ceiling
from photo_errors.json, which reads the answer, so not a method).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nucseg.data import find_data_root, load_samples  # noqa: E402
from eval.metrics import image_scores  # noqa: E402

PAPER_MEAN, PAPER_HIGH = 92.81, 93.68
PAIR = ("573a657d", "bfe8ef19")    # look-alike small fluorescence photos, model +2.0 px and -1.8 px against the answer
BAD, LOOKALIKE = "7b38c917", "0402a81e"
GT_C, MODEL_C, OURS_C, PAPER_C, GREY = "#f2c200", "#00c8ff", "#2a6fdb", "#8a8a8a", "#c9c9c9"


def full_id(root, short):
    return next(p.name for p in Path(root).iterdir() if p.name.startswith(short))


def bright(img):
    show = img.astype(float)
    return np.clip(show / max(1.0, np.percentile(show, 99.8)) * 255, 0, 255).astype(np.uint8)


def busiest_window(gt, size=72, step=8):
    """Top-left corner of the window holding the most nucleus pixels away from its edge."""
    best, corner = -1, (0, 0)
    for y in range(0, gt.shape[0] - size + 1, step):
        for x in range(0, gt.shape[1] - size + 1, step):
            v = gt[y + 8:y + size - 8, x + 8:x + size - 8].sum()
            if v > best:
                best, corner = v, (y, x)
    return corner


def main(run):
    run = Path(run)
    root = find_data_root(None)
    rows = json.loads((run / "photo_errors.json").read_text())
    cross = ndimage.generate_binary_structure(2, 1)

    # what the answer key scores against itself when traced one pixel wider or narrower
    wider, narrower = [], []
    for r in rows:
        gt = load_samples(root, [r["id"]])[r["id"]][1]
        wider.append(image_scores(ndimage.binary_dilation(gt, cross), gt)["dice"])
        narrower.append(image_scores(ndimage.binary_erosion(gt, cross), gt)["dice"])
    folds = range(5)
    ours = [100 * np.mean([r["dice"] for r in rows if r["fold"] == k]) for k in folds]
    matched = [100 * np.mean([r["dice_best_size"] for r in rows if r["fold"] == k]) for k in folds]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.18)

    # A: one pixel
    ax = fig.add_subplot(gs[0, 0])
    bars = [("Answer key vs itself,\ntraced 1 pixel narrower", 100 * np.mean(narrower), GREY),
            ("Answer key vs itself,\ntraced 1 pixel wider", 100 * np.mean(wider), GREY),
            ("Our model", np.mean(ours), OURS_C),
            ("Reference paper", PAPER_MEAN, PAPER_C)]
    y = np.arange(len(bars))[::-1]
    ax.barh(y, [b[1] for b in bars], color=[b[2] for b in bars], height=0.6)
    for yy, (_, v, _) in zip(y, bars):
        ax.text(v + 0.1, yy, f"{v:.2f}", va="center", fontsize=11)
    ax.set_xlim(86, 94)
    ax.set_yticks(y, [b[0] for b in bars], fontsize=11)
    ax.set_xlabel("Dice score against the hand-drawn answer (%), average over all 602 photos")
    ax.set_title("A. One pixel of tracing costs about 9 points.\nOur model and the paper are 0.06 points apart.",
                 loc="left", fontsize=13, weight="bold")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # B: two look-alike photos, two tracing habits
    sub = gs[0, 1].subgridspec(1, 2, wspace=0.05)
    for n, short in enumerate(PAIR):
        i = full_id(root, short)
        img, gt = load_samples(root, [i])[i]
        pred = np.asarray(Image.open(run / "masks" / f"{i}.png")) > 0
        yy, xx = busiest_window(gt)
        s = np.s_[yy:yy + 72, xx:xx + 72]
        a = fig.add_subplot(sub[0, n])
        a.imshow(bright(img)[s], interpolation="nearest")
        a.contour(gt[s], levels=[0.5], colors=GT_C, linewidths=2)
        a.contour(pred[s], levels=[0.5], colors=MODEL_C, linewidths=1.5, linestyles="--")
        a.set_xticks([]), a.set_yticks([])
        a.set_xlabel("answer traced tightly" if n == 0 else "answer traced generously", fontsize=11)
        if n == 0:
            a.set_title("B. Same kind of photo, two tracing habits.\nYellow = hand-drawn answer, blue dashes = our "
                        "model.", loc="left", fontsize=13, weight="bold")

    # C: a plainly wrong answer next to a look-alike with a right one
    sub = gs[1, 0].subgridspec(2, 2, wspace=0.05, hspace=0.25)
    for n, short in enumerate((BAD, LOOKALIKE)):
        i = full_id(root, short)
        img, gt = load_samples(root, [i])[i]
        a, b = fig.add_subplot(sub[n, 0]), fig.add_subplot(sub[n, 1])
        a.imshow(bright(img)), b.imshow(gt, cmap="gray")
        for ax_ in (a, b):
            ax_.set_xticks([]), ax_.set_yticks([])
        a.set_ylabel("wrong answer\n(dropped)" if n == 0 else "look-alike photo,\nright answer", fontsize=11)
        b.set_xlabel(f"answer: {int(gt.sum()):,} nucleus pixels", fontsize=10)
        if n == 0:
            a.set_title("C. Some answers are plainly wrong.", loc="left", fontsize=13, weight="bold")

    # D: the score if each photo's outline width matched its answer
    ax = fig.add_subplot(gs[1, 1])
    groups = (("Mean of 5 folds", np.mean(ours), np.mean(matched), PAPER_MEAN),
              ("Highest fold\n(paper's row is mean + std)", max(ours), max(matched), PAPER_HIGH))
    x = np.arange(len(groups))
    w = 0.26
    for k, (label, color) in enumerate((("Our model", OURS_C),
                                        ("Our model, each photo's outline\nwidth set to match its answer", "#9cc0f5"),
                                        ("Reference paper", PAPER_C))):
        vals = [g[k + 1] for g in groups]
        ax.bar(x + (k - 1) * w, vals, w, color=color, label=label)
        for xx, v in zip(x + (k - 1) * w, vals):
            ax.text(xx, v + 0.03, f"{v:.2f}", ha="center", fontsize=11)
    ax.set_xticks(x, [g[0] for g in groups], fontsize=12)
    ax.set_ylim(92, 94.3)
    ax.set_ylabel("Dice (%)")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title("D. If only the outline width matched each photo's answer,\nwe would pass the paper on both rows.",
                 loc="left", fontsize=13, weight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = ROOT / "reports" / "8-answer-key.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"answer key traced 1 px wider {100 * np.mean(wider):.2f}, narrower {100 * np.mean(narrower):.2f}; "
          f"ours {np.mean(ours):.2f} / {max(ours):.2f}; width matched {np.mean(matched):.2f} / {max(matched):.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
