"""Figures for humans: progress, per-fold scores, data and splits, model architecture, cost and robustness.

    uv run python reports/make_figures.py

Reads results.tsv, runs/<run_id>/metrics.json, eval/splits.json and the local dataset. Writes PNGs to reports/.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nucseg.data import find_data_root, load_samples, load_splits  # noqa: E402

OUT = ROOT / "reports"
TARGET = 0.937
LABELS = {  # slug -> short name for plots
    "s01-base": "01 A\nbaseline",
    "s01-nsl": "01 B\nNSL adversarial",
    "s02-base": "02 A′\nbaseline rerun",
    "s02-hires": "02 C\nfull-res skip",
    "s02-scale2": "02 D\n2× input scale",
    "s03-hires-nsl1": "03 E\nC + NSL 1/255",
    "s04-hires-aug": "04 F\nC + strong aug",
    "s05-hires-scale2": "05 G\nC + 2× scale",
}
TYPES = {  # image_type key -> plain name
    "dark-gray/small": "Fluorescence, small",
    "dark-gray/large": "Fluorescence, large",
    "bright-colour/small": "H&E colour",
    "bright-gray/large": "Brightfield, gray",
}
GREEN, GRAY, GOLD, RED, BLUE = "#4c9a6a", "#a0a0a0", "#e0a526", "#c0392b", "#3a6ea5"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})


def load_runs():
    runs = []
    for row in csv.DictReader(open(ROOT / "results.tsv"), delimiter="\t"):
        m = json.loads((ROOT / "runs" / row["run_id"] / "metrics.json").read_text())
        slug = row["run_id"].split("-", 2)[2]
        runs.append(dict(row=row, m=m, slug=slug, label=LABELS.get(slug, slug)))
    return sorted(runs, key=lambda r: r["label"])


def progress(runs):
    best = max(runs, key=lambda r: r["m"]["cv"].get("tta", r["m"]["cv"])["dice"])
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(runs))
    for i, r in enumerate(runs):
        cv = r["m"]["cv"]
        colour = GOLD if r is best else (GREEN if r["row"]["status"] == "keep" else GRAY)
        ax.bar(i, cv["dice"], color=colour, width=0.6, yerr=cv["dice_std"], capsize=4, ecolor="#555")
        ax.text(i, cv["dice"] - 0.0015, f"{cv['dice']:.4f}", ha="center", va="top", color="white", weight="bold")
        if "tta" in cv:
            ax.plot(i, cv["tta"]["dice"], "D", color="black", ms=6, zorder=5)
            ax.text(i + 0.12, cv["tta"]["dice"], f"{cv['tta']['dice']:.4f}", va="bottom", fontsize=8)
    ax.axhline(TARGET, color=RED, ls="--", lw=2)
    ax.text(len(runs) - 0.5, TARGET + 0.0008, f"target {TARGET}", color=RED, ha="right", weight="bold")
    base = [r["m"]["cv"]["dice"] for r in runs if r["slug"] in ("s01-base", "s02-base")]
    if len(base) == 2:
        ax.axhspan(min(base), max(base), color=BLUE, alpha=0.12)
        ax.text(-0.45, max(base) + 0.0004, "run-to-run noise\n(same config, run twice)", color=BLUE, fontsize=8)
    gap = TARGET - best["m"]["cv"].get("tta", best["m"]["cv"])["dice"]
    ax.set_xticks(x, [r["label"] for r in runs])
    ax.set_ylim(0.895, 0.945)
    ax.set_ylabel("Dice (mean over images, then over 5 folds)")
    ax.set_title(f"Progress towards the 0.937 target: best so far is {gap * 100:.1f} points short", loc="left",
                 weight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GOLD, GREEN, GRAY)]
    handles.append(plt.Line2D([], [], marker="D", color="black", ls=""))
    ax.legend(handles, ["best so far", "kept", "discarded", "with test-time augmentation"], loc="upper left",
              frameon=False, fontsize=8, ncol=4, bbox_to_anchor=(0, 0.93))
    fig.text(0.01, 0.01, "Bars: CV Dice. Whiskers: spread across the 5 folds (std).", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "1-progress.png")


def folds(runs):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    shown = {"s02-base": GRAY, "s02-hires": GOLD, "s04-hires-aug": GREEN, "s05-hires-scale2": BLUE}
    for r in [r for r in runs if r["slug"] in shown]:
        c = shown[r["slug"]]
        f = r["m"]["folds"]
        ax.plot([d["fold"] for d in f], [d["dice"] for d in f], "o-", color=c, lw=2, label=r["label"].replace("\n", ": "))
        early = [d for d in f if d["epochs_run"] < 30]
        ax.plot([d["fold"] for d in early], [d["dice"] for d in early], "o", mfc="none", mec=RED, ms=13, mew=1.5)
    ax.plot([], [], "o", mfc="none", mec=RED, ms=10, label="early stopping ended this fold before epoch 30")
    ax.axhline(TARGET, color=RED, ls="--", lw=2)
    ax.text(4.2, TARGET + 0.0005, "target", color=RED, va="bottom", ha="right")
    ax.set_xticks(range(5), [f"fold {k}\n(~120 images)" for k in range(5)])
    ax.set_ylabel("Dice on the held-out fold")
    ax.set_title("Every fold: fold 3 is easiest, fold 4 hardest", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.set_ylim(0.90, 0.945)
    fig.tight_layout()
    fig.savefig(OUT / "2-folds.png")


def by_type(runs, splits):
    counts = {t: 0 for t in TYPES}
    for t in splits["image_type"].values():
        counts[t] += 1
    shown = [r for r in runs if r["slug"] in ("s02-hires", "s02-scale2", "s04-hires-aug", "s05-hires-scale2")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.2
    for j, (r, c) in enumerate(zip(shown, (GOLD, GRAY, GREEN, BLUE))):
        vals = [r["m"]["dice_by_type"][t] for t in TYPES]
        bars = ax.bar(np.arange(len(TYPES)) + (j - 1.5) * w, vals, w, color=c, label=r["label"].replace("\n", ": "))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + w / 2, v + 0.001, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    ax.axhline(TARGET, color=RED, ls="--", lw=2)
    ax.set_xticks(range(len(TYPES)), [f"{name}\n{counts[t]} images" for t, name in TYPES.items()])
    ax.set_ylim(0.87, 0.96)
    ax.set_ylabel("Dice")
    ax.set_title("Where the errors are: colour and brightfield images lag far behind", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "3-by-image-type.png")


def cost(runs):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in runs:
        m = r["m"]
        a1.scatter(m["minutes"], m["cv"]["dice"], s=m["peak_vram_gb"] * 25, color=GOLD if r["slug"] == "s02-hires" else BLUE,
                   alpha=0.8)
        a1.annotate(r["label"].replace("\n", ": "), (m["minutes"], m["cv"]["dice"]), textcoords="offset points",
                    xytext=(8, -3), fontsize=8)
    a1.set_xlabel("GPU minutes for all 5 folds (Kaggle T4)")
    a1.set_ylabel("CV Dice")
    a1.set_xlim(60, 400)
    a1.margins(y=0.15)
    a1.set_title("Cost vs score (dot size = GPU memory)", loc="left", weight="bold")
    x = np.arange(len(runs))
    clean = [r["m"]["cv"]["dice"] for r in runs]
    attacked = [r["m"]["cv"]["robust_dice"] for r in runs]
    for dx_, vals, c, lab in ((-0.2, clean, BLUE, "normal images"),
                              (0.2, attacked, RED, "after a tiny adversarial change (2/255)")):
        a2.bar(x + dx_, vals, 0.4, color=c, label=lab)
        for xi, v in zip(x, vals):
            a2.text(xi + dx_, v + 0.003, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    a2.set_xticks(x, [r["label"].split("\n")[0] for r in runs])
    a2.set_ylim(0.75, 0.97)
    a2.set_ylabel("Dice")
    a2.set_title("NSL (01 B, 03 E) bought robustness, not accuracy", loc="left", weight="bold")
    a2.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "5-cost-and-robustness.png")


def data_and_splits(splits):
    root = find_data_root()
    picks = {}
    for i, t in sorted(splits["image_type"].items()):
        picks.setdefault(t, i)
    samples = load_samples(root, [picks[t] for t in TYPES])
    fig = plt.figure(figsize=(11, 6.8))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9])
    for j, t in enumerate(TYPES):
        img, gt = samples[picks[t]]
        ax = fig.add_subplot(gs[0, j])
        dark = img.mean() < 60  # stretch dark fluorescence images so the nuclei are visible
        ax.imshow(np.clip(img / np.percentile(img, 99.5), 0, 1) if dark else img)
        ax.set_title(f"{TYPES[t]}\n{img.shape[1]}×{img.shape[0]} px" + (", brightened" if dark else ""), fontsize=9)
        ax.axis("off")
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(gt, cmap="gray")
        ax.set_title(f"answer mask: {gt.mean() * 100:.0f}% nucleus", fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    ax.set_xlim(0, 670)
    ax.set_ylim(0, 3)
    n_test = len(splits["test"])
    fold_sizes = [len(splits["folds"][str(k)]) for k in range(5)]
    ax.barh(2.2, n_test, 0.6, color=RED)
    ax.text(n_test / 2, 2.2, f"test\n{n_test}", ha="center", va="center", color="white", fontsize=8, weight="bold")
    ax.text(n_test / 2, 1.75, "locked away,\nnot used yet", ha="center", va="top", color=RED, fontsize=8)
    left = n_test
    for k, n in enumerate(fold_sizes):
        ax.barh(2.2, n, 0.6, left=left, color=plt.cm.Blues(0.35 + 0.12 * k), edgecolor="white")
        ax.text(left + n / 2, 2.2, f"fold {k}\n{n}", ha="center", va="center", fontsize=8)
        left += n
    ax.text(0, 2.75, "All 670 labelled images (no image appears in two places; near-duplicates are kept together)",
            fontsize=9, weight="bold")
    n_stop = len(splits["stop"]["0"])
    n_train = sum(fold_sizes) - fold_sizes[0] - n_stop
    segs = [(n_train, GREEN, f"train on it: {n_train}"), (n_stop, GOLD, f"stop set: {n_stop}"),
            (fold_sizes[0], BLUE, f"score it: {fold_sizes[0]}")]
    left = n_test
    for n, c, txt in segs:
        ax.barh(0.8, n, 0.6, left=left, color=c)
        ax.text(left + n / 2, 0.8, txt, ha="center", va="center", color="white", fontsize=8, weight="bold")
        left += n
    ax.text(n_test, 1.35, "One run of fold 0 (repeated for each of the 5 folds, so every image is scored once):",
            fontsize=9)
    ax.text(n_test, 0.1, "The stop set decides when to stop training; the scored fold is never seen until the end.",
            fontsize=8, color="#555")
    fig.suptitle("The data: 4 kinds of microscope image, and how they are split", weight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "0-data-and-splits.png")


def box(ax, x, y, w, h, text, colour, fs=8, tc="black"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=colour, ec="#333", lw=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, p, q, colour="#333", style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=colour, lw=1.4, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.6)
    ax.axis("off")
    enc, dec, new, io = "#cfe0f3", "#f6dcc2", "#f7e08a", "#e3e3e3"
    # Encoder (left column), from full resolution down
    ex, dx = 3.0, 9.0
    ys = [6.3, 5.0, 3.7, 2.4]
    enc_txt = ["Swin stage 1 · 2 blocks\n64×64 · 96 ch (1/4)", "Swin stage 2 · 2 blocks\n32×32 · 192 ch (1/8)",
               "Swin stage 3 · 6 blocks\n16×16 · 384 ch (1/16)", "Swin stage 4 · 2 blocks\n8×8 · 768 ch (1/32)"]
    box(ax, ex, 7.9, 2.6, 0.55, "Input crop  256×256 · RGB", io, 9)
    box(ax, ex, 7.15, 2.6, 0.45, "patch embedding: 4×4 pixels → 1 token", enc)
    for y, t in zip(ys, enc_txt):
        box(ax, ex, y, 2.6, 0.8, t, enc, 8.5)
    arrow(ax, (ex, 7.62), (ex, 7.38))
    arrow(ax, (ex, 6.92), (ex, 6.7))
    for a, b in zip(ys, ys[1:]):
        arrow(ax, (ex, a - 0.4), (ex, b + 0.4))
    # Decoder (right column), from coarse back to full resolution
    dys = [3.7, 5.0, 6.3]
    dec_txt = ["Decoder 1: upsample 2×, join,\n2× conv → 16×16 · 256 ch", "Decoder 2: upsample 2×, join,\n2× conv → 32×32 · 128 ch",
               "Decoder 3: upsample 2×, join,\n2× conv → 64×64 · 64 ch"]
    for y, t in zip(dys, dec_txt):
        box(ax, dx, y, 2.9, 0.8, t, dec, 8.5)
    arrow(ax, (ex + 1.3, 2.4), (dx - 1.45, 3.55), rad=0.15)
    for (y_enc, y_dec) in zip(ys[2::-1], dys):
        arrow(ax, (ex + 1.3, y_enc), (dx - 1.45, y_dec), colour="#777", ls="--")
    ax.text(6.0, 5.15, "skip connections\n(copy encoder features across)", ha="center", fontsize=8, color="#555")
    for a, b in zip(dys, dys[1:]):
        arrow(ax, (dx, a + 0.4), (dx, b - 0.4))
    # Final two upsampling stages and the new full-resolution stem
    box(ax, dx, 7.2, 2.9, 0.6, "Up 4: upsample 2×, join,\n2× conv → 128×128 · 32 ch", dec, 8.5)
    box(ax, dx, 7.95, 2.9, 0.6, "Up 5: upsample 2×, join,\n2× conv → 256×256 · 16 ch", dec, 8.5)
    arrow(ax, (dx, 6.7), (dx, 6.9))
    arrow(ax, (dx, 7.5), (dx, 7.65))
    box(ax, 11.2, 7.95, 1.3, 0.6, "1×1 conv\n→ nucleus\nprobability", io, 8)
    arrow(ax, (dx + 1.45, 7.95), (11.2 - 0.65, 7.95))
    box(ax, 6.0, 7.95, 2.3, 0.55, "Stem: 2× conv on raw pixels\n256×256 · 16 ch", new, 8)
    box(ax, 6.0, 7.2, 2.3, 0.55, "Stem: pool 2×, 2× conv\n128×128 · 32 ch", new, 8)
    arrow(ax, (ex + 1.3, 7.95), (6.0 - 1.15, 7.95))
    arrow(ax, (6.0, 7.67), (6.0, 7.48))
    arrow(ax, (6.0 + 1.15, 7.95), (dx - 1.45, 7.95), colour="#b8860b")
    arrow(ax, (6.0 + 1.15, 7.2), (dx - 1.45, 7.2), colour="#b8860b")
    # Legend and notes
    notes = [(enc, "Swin-T encoder (pretrained on ImageNet), 27.5 M parameters"),
             (dec, "U-Net decoder, trained from scratch, 4.1 M parameters"),
             (new, "Full-resolution stem, added in spec 02 arm C (+28 k parameters, +0.5 Dice)")]
    for k, (c, t) in enumerate(notes):
        box(ax, 0.55, 1.35 - k * 0.42, 0.35, 0.28, "", c)
        ax.text(0.85, 1.35 - k * 0.42, t, va="center", fontsize=9)
    ax.text(6.4, 1.35, "Inside a Swin block: split the token grid into 8×8 windows, let tokens in each window\n"
                       "attend to each other, then shift the windows by half so neighbouring windows mix.",
            fontsize=8.5, va="center")
    ax.text(6.4, 0.6, "At test time the model slides over the full image in 256×256 tiles (50% overlap),\n"
                      "averages the overlapping predictions, and marks a pixel as nucleus if probability > 0.5.",
            fontsize=8.5, va="center")
    ax.set_title("The current best model: Swin-T U-Net with a full-resolution skip (31.7 M parameters)", loc="left",
                 weight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "4-architecture.png")


if __name__ == "__main__":
    runs = load_runs()
    splits = load_splits()
    data_and_splits(splits)
    progress(runs)
    folds(runs)
    by_type(runs, splits)
    architecture()
    cost(runs)
    print("\n".join(sorted(str(p.relative_to(ROOT)) for p in OUT.glob("*.png"))))
