"""Splits each photo's wrong pixels into missed nuclei, invented blobs and outline rings, for a run with saved masks.

    uv run python reports/photo_errors.py runs/<run_id>

Reads masks/<id>.png and per_image.json from the run and the hand-drawn nuclei from the local dataset. Writes
photo_errors.json into the run folder, prints the fold and type tables, and draws reports/6-lowest-photos-*.png.
A nucleus is missed when less than half of it is marked. A blob is invented when less than half of it lies on a nucleus.
`side` is (too much - too little) / wrong pixels: +1 when the model only marks too much, -1 when only too little.
`dice_best_size` is the score after growing or shrinking the whole mask by the best of -2 to 2 pixels for that photo.
It uses the answer to pick the size, so it is a ceiling for "right size per photo", not a method.
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
from nucseg.data import find_data_root  # noqa: E402
from eval.metrics import image_scores  # noqa: E402

TARGET, TARGET_MEAN = 0.9368, 0.9281
TYPES = {"dark-gray/small": "Fluorescence, small", "dark-gray/large": "Fluorescence, large",
         "bright-colour/small": "H&E colour", "bright-gray/large": "Brightfield, gray"}


def analyse(pred, nuclei):
    """Scores of the mask as it is, with missed and invented nuclei repaired, and with only the rings repaired."""
    gt = np.any(nuclei, axis=0)
    missed = [n for n in nuclei if (n & pred).sum() < 0.5 * n.sum()]
    missed_px = np.any(missed, axis=0) if missed else np.zeros_like(gt)
    lab, n = ndimage.label(pred)
    invented_px = np.zeros_like(gt)
    n_invented = 0
    if n:
        size = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1))
        on_gt = ndimage.sum_labels(gt, lab, range(1, n + 1))
        bad = np.flatnonzero(on_gt < 0.5 * size) + 1
        invented_px, n_invented = np.isin(lab, bad), len(bad)
    err = pred != gt
    cross = ndimage.generate_binary_structure(2, 1)
    sized = [image_scores(ndimage.binary_erosion(pred, cross, iterations=-s) if s < 0 else
                          ndimage.binary_dilation(pred, cross, iterations=s) if s else pred, gt)["dice"]
             for s in (-2, -1, 0, 1, 2)]
    objects = (pred | missed_px) & ~(invented_px & ~gt)   # missed nuclei filled in, invented blobs removed
    rings = np.where(missed_px | invented_px, pred, gt)   # every found nucleus drawn exactly, the rest as predicted
    return {"dice": image_scores(pred, gt)["dice"], "dice_objects_fixed": image_scores(objects, gt)["dice"],
            "dice_rings_fixed": image_scores(rings, gt)["dice"], "nuclei": len(nuclei), "missed": len(missed),
            "invented": n_invented, "err": int(err.sum()), "err_missed": int((err & missed_px & ~pred).sum()),
            "err_invented": int((err & invented_px).sum()),
            "side": float(((pred & ~gt).sum() - (~pred & gt).sum()) / max(1, err.sum())),
            "best_size": int(np.argmax(sized)) - 2, "dice_best_size": max(sized)}


def load(data_root, i):
    d = Path(data_root) / i
    img = np.asarray(Image.open(next((d / "images").glob("*.png"))).convert("RGB"))
    return img, np.stack([np.asarray(Image.open(f)) > 0 for f in sorted((d / "masks").glob("*.png"))])


def sheet(rows, run, data_root, title, path, cols=4):
    """Per photo: the photo (brightened if dark), the hand-drawn answer, and the model's answer with its errors."""
    n = len(rows)
    fig, axes = plt.subplots(-(-n // cols), cols * 3, figsize=(cols * 6.3, -(-n // cols) * 2.35))
    for ax in axes.ravel():
        ax.axis("off")
    for k, r in enumerate(rows):
        img, nuclei = load(data_root, r["id"])
        gt = np.any(nuclei, axis=0)
        pred = np.asarray(Image.open(run / "masks" / f"{r['id']}.png")) > 0
        show = img.astype(float)
        if show.mean() < 60:
            show = show / max(1.0, np.percentile(show, 99.8)) * 255
        over = np.zeros((*gt.shape, 3))
        over[pred & gt] = 1
        over[pred & ~gt] = (0.85, 0.2, 0.15)
        over[~pred & gt] = (0.2, 0.45, 0.95)
        a = axes.ravel()[k * 3:k * 3 + 3]
        a[0].imshow(np.clip(show, 0, 255).astype(np.uint8))
        a[1].imshow(gt, cmap="gray")
        a[2].imshow(over)
        a[0].set_title(f"{r['id'][:8]}  fold {r['fold']}\n{TYPES[r['type']]}", fontsize=7, loc="left")
        a[1].set_title(f"hand-drawn, {r['nuclei']} nuclei", fontsize=7, loc="left")
        a[2].set_title(f"model {r['dice']:.3f}   missed {r['missed']}, invented {r['invented']}", fontsize=7, loc="left")
    fig.suptitle(f"{title}\nright panel: white = correct, red = marked but not a nucleus, blue = nucleus not marked",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def table(rows, key, names):
    print(f"{key:<22}{'n':>5}{'Dice':>8}{'median':>8}{'<0.85':>7}{'objects fixed':>15}{'rings fixed':>13}"
          f"{'missed':>8}{'invented':>10}{'nuclei':>8}")
    for g in names:
        v = [r for r in rows if r[key] == g]
        d = np.array([r["dice"] for r in v])
        print(f"{str(names[g]):<22}{len(v):>5}{100 * d.mean():>8.2f}{100 * np.median(d):>8.2f}{(d < 0.85).sum():>7}"
              f"{100 * np.mean([r['dice_objects_fixed'] for r in v]):>15.2f}"
              f"{100 * np.mean([r['dice_rings_fixed'] for r in v]):>13.2f}"
              f"{sum(r['missed'] for r in v):>8}{sum(r['invented'] for r in v):>10}{sum(r['nuclei'] for r in v):>8}")


def main(run):
    run = Path(run)
    data_root = find_data_root(None)
    rows = []
    for p in json.loads((run / "per_image.json").read_text()):
        pred = np.asarray(Image.open(run / "masks" / f"{p['id']}.png")) > 0
        a = analyse(pred, load(data_root, p["id"])[1])
        assert abs(a["dice"] - p["dice_tta"]) < 1e-9, f"saved mask of {p['id']} does not give the reported score"
        rows.append({"id": p["id"], "fold": p["fold"], "type": p["type"], **a})
    (run / "photo_errors.json").write_text(json.dumps(rows) + "\n")

    table(rows, "fold", {k: f"fold {k}" for k in range(5)})
    print()
    table(rows, "type", TYPES)
    err = sum(r["err"] for r in rows)
    print(f"\nwrong pixels: missed nuclei {sum(r['err_missed'] for r in rows) / err:.1%}, invented blobs "
          f"{sum(r['err_invented'] for r in rows) / err:.1%}, rings {1 - sum(r['err_missed'] + r['err_invented'] for r in rows) / err:.1%}")
    low, side = np.array([r["dice"] < 0.85 for r in rows]), np.array([r["side"] for r in rows])
    for name, m in (("all photos", ~low | low), ("photos below 0.85", low)):
        print(f"{name}: {m.sum()}, of which the model draws too big in {(side[m] > 0.5).sum()} and too small in "
              f"{(side[m] < -0.5).sum()} (three quarters or more of the wrong pixels on one side)")
    for name, key in (("objects fixed", "dice_objects_fixed"), ("rings fixed", "dice_rings_fixed"),
                      ("best size", "dice_best_size"), ("as it is", "dice")):
        f = [np.mean([r[key] for r in rows if r["fold"] == k]) for k in range(5)]
        print(f"{name:<14} mean {100 * np.mean(f):.2f} (target {100 * TARGET_MEAN:.2f})  highest {100 * max(f):.2f} "
              f"(target {100 * TARGET:.2f})")

    best = max(range(5), key=lambda k: np.mean([r["dice"] for r in rows if r["fold"] == k]))
    low = sorted(rows, key=lambda r: r["dice"])
    sheet([r for r in low if r["fold"] == best][:16], run, data_root,
          f"The 16 lowest photos of the best fold (fold {best})", ROOT / "reports" / "6-lowest-photos-best-fold.png")
    sheet(low[:24], run, data_root, "The 24 lowest photos of all five folds", ROOT / "reports" / "6-lowest-photos-all.png")


if __name__ == "__main__":
    main(sys.argv[1])
