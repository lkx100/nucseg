"""How many hand-drawn answers (GT masks) look wrong, and of what kind. No GPU, no training.

    uv run python reports/label_check.py runs/<run_id with saved masks>

Three groups of checks:
1. Mask files alone, all 670 photos: empty files, duplicate nuclei, nuclei drawn on top of each other, nuclei in
   several pieces, nuclei with holes, tiny nuclei, and identical photos whose answers differ.
2. The photo alone, fluorescence only (dark background, bright nuclei; 546 of 670 photos): bright blobs that no answer
   covers (a nucleus nobody drew?) and drawn nuclei that are no brighter than the background (drawn on nothing?).
3. The photo and the saved model masks (602 CV photos): blobs the model finds that the answer lacks, nuclei in the
   answer the model does not find, and how many pixels wider or narrower each photo's answer is drawn.
4. Two patterns found by looking at the sheets: answer nuclei that are an exact 5x5 disk (21 px, a stamp, often on
   empty background), and model blobs the answer lacks that touch the photo border (nuclei cut by the edge, skipped).
   A what-if fold table rescores the model against answers with those fixed. It reads the model, so it is a guide
   to how much the fixes are worth, not a score.

Writes runs/<run_id>/label_check.json and reports/9-label-check-<kind>.png (six sheets), one sheet per kind of suspect, drawn
from a seeded random sample so a person can look and count how many suspects are real errors.
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nucseg.data import find_data_root, load_splits  # noqa: E402
from eval.metrics import image_scores  # noqa: E402

TINY = 10            # a nucleus under this many pixels is "tiny"
BLOB_MIN = 20        # bright blobs or model blobs smaller than this are ignored
BRIGHT, DIM = 0.5, 0.25   # contrast, on a scale where 0 is the background and 1 the photo's typical nucleus
SAMPLE = 20          # suspects drawn per sheet
EIGHT = np.ones((3, 3), bool)
DISK = np.array([[0, 1, 1, 1, 0]] + [[1] * 5] * 3 + [[0, 1, 1, 1, 0]], bool)


def load_photo(d):
    img = np.asarray(Image.open(next((d / "images").glob("*.png"))).convert("RGB"))
    masks = [np.asarray(Image.open(f)) > 0 for f in sorted((d / "masks").glob("*.png"))]
    return img, masks


def contrast_scale(gray, gt, nuclei):
    """Background level and the median nucleus level of the photo, for a 0-to-1 contrast scale."""
    bg = float(np.median(gray[~ndimage.binary_dilation(gt, iterations=3)])) if (~gt).any() else 0.0
    nuc = float(np.median([gray[n].mean() for n in nuclei if n.any()])) if nuclei else bg + 1
    return bg, max(nuc - bg, 1.0)


def is_disk(m):
    """True when the nucleus is exactly the 21-pixel 5x5 disk."""
    if m.sum() != 21:
        return False
    ys, xs = np.nonzero(m)
    y, x = ys.min(), xs.min()
    return np.ptp(ys) == 4 and np.ptp(xs) == 4 and bool((m[y:y + 5, x:x + 5] == DISK).all())


def mask_checks(masks):
    """Problems visible from the mask files alone."""
    out = Counter()
    sizes = [int(m.sum()) for m in masks]
    out["nuclei"] = len(masks)
    out["empty"] = sum(s == 0 for s in sizes)
    out["tiny"] = sum(0 < s < TINY for s in sizes)
    out["disk5"] = sum(is_disk(m) for m in masks)
    out["border_nuclei"] = sum(bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()) for m in masks)
    hashes = Counter(hashlib.md5(np.packbits(m).tobytes()).hexdigest() for m in masks if m.any())
    out["duplicate"] = sum(c - 1 for c in hashes.values())
    cover = np.sum(masks, axis=0) if masks else None
    out["overlap_px"] = int((cover > 1).sum()) if masks else 0
    out["overlapping_nuclei"] = sum(int((m & (cover > 1)).any()) for m in masks)
    for m in masks:
        if not m.any():
            continue
        lab, n = ndimage.label(m, structure=EIGHT)
        if n > 1:
            out["pieces"] += 1
            parts = np.sort(ndimage.sum_labels(m, lab, range(1, n + 1)))[::-1]
            out["pieces_stray_px"] += int(parts[1:].sum())
        if ndimage.binary_fill_holes(m).sum() > m.sum():
            out["holes"] += 1
    return out, sizes


def fluo_checks(gray, gt, nuclei):
    """Bright blobs nobody drew, and drawn nuclei no brighter than the background. Fluorescence photos only."""
    bg, scale = contrast_scale(gray, gt, nuclei)
    c = (gray - bg) / scale
    lab, n = ndimage.label(c > BRIGHT)
    unlabelled = []
    if n:
        size = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1))
        on_gt = ndimage.sum_labels(gt, lab, range(1, n + 1))
        for k in np.flatnonzero((size >= BLOB_MIN) & (on_gt < 0.1 * size)) + 1:
            ys, xs = np.nonzero(lab == k)
            unlabelled.append({"y": int(ys.mean()), "x": int(xs.mean()), "px": int(len(ys)),
                               "contrast": float(c[lab == k].mean())})
    dim = []
    for j, m in enumerate(nuclei):
        if m.sum() >= BLOB_MIN and c[m].mean() < DIM:
            ys, xs = np.nonzero(m)
            dim.append({"nucleus": j, "y": int(ys.mean()), "x": int(xs.mean()), "px": int(len(ys)),
                        "contrast": float(c[m].mean())})
    return unlabelled, dim, c


def model_checks(pred, gt, nuclei, c):
    """Model blobs the answer lacks and answer nuclei the model misses (same rule as photo_errors.py: under half)."""
    invented, missed = [], []
    border_px = np.zeros_like(gt)
    lab, n = ndimage.label(pred)
    if n:
        size = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1))
        on_gt = ndimage.sum_labels(gt, lab, range(1, n + 1))
        edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        for k in np.flatnonzero((size >= BLOB_MIN) & (on_gt < 0.5 * size)) + 1:
            ys, xs = np.nonzero(lab == k)
            invented.append({"y": int(ys.mean()), "x": int(xs.mean()), "px": int(len(ys)), "border": k in edge,
                             "contrast": None if c is None else float(c[lab == k].mean())})
            if k in edge:
                border_px |= lab == k
    for j, m in enumerate(nuclei):
        if m.sum() >= BLOB_MIN and (m & pred).sum() < 0.5 * m.sum():
            ys, xs = np.nonzero(m)
            missed.append({"nucleus": j, "y": int(ys.mean()), "x": int(xs.mean()), "px": int(len(ys)),
                           "contrast": None if c is None else float(c[m].mean())})
    return invented, missed, border_px


def sheet(cases, root, run, title, out, half=40):
    """Grid of crops around each suspect: photo, yellow = answer, blue dashes = model, red box = the suspect."""
    cols = 5
    rows = (len(cases) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for n, (ax, s) in enumerate(zip(axes.ravel(), cases)):
        d = Path(root) / s["id"]
        img, masks = load_photo(d)
        gt = np.any(masks, axis=0)
        y0, x0 = max(0, s["y"] - half), max(0, s["x"] - half)
        sl = np.s_[y0:y0 + 2 * half, x0:x0 + 2 * half]
        show = img.astype(float)
        show = np.clip(show / max(1.0, np.percentile(show, 99.8)) * 255, 0, 255).astype(np.uint8)
        ax.imshow(show[sl], interpolation="nearest")
        ax.contour(gt[sl], levels=[0.5], colors="#f2c200", linewidths=1.4)
        mp = Path(run) / "masks" / f"{s['id']}.png"
        if mp.exists():
            ax.contour((np.asarray(Image.open(mp)) > 0)[sl], levels=[0.5], colors="#00c8ff", linewidths=1.0,
                       linestyles="--")
        r = max(4, int(np.sqrt(s["px"] / np.pi)) + 3)
        ax.add_patch(plt.Rectangle((s["x"] - x0 - r, s["y"] - y0 - r), 2 * r, 2 * r, fill=False, ec="red", lw=1.2))
        ax.set_title(f"#{n + 1} {s['id'][:8]} {s['type'].split('/')[0]}\n{s['px']} px"
                     + ("" if s.get("contrast") is None else f", contrast {s['contrast']:.2f}"), fontsize=8, loc="left")
    fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main(run):
    run = Path(run)
    root = find_data_root(None)
    splits = load_splits()
    types = splits["image_type"]
    cv = {i: int(k) for k, ids in splits["folds"].items() for i in ids}
    ids = sorted(types)
    rng = np.random.default_rng(0)

    per_photo, suspects = [], defaultdict(list)
    whatif = defaultdict(lambda: defaultdict(list))
    sizes_all, photo_hash = [], defaultdict(list)
    for i in ids:
        img, masks = load_photo(root / i)
        t = types[i]
        gt = np.any(masks, axis=0)
        mc, sizes = mask_checks(masks)
        sizes_all += sizes
        photo_hash[hashlib.md5(img.tobytes()).hexdigest()].append(i)
        row = {"id": i, "type": t, "fold": cv.get(i, "test"), **mc}
        c = None
        if t.startswith("dark-gray"):
            gray = img.astype(float).mean(axis=2)
            unl, dim, c = fluo_checks(gray, gt, masks)
            row["bright_unlabelled"], row["dim_labelled"] = len(unl), len(dim)
            suspects["bright_unlabelled"] += [{"id": i, "type": t, **u} for u in unl]
            suspects["dim_labelled"] += [{"id": i, "type": t, **u} for u in dim]
        mp = run / "masks" / f"{i}.png"
        disks = [is_disk(m) for m in masks]
        suspects["disk5"] += [{"id": i, "type": t, "y": int(np.nonzero(m)[0].mean()),
                               "x": int(np.nonzero(m)[1].mean()), "px": 21,
                               "contrast": None if c is None else float(c[m].mean())}
                              for m, k in zip(masks, disks) if k]
        if mp.exists():
            pred = np.asarray(Image.open(mp)) > 0
            inv, mis, border_px = model_checks(pred, gt, masks, c)
            kept = [m for m, k in zip(masks, disks) if not k]
            no_disk = np.any(kept, axis=0) if kept else np.zeros_like(gt)
            for name, answer in (("as it is", gt), ("5x5 disks removed", no_disk),
                                 ("border blobs added", gt | border_px), ("both", no_disk | border_px)):
                whatif[name][row["fold"]].append(image_scores(pred, answer)["dice"])
            row["model_extra"], row["model_missed"] = len(inv), len(mis)
            suspects["model_extra"] += [{"id": i, "type": t, **u} for u in inv]
            suspects["model_missed"] += [{"id": i, "type": t, **u} for u in mis]
        per_photo.append(row)

    # identical photos whose answers differ
    gts = {}
    dup_photos = []
    for group in (g for g in photo_hash.values() if len(g) > 1):
        masks = [gts.setdefault(i, np.any(load_photo(root / i)[1], axis=0)) for i in group]
        agree = 2 * (masks[0] & masks[1]).sum() / max(1, masks[0].sum() + masks[1].sum())
        dup_photos.append({"ids": group, "answer_dice": float(agree)})

    width = {p["id"]: p["offset"] for p in json.loads((run / "width_check.json").read_text())["photos"]} \
        if (run / "width_check.json").exists() else {}
    for row in per_photo:
        row["width_offset_px"] = width.get(row["id"])

    # the same suspects, split by whether the photo itself agrees (fluorescence only)
    for s in suspects["model_extra"]:
        s["bright"] = s["contrast"] is not None and s["contrast"] >= BRIGHT
    for s in suspects["model_missed"]:
        s["dim"] = s["contrast"] is not None and s["contrast"] < DIM

    report = {"photos": per_photo, "identical_photos": dup_photos, "suspects": suspects}
    (run / "label_check.json").write_text(json.dumps(report, default=int))

    def tot(key, rows=per_photo):
        return sum(r.get(key, 0) or 0 for r in rows)

    def photos_with(key, rows=per_photo, at_least=1):
        return sum((r.get(key, 0) or 0) >= at_least for r in rows)

    fl = [r for r in per_photo if r["type"].startswith("dark-gray")]
    print(f"photos {len(per_photo)}, nuclei {tot('nuclei')}, fluorescence photos {len(fl)}")
    print("\n1. mask files alone (count / photos affected)")
    for k in ("empty", "duplicate", "overlapping_nuclei", "pieces", "holes", "tiny", "disk5"):
        print(f"  {k:20s} {tot(k):6d} / {photos_with(k)}")
    print(f"  overlap pixels {tot('overlap_px')}, stray pixels of split nuclei {tot('pieces_stray_px')}")
    print(f"  nucleus size: median {np.median(sizes_all):.0f} px, 1st pct {np.percentile(sizes_all, 1):.0f}")
    print(f"  identical photo pairs: {len(dup_photos)}; answers agree (Dice): "
          + ", ".join(f"{d['answer_dice']:.3f}" for d in dup_photos))
    print("\n2. fluorescence photo alone")
    for k in ("bright_unlabelled", "dim_labelled"):
        print(f"  {k:20s} {tot(k, fl):6d} / {photos_with(k, fl)}")
    print("\n3. with the model's masks (CV photos)")
    for k in ("model_extra", "model_missed"):
        cvrows = [r for r in per_photo if k in r]
        print(f"  {k:20s} {tot(k, cvrows):6d} / {photos_with(k, cvrows)} of {len(cvrows)}")
    ex = suspects["model_extra"]
    print(f"  model_extra on fluorescence that is bright like a nucleus: {sum(s['bright'] for s in ex)} of "
          f"{sum(s['contrast'] is not None for s in ex)}")
    mi = suspects["model_missed"]
    print(f"  model_missed on fluorescence that is dim like background: {sum(s['dim'] for s in mi)} of "
          f"{sum(s['contrast'] is not None for s in mi)}")
    off = np.array([v for v in width.values() if v is not None])
    if len(off):
        print(f"  outline width offset: |off| >= 1 px in {(abs(off) >= 1).sum()} of {len(off)} photos, "
              f">= 1.5 px in {(abs(off) >= 1.5).sum()}, >= 2 px in {(abs(off) >= 2).sum()}")
    dk = suspects["disk5"]
    dim_dk = [s for s in dk if s["contrast"] is not None and s["contrast"] < BRIGHT]
    print(f"\n4. 5x5 disks: {len(dk)} in {len({s['id'] for s in dk})} photos; on fluorescence "
          f"{sum(s['contrast'] is not None for s in dk)}, of which dim (contrast < {BRIGHT}) {len(dim_dk)} "
          f"in {len({s['id'] for s in dim_dk})} photos")
    gt_touch = sum(r.get("border_nuclei", 0) for r in per_photo)
    print(f"  model_extra touching the border: {sum(s['border'] for s in ex)} of {len(ex)}, in "
          f"{len({s['id'] for s in ex if s['border']})} photos; answer nuclei touching the border: "
          f"{gt_touch} of {tot('nuclei')}")
    print("  what-if fold table (TTA, per-image Dice, population std as in the paper):")
    for name, folds in whatif.items():
        f = np.array([100 * np.mean(folds[k]) for k in range(5)])
        print(f"    {name:20s} " + " ".join(f"{v:.2f}" for v in f)
              + f"  mean {f.mean():.2f} ± {f.std():.2f}  mean + std {f.mean() + f.std():.2f}  highest {f.max():.2f}")
    by_type = defaultdict(lambda: [0, 0, 0])
    for r in per_photo:
        if "model_extra" in r:
            b = by_type[r["type"]]
            b[0] += 1
            b[1] += r["model_extra"]
            b[2] += r["model_missed"]
    for t, (n, e, m) in sorted(by_type.items()):
        print(f"    {t:22s} photos {n:4d}  model extra {e:4d}  model missed {m:4d}")

    # sheets for looking: a seeded random sample of each kind
    kinds = {
        "bright-unlabelled": (suspects["bright_unlabelled"], "Bright blob no answer covers (fluorescence)"),
        "dim-labelled": (suspects["dim_labelled"], "Drawn nucleus no brighter than background (fluorescence)"),
        "model-extra": (ex, "Model finds a blob the answer lacks (all types)"),
        "model-missed": (mi, "Answer has a nucleus the model does not find (all types)"),
        "disks": (dk, "Answer nucleus that is an exact 5x5 disk (21 px)"),
        "border": ([s for s in ex if s["border"]], "Model blob the answer lacks, touching the photo border"),
    }
    picked = {}
    rng2 = np.random.default_rng(1)   # the two pattern sheets were drawn with their own seed when first looked at
    for name, (pool, title) in kinds.items():
        if not pool:
            continue
        idx = sorted((rng2 if name in ("disks", "border") else rng).choice(len(pool), size=min(SAMPLE, len(pool)), replace=False))
        cases = [pool[k] for k in idx]
        picked[name] = cases
        sheet(cases, root, run, f"{title}: random {len(cases)} of {len(pool)}. Yellow = answer, blue dashes = "
              "model, red box = suspect", ROOT / "reports" / f"9-label-check-{name}.png", half=24 if name == "disks" else 40)
    (run / "label_check_sample.json").write_text(json.dumps(picked, default=int))
    print("\nwrote", run / "label_check.json", "and reports/9-label-check-*.png")


if __name__ == "__main__":
    main(sys.argv[1])
