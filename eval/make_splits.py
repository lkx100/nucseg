"""Build the frozen splits for spec 01 onward. Run once; the output is committed and then read-only.

    uv run python eval/make_splits.py data/raw/stage1_train

Steps:
1. Leak groups. Images that are exact duplicates, near-duplicates (for example two frames of the
   same recording), or that share textured 16x16 patches pixel for pixel (crops of the same field
   of view) are joined into one group. A group never spans two splits.
2. Strata: image type (grayscale or colour, dark or bright background) x size (small if the
   longest side is at most 360 px).
3. Test hold-out: one tenth of the groups, stratified. Then 5 CV folds on the rest. Inside each
   fold's training part, one tenth becomes the stop set used for early stopping.

Writes eval/splits.json and prints the make-up of every split.
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

SEED = 0
PATCH = 16          # window size for overlap detection
GRID = 32           # stride of the query windows taken from each image
MIN_STD = 10.0      # query windows flatter than this are skipped (background)
MIN_SHARED = 2      # distinct shared windows needed to link two images
NEAR_DUP = 0.9      # correlation of 32x32 thumbnails that marks a near-duplicate
OUT = Path(__file__).with_name("splits.json")


def image_type(rgb):
    gray = np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean() < 1
    bright = rgb.mean() > 100
    return ("bright" if bright else "dark") + "-" + ("gray" if gray else "colour")


def window_hashes(gray):
    """Hash of every PATCH x PATCH window, as a 2D polynomial hash (uint64, wrapping)."""
    x = gray.astype(np.uint64) + np.uint64(1)
    h, w = x.shape
    rows = np.zeros((h, w - PATCH + 1), np.uint64)
    for k in range(PATCH):
        rows = rows * np.uint64(1_000_003) + x[:, k:k + w - PATCH + 1]
    out = np.zeros((h - PATCH + 1, rows.shape[1]), np.uint64)
    for k in range(PATCH):
        out = out * np.uint64(998_244_353) + rows[k:k + h - PATCH + 1]
    return out


def leak_groups(grays, rgbs):
    n = len(grays)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    links = []
    by_hash = defaultdict(list)
    for i, rgb in enumerate(rgbs):
        by_hash[hashlib.md5(rgb.tobytes()).hexdigest()].append(i)
    for idx in by_hash.values():
        links += [(idx[0], j, "duplicate") for j in idx[1:]]

    thumbs = np.stack([np.asarray(Image.fromarray(g).resize((32, 32), Image.BILINEAR), float).ravel() for g in grays])
    thumbs = (thumbs - thumbs.mean(1, keepdims=True)) / (thumbs.std(1, keepdims=True) + 1e-6)
    corr = np.triu(thumbs @ thumbs.T / thumbs.shape[1], 1)
    links += [(i, j, "near-duplicate") for i, j in zip(*np.where(corr > NEAR_DUP))]

    # Query keys: textured windows on a coarse grid of each image.
    owners = defaultdict(set)
    for i, g in enumerate(grays):
        hashes = window_hashes(g)
        for y in range(0, hashes.shape[0], GRID):
            for x in range(0, hashes.shape[1], GRID):
                if g[y:y + PATCH, x:x + PATCH].std() > MIN_STD:
                    owners[int(hashes[y, x])].add(i)
    keys = np.array(sorted(owners), dtype=np.uint64)

    # Look for every query key anywhere inside every image.
    shared = Counter()
    for j, g in enumerate(grays):
        found = np.unique(window_hashes(g).ravel())
        for k in found[np.isin(found, keys, assume_unique=True)]:
            for i in owners[int(k)]:
                if i != j:
                    shared[(min(i, j), max(i, j))] += 1
    links += [(i, j, "overlap") for (i, j), c in shared.items() if c >= MIN_SHARED]

    for i, j, _ in links:
        parent[find(i)] = find(j)
    return [find(i) for i in range(n)], links


def split_once(idx, strata, groups, n_splits, seed):
    """Stratified group split of `idx`: returns (rest, held) with about 1/n_splits held out."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest, held = next(sgkf.split(idx, strata[idx], groups[idx]))
    return idx[rest], idx[held]


def describe(name, idx, strata, fg):
    counts = Counter(strata[idx])
    cells = "  ".join(f"{s}={counts.get(s, 0):3d}" for s in sorted(set(strata)))
    print(f"{name:10s} n={len(idx):3d}  fg={fg[idx].mean():.3f}  {cells}")


def main(root):
    dirs = sorted(p for p in Path(root).iterdir() if p.is_dir())
    ids, rgbs, grays, types, sizes, fg = [], [], [], [], [], []
    for d in dirs:
        im = Image.open(next((d / "images").glob("*.png"))).convert("RGB")
        rgb = np.asarray(im)
        mask = np.zeros(rgb.shape[:2], bool)
        for f in (d / "masks").glob("*.png"):
            mask |= np.asarray(Image.open(f)) > 0
        ids.append(d.name)
        rgbs.append(rgb)
        grays.append(np.asarray(im.convert("L")))
        types.append(image_type(rgb.astype(float)))
        sizes.append("small" if max(rgb.shape[:2]) <= 360 else "large")
        fg.append(mask.mean())
    ids, fg = np.array(ids), np.array(fg)
    strata = np.array([f"{t}/{s}" for t, s in zip(types, sizes)])

    groups, links = leak_groups(grays, rgbs)
    groups = np.array(groups)
    multi = [c for c in Counter(groups).values() if c > 1]
    print(f"images={len(ids)}  links: {Counter(kind for *_, kind in links)}")
    print(f"groups with >1 image: {len(multi)}  images in them: {sum(multi)}  largest: {max(multi, default=0)}")

    everything = np.arange(len(ids))
    pool, test = split_once(everything, strata, groups, 10, SEED)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds, stops = {}, {}
    for k, (tr, va) in enumerate(sgkf.split(pool, strata[pool], groups[pool])):
        _, stop = split_once(pool[tr], strata, groups, 10, SEED)
        folds[k], stops[k] = pool[va], stop

    print()
    describe("all", everything, strata, fg)
    describe("test", test, strata, fg)
    for k in folds:
        describe(f"fold{k} val", folds[k], strata, fg)
        describe(f"fold{k} stop", stops[k], strata, fg)

    for k in folds:  # sanity: no image or group in two places
        train = np.setdiff1d(pool, np.union1d(folds[k], stops[k]))
        parts = [set(groups[p]) for p in (test, folds[k], stops[k], train)]
        assert all(not (a & b) for n, a in enumerate(parts) for b in parts[n + 1:]), f"group leak in fold {k}"

    OUT.write_text(json.dumps({
        "source": "kaggle sindhu9642/nuclei-seg-corrected, stage1_train",
        "seed": SEED,
        "sklearn": sklearn.__version__,
        "test": sorted(ids[test].tolist()),
        "folds": {str(k): sorted(ids[v].tolist()) for k, v in folds.items()},
        "stop": {str(k): sorted(ids[v].tolist()) for k, v in stops.items()},
        "leak_groups": sorted(sorted(ids[groups == g].tolist()) for g in set(groups) if (groups == g).sum() > 1),
        "image_type": dict(zip(ids.tolist(), strata.tolist())),
    }, indent=1) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/stage1_train")
