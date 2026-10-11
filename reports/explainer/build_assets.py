"""Exports the images and numbers the explainer page shows. Reads runs/, results.tsv, eval/splits.json and data/.
Run: uv run python reports/explainer/build_assets.py"""
import csv, json, re
from pathlib import Path

import numpy as np
from PIL import Image

from nucseg.data import find_data_root, load_samples

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "assets"
H4, CSTAR = "20261003-512e574-s07-ens4", "20261002-fa7e63a-s06-hires-min30"
TYPES = ["dark-gray/small", "dark-gray/large", "bright-colour/small", "bright-gray/large"]
TYPE_NAMES = ["Fluorescence, small nuclei", "Fluorescence, large image", "H&E stained tissue", "Brightfield, gray"]
IDEAS = {  # slug -> (label, plain idea, verdict)
    "s01-base": ("01 A", "The plain Swin-T U-Net", "start"),
    "s01-nsl": ("01 B", "Adversarial training (NSL)", "no gain"),
    "s02-base": ("02 A′", "The baseline again, nothing changed", "luck check"),
    "s02-hires": ("02 C", "The full-size branch", "win"),
    "s02-scale2": ("02 D", "Photos enlarged 2×", "no win"),
    "s03-hires-nsl1": ("03 E", "C + gentler adversarial training", "no gain"),
    "s04-hires-aug": ("04 F", "C + altered copies of the photos", "no win"),
    "s05-hires-scale2": ("05 G", "C + photos enlarged 2×", "no gain"),
    "s06-hires-min30": ("06 C*", "C + at least 30 epochs of training", "new baseline"),
    "s06-hires-aug-min30": ("06 F*", "F + at least 30 epochs of training", "no gain"),
    "s07-ens2": ("07 H2", "2 finished models averaged", "no win"),
    "s07-ens4": ("07 H4", "4 finished models averaged", "best so far"),
    "s08-hires-edge": ("08 I", "C* + edge mistakes count 5 times", "no win"),
    "s09-plain-cv": ("09 J", "C* again, with the wrong answer key dropped", "new baseline"),
}


def save(arr, name, size=None):
    im = Image.fromarray(arr)
    if size:
        im = im.resize(size, Image.LANCZOS)
    (OUT / name).parent.mkdir(parents=True, exist_ok=True)
    im.save(OUT / name, optimize=True)


def mask_img(m):
    return (m.astype(np.uint8) * 255)


def centre_crop(a, n=256):
    y, x = (a.shape[0] - n) // 2, (a.shape[1] - n) // 2
    return a[y:y + n, x:x + n]


splits = json.loads((ROOT / "eval/splits.json").read_text())
per = json.loads((ROOT / "runs" / H4 / "per_image.json").read_text())
S = load_samples(find_data_root(), [p["id"] for p in per])
by_id = {p["id"]: p for p in per}

# 1. One typical image per image type (the median scorer of that type), centre crop 256
type_examples = []
for k, t in enumerate(TYPES):
    cand = sorted((p for p in per if p["type"] == t), key=lambda p: p["dice_tta"])
    p = cand[len(cand) // 2]
    img, m = S[p["id"]]
    save(centre_crop(img), f"types/{k}-img.png")
    save(mask_img(centre_crop(m)), f"types/{k}-mask.png")
    type_examples.append(dict(name=TYPE_NAMES[k], img=f"assets/types/{k}-img.png", mask=f"assets/types/{k}-mask.png",
                              n_cv=len(cand), n_all=sum(v == t for v in splits["image_type"].values()),
                              mean=round(float(np.mean([c["dice_tta"] for c in cand])), 4)))

# 2. Hero image for the model walk-through: a clear 256x256 fluorescence image, brightened for display
cand = [p for p in per if p["type"] == TYPES[0] and S[p["id"]][0].shape[:2] == (256, 256) and p["dice_tta"] > 0.94
        and 0.10 < S[p["id"]][1].mean() < 0.20]
hero = max(cand, key=lambda p: S[p["id"]][0].std())
img, m = S[hero["id"]]
g = img.astype(np.float32)
g = np.clip(g / max(np.percentile(g, 99.8), 1) * 255, 0, 255).astype(np.uint8)
save(g, "hero-img.png")
save(mask_img(m), "hero-mask.png")

# 3. The three 260x347 images: the wrongly labelled one and its two look-alikes
family = []
for p in per:
    img, m = S[p["id"]]
    if img.shape[:2] != (260, 347):
        continue
    key = p["id"][:8]
    save(img, f"family/{key}-img.png")
    save(mask_img(m), f"family/{key}-mask.png")
    family.append(dict(id=key, fold=p["fold"], dice=round(p["dice_tta"], 3), mask_px=int(m.sum()), px=int(m.size),
                       img=f"assets/family/{key}-img.png", mask=f"assets/family/{key}-mask.png",
                       stop_folds=[int(k) for k, v in splits["stop"].items() if p["id"] in v]))
family.sort(key=lambda f: f["dice"])

# 4. Real predictions: the 16 panels of C*'s fold-0 prediction grid (image, expert mask, model mask, errors)
grid = np.asarray(Image.open(ROOT / "runs" / CSTAR / "plots/fold0_preds.png").convert("RGB"))
cell = grid.shape[0] // 4
for r in range(4):
    for c in range(4):
        a = grid[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
        if c == 0:  # the photo has no white, so it gives the panel box; the other columns share its rows and size
            ink = (a < 250).any(2)
            rows = np.where(ink.mean(1) > 0.8)[0]
            cols = np.where(ink[rows].mean(0) > 0.8)[0]
            w = cols[-1] - cols[0] + 1
        dark = np.where((a[rows] < 128).all(2).any(0))[0] if c in (1, 2) else cols
        x0 = dark[0] if c in (1, 2) else np.where((a[rows] < 250).any(2).mean(0) > 0.8)[0][0]
        save(a[rows[0]:rows[-1] + 1, x0:x0 + w], f"pred/r{r}c{c}.png")
pred_rows = [dict(type="H&E stained tissue", dice=0.807), dict(type="Brightfield, gray", dice=0.930),
             dict(type="Fluorescence, large image", dice=0.968), dict(type="Fluorescence, small nuclei", dice=0.961)]

# 5. Thumbnails of the 24 lowest-scoring images
tail = sorted(per, key=lambda p: p["dice_tta"])
for i, p in enumerate(tail[:24]):
    img, m = S[p["id"]]
    n = min(img.shape[:2])
    save(centre_crop(img, n), f"tail/{i:02d}-img.png", (128, 128))
    save(mask_img(centre_crop(m, n)), f"tail/{i:02d}-mask.png", (128, 128))

# 6. Numbers
runs = []
for row in csv.DictReader(open(ROOT / "results.tsv"), delimiter="\t"):
    m = json.loads((ROOT / "runs" / row["run_id"] / "metrics.json").read_text())
    slug = row["run_id"].split("-", 2)[2]
    if slug not in IDEAS:  # scoring-only reruns of a model that is already listed
        continue
    label, idea, verdict = IDEAS[slug]
    cv = m["cv"]
    runs.append(dict(slug=slug, label=label, idea=idea, verdict=verdict, plainCv=slug == "s09-plain-cv",
                     dice=round(cv["dice"], 4),
                     tta=round(cv["tta"]["dice"], 4) if "tta" in cv else None,
                     folds=[round(f["dice"], 4) for f in m["folds"]], minutes=round(m["minutes"]),
                     attacked=round(cv["robust_dice"], 3) if "robust_dice" in cv else None))
runs.sort(key=lambda r: r["label"])

scores = np.array([p["dice_tta"] for p in tail])
fam_ids = {f["id"] for f in family}
images = [dict(s=round(p["dice_tta"], 4), t=TYPES.index(p["type"]), f=p["fold"],
               fam=p["id"][:8] if p["id"][:8] in fam_ids else None) for p in tail]
low = scores < 0.85
in_fam = np.array([i["fam"] is not None for i in images])
stats = dict(n=len(scores), mean=round(float(scores.mean()), 4), median=round(float(np.median(scores)), 4), target=0.937,
             n_low=int(low.sum()), low_cut=0.85, mean_without_low=round(float(scores[~low].mean()), 4),
             points_lost_by_low=round(float((0.937 - scores[low]).sum() / len(scores) * 100), 2),
             mean_without_family=round(float(scores[~in_fam].mean()), 4),
             low_by_type=[int((low & np.array([i["t"] == k for i in images])).sum()) for k in range(4)])

log = (ROOT / "runs" / CSTAR / "train.log").read_text()
stop_curves = [[] for _ in range(5)]
for f, d in re.findall(r"fold (\d) epoch \d+/\d+ loss [\d.]+ stop_dice ([\d.]+)", log):
    stop_curves[int(f)].append(float(d))

data = dict(
    typeNames=TYPE_NAMES, typeExamples=type_examples, hero=dict(img="assets/hero-img.png", mask="assets/hero-mask.png"),
    family=family, predRows=pred_rows, runs=runs, images=images, stats=stats, stopCurves=stop_curves,
    tail=[dict(img=f"assets/tail/{i:02d}-img.png", mask=f"assets/tail/{i:02d}-mask.png", s=round(p["dice_tta"], 3),
               t=TYPES.index(p["type"]), fam=p["id"][:8] if p["id"][:8] in fam_ids else None) for i, p in enumerate(tail[:24])],
    splits=dict(total=len(splits["image_type"]), test=len(splits["test"]), cv=len(per),
                folds=[sum(p["fold"] == k for p in per) for k in range(5)], stop=49, train=433),
    noise=dict(a=0.9178, b=0.9143),
)
(OUT / "data.js").write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
print(json.dumps({k: v for k, v in data.items() if k not in ("images", "stopCurves", "tail", "runs")}, indent=1))
print("runs", [(r["label"], r["dice"], r["tta"]) for r in runs])
print("stop curve lengths", [len(c) for c in stop_curves], "fold1 e20-26", stop_curves[1][19:26])
