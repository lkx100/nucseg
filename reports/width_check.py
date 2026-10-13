"""Can the width of a photo's hand-drawn outlines be told from the photo? Follow-up to spec 10, no GPU.

    uv run python reports/width_check.py runs/<run_id with masks and photo_errors.json>

For every photo it measures, without the answer: brightness, blur, the size and number of the model's nuclei, and how
the brightness falls off across the model's outline. With the answer: `offset`, how many pixels wider (+) or narrower
(-) the model draws than the hand-drawn answer, and `level`, how far down the brightness slope the hand-drawn outline
sits (1 = at the nucleus's own brightness, 0 = at the background's).
Then, fold by fold, a predictor trained on the other four folds guesses each photo's offset, and the mask is grown
or shrunk by the rounded guess. That never reads the photo's own answer, so its Dice is a real score.
Writes width_check.json into the run folder and draws reports/7-outline-width.png and reports/7-look-alike-pairs.png.
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
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nucseg.data import find_data_root, load_samples  # noqa: E402
from eval.metrics import image_scores  # noqa: E402

TYPES = ["dark-gray/small", "dark-gray/large", "bright-colour/small", "bright-gray/large"]
CROSS = ndimage.generate_binary_structure(2, 1)
TARGET, TARGET_MEAN = 0.9368, 0.9281


def resize(mask, s):
    return mask if s == 0 else (ndimage.binary_dilation(mask, CROSS, iterations=s) if s > 0
                                else ndimage.binary_erosion(mask, CROSS, iterations=-s))


def slope(g, mask):
    """Brightness on rings 3 px inside to 3 px outside the mask's outline, scaled so inside = 1 and background = 0."""
    dist = ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)  # < 0 inside
    inside, outside = g[dist < -3], g[(dist > 5) & (dist < 12)]
    if inside.size < 5 or outside.size < 5 or abs(inside.mean() - outside.mean()) < 1e-3:
        inside, outside = g[mask], g[dist > 3]
    if inside.size == 0 or outside.size == 0 or abs(inside.mean() - outside.mean()) < 1e-3:
        return [0.0] * 6
    lo, hi = outside.mean(), inside.mean()
    rings = [g[(dist >= a - 0.5) & (dist < a + 0.5)] for a in (-3, -2, -1, 1, 2, 3)]  # -1 and 1 touch the outline
    return [float((r.mean() - lo) / (hi - lo)) if r.size else 0.0 for r in rings]


def features(img, pred):
    g = img.mean(axis=2) / 255
    lab, n = ndimage.label(pred)
    area = pred.sum() / max(1, n)
    lap = ndimage.laplace(ndimage.gaussian_filter(g, 1))
    s = slope(g, pred)
    return {"h": img.shape[0], "w": img.shape[1], "mean": g.mean(), "std": g.std(), "p1": np.percentile(g, 1),
            "p50": np.percentile(g, 50), "p99": np.percentile(g, 99), "blur": lap.std() / (g.std() + 1e-6),
            "n": n, "log_area": np.log1p(area), "cover": pred.mean(), "colour": float(np.ptp(img.astype(float), axis=2).mean()),
            **{f"slope{k}": v for k, v in enumerate(s)}, "edge_drop": s[2] - s[3]}


def main(run):
    run = Path(run)
    rows = json.loads((run / "photo_errors.json").read_text())
    S = load_samples(find_data_root(None), [r["id"] for r in rows])
    out, P = [], {}
    for r in rows:
        img, gt = S[r["id"]]
        pred = np.asarray(Image.open(run / "masks" / f"{r['id']}.png")) > 0
        P[r["id"]] = pred
        edge = pred & ~ndimage.binary_erosion(pred, CROSS)
        g = img.mean(axis=2) / 255
        out.append({**r, "feat": features(img, pred), "offset": float((pred.sum() - gt.sum()) / max(1, edge.sum())),
                    "level": float(np.mean(slope(g, gt)[2:4])), "level_model": float(np.mean(slope(g, pred)[2:4]))})
    names = list(out[0]["feat"])
    X = np.array([[o["feat"][n] for n in names] + [o["type"] == t for t in TYPES] for o in out], float)
    y = np.clip(np.array([o["offset"] for o in out]), -3, 3)
    fold = np.array([o["fold"] for o in out])
    typ = np.array([o["type"] for o in out])
    dice = np.array([o["dice"] for o in out])

    models = {"straight line": lambda: make_pipeline(StandardScaler(), RidgeCV(np.logspace(-2, 3, 12))),
              "20 most similar photos": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(20, weights="distance")),
              "boosted trees": lambda: GradientBoostingRegressor(random_state=0)}
    report = {}
    for name, make in models.items():
        guess = np.zeros(len(y))
        for k in range(5):
            guess[fold == k] = make().fit(X[fold != k], y[fold != k]).predict(X[fold == k])
        for o, gu in zip(out, guess):
            o.setdefault("guess", {})[name] = float(gu)
        shift = np.clip(-np.round(guess), -2, 2).astype(int)
        new = np.array([image_scores(resize(P[o["id"]], s), S[o["id"]][1])["dice"] for o, s in zip(out, shift)])
        f_new, f_old = [new[fold == k].mean() for k in range(5)], [dice[fold == k].mean() for k in range(5)]
        r2 = lambda m: 1 - ((y[m] - guess[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()  # noqa: E731
        report[name] = {"r2": r2(fold >= 0), "r2_small_fluo": r2(typ == TYPES[0]), "changed": int((shift != 0).sum()),
                        "better": int((new > dice + 1e-9).sum()), "worse": int((new < dice - 1e-9).sum()),
                        "folds": f_new, "mean": float(np.mean(f_new)), "highest": float(max(f_new))}
        print(f"{name:<24} explains {100 * report[name]['r2']:5.1f}% of the offset ({100 * report[name]['r2_small_fluo']:.1f}% in small "
              f"fluorescence); resized {report[name]['changed']} photos, {report[name]['better']} better, {report[name]['worse']} worse")
        print(f"{'':<24} folds {' '.join(f'{100 * v:.2f}' for v in f_new)}  mean {100 * np.mean(f_new):.2f} "
              f"(was {100 * np.mean(f_old):.2f}, target {100 * TARGET_MEAN:.2f})  highest {100 * max(f_new):.2f} "
              f"(was {100 * max(f_old):.2f}, target {100 * TARGET:.2f})")

    sf = typ == TYPES[0]
    lev, levm = np.array([o["level"] for o in out]), np.array([o["level_model"] for o in out])
    print(f"\nsmall fluorescence, where the outline sits on the brightness slope: hand-drawn {lev[sf].mean():.2f} ± "
          f"{lev[sf].std():.2f}, model {levm[sf].mean():.2f} ± {levm[sf].std():.2f}")
    print(f"offset in pixels, all photos: mean {y.mean():+.2f}, std {y.std():.2f}; small fluorescence std {y[sf].std():.2f}")
    print("strongest single clues (correlation with the offset, small fluorescence):")
    cors = sorted(((abs(np.corrcoef(X[sf, j], y[sf])[0, 1]), n, np.corrcoef(X[sf, j], y[sf])[0, 1])
                   for j, n in enumerate(names) if X[sf, j].std() > 0), reverse=True)[:5]
    print("  " + ", ".join(f"{n} {c:+.2f}" for _, n, c in cors))

    # Look-alike pairs: nearest photos in feature space (small fluorescence, different folds not required) with opposite offsets
    Z = (X[:, :len(names)] - X[sf][:, :len(names)].mean(0)) / (X[sf][:, :len(names)].std(0) + 1e-9)
    idx = np.flatnonzero(sf)
    d = np.linalg.norm(Z[idx][:, None] - Z[idx][None], axis=2)
    near = np.sort(d + np.eye(len(idx)) * 1e9, axis=1)[:, 4]  # distance to the 5th most similar photo
    gap = y[idx][:, None] - y[idx][None]
    pairs = sorted(((gap[a, b], idx[a], idx[b]) for a in range(len(idx)) for b in range(len(idx))
                    if d[a, b] <= min(near[a], near[b]) and gap[a, b] > 0), reverse=True)
    nn_gap = np.array([abs(y[idx][np.argsort(d[a])[1:6]] - y[idx][a]).mean() for a in range(len(idx))])
    rnd = np.random.default_rng(0)
    rand_gap = np.array([abs(y[idx][rnd.choice(len(idx), 5)] - y[idx][a]).mean() for a in range(len(idx))])
    print(f"offset difference to the 5 most similar photos: {nn_gap.mean():.2f} px; to 5 random ones: {rand_gap.mean():.2f} px")
    (run / "width_check.json").write_text(json.dumps({"report": report, "photos": [
        {k: o[k] for k in ("id", "fold", "type", "dice", "offset", "level", "level_model", "guess")} for o in out]}) + "\n")

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.3))
    a.scatter(levm[sf], lev[sf], s=8, color="#3a6ea5", alpha=0.5)
    a.plot([0.1, 0.8], [0.1, 0.8], color="#c0392b", ls="--", lw=1.5)
    a.set(xlabel="model's outline: place on the brightness slope\n(1 = as bright as the nucleus, 0 = as dark as the background)",
          ylabel="hand-drawn outline: place on the slope", xlim=(0.1, 0.8), ylim=(0.1, 0.8))
    a.set_title(f"One dot per small fluorescence photo. The model follows\nthe answer's outline only loosely (correlation "
                f"{np.corrcoef(levm[sf], lev[sf])[0, 1]:.2f})", loc="left", weight="bold", fontsize=10)
    b.hist(y[sf], np.linspace(-3, 3, 41), color="#a0a0a0")
    b.axvline(0, color="#555", lw=1)
    b.set(xlabel="pixels the model draws wider (+) or narrower (−) than the answer", ylabel="photos")
    b.set_title(f"Small fluorescence photos: the model is off\nby {y[sf].std():.1f} px either way", loc="left", weight="bold", fontsize=10)
    best = max(report, key=lambda n: report[n]["r2"])
    gb = np.array([o["guess"][best] for o in out])
    c.scatter(gb[sf], y[sf], s=8, color="#3a6ea5", alpha=0.5)
    c.plot([-2, 2], [-2, 2], color="#c0392b", ls="--", lw=1.5)
    c.set(xlabel=f"offset guessed from the photo alone ({best}, px)", ylabel="true offset (px)", xlim=(-2.5, 2.5), ylim=(-3.1, 3.1))
    c.set_title(f"Guessing the offset from the photo explains\n{100 * report[best]['r2_small_fluo']:.0f}% of it (dots on the red line = perfect)",
                loc="left", weight="bold", fontsize=10)
    for ax in (a, b, c):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "7-outline-width.png", dpi=110)

    used, chosen = set(), []
    for g_, i, j in pairs:
        if i not in used and j not in used:
            chosen.append((i, j)); used |= {i, j}
        if len(chosen) == 4:
            break
    fig, axes = plt.subplots(4, 4, figsize=(11, 11.4))
    for row, (i, j) in zip(axes, chosen):
        for ax2, k in zip((row[:2], row[2:]), (i, j)):
            o = out[k]
            img, gt = S[o["id"]]
            show = img.astype(float)
            show = np.clip(show / max(1.0, np.percentile(show, 99.8)) * 255, 0, 255).astype(np.uint8)
            over = np.zeros((*gt.shape, 3))
            pr = P[o["id"]]
            over[pr & gt] = 1; over[pr & ~gt] = (0.85, 0.2, 0.15); over[~pr & gt] = (0.2, 0.45, 0.95)
            ax2[0].imshow(show); ax2[1].imshow(over)
            ax2[0].set_title(f"{o['id'][:8]}", fontsize=8, loc="left")
            ax2[1].set_title(f"Dice {o['dice']:.3f}, model {o['offset']:+.1f} px", fontsize=8, loc="left")
            for x_ in ax2:
                x_.axis("off")
    fig.suptitle("Four pairs of photos that look alike to the measurements, with opposite outline widths\n"
                 "red = model wider than the answer, blue = model narrower", x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ROOT / "reports" / "7-look-alike-pairs.png", dpi=100)


if __name__ == "__main__":
    main(sys.argv[1])
