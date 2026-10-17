"""The fold table of a run in the form of the reference paper: one row per fold, then mean ± std, the highest fold, and
mean + std (the paper's row "Highest" is mean + std in all three of its columns).

    uv run python -m nucseg.foldtable runs/<run_id> [runs/<run_id> ...]

With several runs it also prints one line per run. Values are percentages. Std is the population std (ddof=0), as in
the paper's table; `cv.*_std` in metrics.json is the sample std.
"""
import json
import sys
from pathlib import Path

import numpy as np

COLS = (("Dice", "dice"), ("Dice TTA", "tta"), ("Dice pooled", "dice_pooled"), ("IoU", "iou"), ("Pix acc", "pixacc"))


def columns(folds):
    """Column name -> one value per fold, in percent. `Dice TTA` only if every fold was scored with TTA."""
    cols = {}
    for name, key in COLS:
        if key == "tta":
            if all("tta" in f for f in folds):
                cols[name] = np.array([100 * f["tta"]["dice"] for f in folds])
        else:
            cols[name] = np.array([100 * f[key] for f in folds])
    return cols


def fold_table(folds):
    cols = columns(folds)
    lines = ["Fold       " + "".join(f"{n:>14}" for n in cols)]
    for i, f in enumerate(folds):
        lines.append(f"{f['fold'] + 1:<11}" + "".join(f"{v[i]:>14.2f}" for v in cols.values()))
    lines.append("Mean ± std " + "".join(f"{f'{v.mean():.2f} ± {v.std():.2f}':>14}" for v in cols.values()))
    lines.append("Highest    " + "".join(f"{v.max():>14.2f}" for v in cols.values()))
    lines.append("Mean + std " + "".join(f"{v.mean() + v.std():>14.2f}" for v in cols.values()))
    return "\n".join(lines)


def main(run_dirs):
    rows = []
    for d in run_dirs:
        m = json.loads((Path(d) / "metrics.json").read_text())
        print(f"{m['run_id']}\n{fold_table(m['folds'])}\n")
        rows.append((m["run_id"], columns(m["folds"])))
    if len(rows) > 1:
        names = [n for n, _ in COLS]
        print(f"{'mean / highest':<40}" + "".join(f"{n:>16}" for n in names))
        for run_id, cols in rows:
            print(f"{run_id:<40}" + "".join(
                f"{f'{cols[n].mean():.2f} / {cols[n].max():.2f}' if n in cols else '':>16}" for n in names))


if __name__ == "__main__":
    main(sys.argv[1:])
