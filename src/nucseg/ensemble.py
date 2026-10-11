"""Ensemble entry point: average the predictions of finished runs, fold by fold. No training.

    python -m nucseg.ensemble --config configs/X.yaml --run-id <id> [--smoke] [--out DIR] [--data DIR] [--ckpt-root DIR]

Each member is a finished run whose output holds ckpt/fold<k>.pt. For fold k every member's fold-k model predicts
the fold's validation images, and the members' probabilities are averaged before the 0.5 threshold. Writes
metrics.json, summary.txt, per_image.json, train.log and config.yaml into the output directory.
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from nucseg.data import ROOT, find_data_root, fold_ids, load_samples, load_splits
from nucseg.foldtable import fold_table
from nucseg.model import build_model
from nucseg.predict import predict
from nucseg.train import SMOKE_LIMITS, WANDB_ENTITY, WANDB_PROJECT, git, log, mark_smoke_ok, setup_logging
from eval.metrics import binarize, cv_summary, fold_summary, image_scores  # noqa: E402 (path set by nucseg.data)

EDGE_PX = 2  # half-width of the band around the ground-truth boundary used by the edge diagnostic


def find_ckpt_dir(slug, root):
    """The `ckpt` folder of the run with this kernel slug, searched under `root`."""
    found = [p.parent for p in Path(root).rglob("fold0.pt") if p.parent.name == "ckpt"]
    match = [d for d in found if slug in d.parts]
    if len(match) != 1:
        raise FileNotFoundError(f"expected one ckpt folder for {slug} under {root}, found {match} among {found}")
    return match[0]


def load_member(member, k, ckpt_dir, device, smoke):
    cfg = yaml.safe_load((ROOT / member["config"]).read_text())
    cfg["model"]["pretrained"] = False  # the checkpoint supplies every weight
    model = build_model(cfg, smoke).to(device)
    state = torch.load(ckpt_dir / f"fold{k}.pt", map_location="cpu")
    model.load_state_dict({n: v.float() if v.is_floating_point() else v for n, v in state.items()})
    return model.eval()


def edge_stats(pred, gt):
    """Error pixels inside the band within EDGE_PX of the ground-truth boundary, and Dice with that band ignored."""
    g = torch.from_numpy(gt)[None, None].float()
    size = 2 * EDGE_PX + 1
    grown = F.max_pool2d(g, size, 1, EDGE_PX)[0, 0].bool().numpy()
    shrunk = ~F.max_pool2d(1 - g, size, 1, EDGE_PX)[0, 0].bool().numpy()
    band = grown & ~shrunk
    err = pred != gt
    return {"err": int(err.sum()), "err_edge": int((err & band).sum()),
            "dice_off_edge": image_scores(pred & ~band, gt & ~band)["dice"]}


def run_fold(k, cfg, members, ckpt_dirs, splits, data_root, device, amp, smoke):
    ev = cfg["eval"]
    t0 = time.time()
    val_ids = fold_ids(splits, k)[2]
    if smoke:
        val_ids = val_ids[:SMOKE_LIMITS["val"]]
    val_s = load_samples(data_root, val_ids)
    modes = (False, True) if ev.get("tta") else (False,)
    mean = {tta: {} for tta in modes}  # tta -> image id -> mean probability over the members
    solo = {}
    for m in members:
        model = load_member(m, k, ckpt_dirs[m["slug"]], device, smoke)
        solo[m["name"]] = {}
        for tta in modes:
            scores = []
            for i, (img, gt) in val_s.items():
                probs = predict(model, img, ev["tile"], ev["overlap"], device, amp, tta=tta)
                scores.append(image_scores(binarize(probs), gt))
                mean[tta][i] = mean[tta].get(i, 0) + probs / len(members)
            solo[m["name"]]["tta_dice" if tta else "dice"] = fold_summary(scores)["dice"]
        del model

    scores = {tta: [{"id": i, **image_scores(binarize(mean[tta][i]), gt)} for i, (_, gt) in val_s.items()]
              for tta in modes}
    edge = [edge_stats(binarize(mean[False][i]), gt) for i, (_, gt) in val_s.items()]
    fold = {"fold": k, **fold_summary(scores[False]), "members": solo,
            "dice_off_edge": float(np.mean([e["dice_off_edge"] for e in edge])),
            "err": sum(e["err"] for e in edge), "err_edge": sum(e["err_edge"] for e in edge)}
    if True in modes:
        fold["tta"] = fold_summary(scores[True])
    fold["minutes"] = round((time.time() - t0) / 60, 1)
    log.info(f"fold {k}: ensemble dice {fold['dice']:.4f}"
             + (f" tta_dice {fold['tta']['dice']:.4f}" if "tta" in fold else "")
             + " members " + " ".join(f"{n}={s['dice']:.4f}" for n, s in solo.items())
             + f" edge_err_share {fold['err_edge'] / max(1, fold['err']):.3f} ({fold['minutes']} min)")
    return fold, scores, edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--data")
    ap.add_argument("--ckpt-root", default="/kaggle/input")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out or ROOT / "runs" / args.run_id)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")
    setup_logging(out)
    commit = os.environ.get("NUCSEG_COMMIT") or git("rev-parse", "HEAD") or "unknown"

    if args.smoke:
        device = torch.device("cpu")
        cfg["folds"] = [0]
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        sys.exit("No GPU found. Real runs need a GPU; use --smoke for a CPU check.")
    amp = device.type == "cuda"

    members = cfg["members"]
    ckpt_dirs = {m["slug"]: find_ckpt_dir(m["slug"], args.ckpt_root) for m in members}
    data_root = find_data_root(args.data)
    splits = load_splits()
    log.info(f"run {args.run_id} commit {commit[:7]} device {device} data {data_root} smoke={args.smoke} "
             f"ckpt {' '.join(str(d) for d in ckpt_dirs.values())}")

    wb = None  # smoke skips W&B unless WANDB_MODE is set explicitly
    if os.environ.get("WANDB_MODE", "disabled" if args.smoke else "offline") != "disabled":
        import wandb
        wb = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=args.run_id, job_type="ensemble",
                        config={**cfg, "commit": commit, "run_id": args.run_id})

    t0 = time.time()
    folds, per_image, by_type, by_type_tta, off_edge = [], [], defaultdict(list), defaultdict(list), defaultdict(list)
    for k in cfg["folds"]:
        fold, scores, edge = run_fold(k, cfg, members, ckpt_dirs, splits, data_root, device, amp, args.smoke)
        folds.append(fold)
        tta = {s["id"]: s["dice"] for s in scores.get(True, [])}
        for s, e in zip(scores[False], edge):
            t = splits["image_type"][s["id"]]
            by_type[t].append(s["dice"])
            off_edge[t].append(e["dice_off_edge"])
            if s["id"] in tta:
                by_type_tta[t].append(tta[s["id"]])
            per_image.append({"id": s["id"], "fold": k, "type": t, "dice": s["dice"], "dice_tta": tta.get(s["id"]),
                              **e})

    cv = cv_summary(folds)
    if all("tta" in f for f in folds):
        cv["tta"] = cv_summary([f["tta"] for f in folds])
    names = [m["name"] for m in members]
    edge = {"px": EDGE_PX, "error_share": sum(f["err_edge"] for f in folds) / max(1, sum(f["err"] for f in folds)),
            "dice_off_edge": float(np.mean([f["dice_off_edge"] for f in folds])),
            "dice_off_edge_by_type": {t: float(np.mean(v)) for t, v in sorted(off_edge.items())}}
    metrics = {
        "run_id": args.run_id, "commit": commit, "config": args.config, "smoke": args.smoke, "members": names,
        "cv": cv, "member_dice": {n: float(np.mean([f["members"][n]["dice"] for f in folds])) for n in names},
        "dice_by_type": {t: float(np.mean(v)) for t, v in sorted(by_type.items())},
        "dice_by_type_tta": {t: float(np.mean(v)) for t, v in sorted(by_type_tta.items())},
        "edge": edge, "folds": folds, "minutes": round((time.time() - t0) / 60, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0,
        "versions": {"torch": torch.__version__, "timm": __import__("timm").__version__},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    (out / "per_image.json").write_text(json.dumps(per_image) + "\n")
    summary = (f"RESULT run={args.run_id} dice={cv['dice']:.4f}±{cv['dice_std']:.4f} iou={cv['iou']:.4f} "
               f"pixacc={cv['pixacc']:.4f} dice_pooled={cv['dice_pooled']:.4f} "
               + (f"dice_tta={cv['tta']['dice']:.4f}±{cv['tta']['dice_std']:.4f} " if "tta" in cv else "")
               + "members=" + ",".join(f"{n}:{metrics['member_dice'][n]:.4f}" for n in names)
               + f" edge_err_share={edge['error_share']:.3f} dice_off_edge={edge['dice_off_edge']:.4f} "
               f"vram={metrics['peak_vram_gb']}GB time={metrics['minutes']}m")
    (out / "summary.txt").write_text(f"{summary}\n\n{fold_table(folds)}\n")
    log.info(summary)
    if wb:
        wb.summary.update({f"cv/{m}": v for m, v in cv.items()} | {f"edge/{m}": v for m, v in edge.items()})
        wb.finish()

    if args.smoke:
        mark_smoke_ok(commit)


if __name__ == "__main__":
    main()
