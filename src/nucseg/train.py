"""Training entry point (job contract in WORKFLOW.md section 4).

    python -m nucseg.train --config configs/X.yaml --run-id <id> [--smoke] [--out DIR] [--data DIR]

Trains one model per CV fold. Early stopping watches the fold's stop set. The validation fold scores
the kept weights once, at the end. Writes metrics.json, summary.txt, train.log, config.yaml, plots/
and ckpt/ into the output directory.
"""
import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from nucseg.data import ROOT, CropDataset, find_data_root, fold_ids, load_samples, load_splits
from nucseg.losses import adversarial_neighbour, bce_dice
from nucseg.model import build_model
from nucseg.predict import fgsm, predict
from eval.metrics import binarize, cv_summary, fold_summary, image_scores  # noqa: E402 (path set by nucseg.data)

WANDB_ENTITY, WANDB_PROJECT = "lkx100-kl-university", "nucseg"
SMOKE_LIMITS = dict(train=4, stop=2, val=3, batches=2, epochs=2)
log = logging.getLogger("nucseg")


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def setup_logging(out):
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out / "train.log")):
        h.setFormatter(fmt)
        log.addHandler(h)


def lr_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return f


def score(model, samples, ev, device, amp):
    """Per-image scores, with the image id attached."""
    model.eval()
    out = []
    for i, (img, gt) in samples.items():
        probs = predict(model, img, ev["tile"], ev["overlap"], device, amp)
        out.append({"id": i, **image_scores(binarize(probs), gt)})
    return out


def robust_dice(model, samples, ev, device, amp):
    model.eval()
    eps = ev["robust_eps_255"] / 255
    dice = []
    for img, gt in samples.values():
        x_adv = fgsm(model, img, gt, eps, ev["tile"], ev["overlap"], device)
        dice.append(image_scores(binarize(predict(model, x_adv, ev["tile"], ev["overlap"], device, amp)), gt)["dice"])
    return float(np.mean(dice))


def prediction_grid(model, samples, types, ev, device, amp, path):
    """One validation image per image type: image, ground truth, prediction, errors (red FP, blue FN)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = {}
    for i in sorted(samples):
        picks.setdefault(types[i], i)
    fig, axes = plt.subplots(len(picks), 4, figsize=(12, 3 * len(picks)), squeeze=False)
    for row, (t, i) in zip(axes, sorted(picks.items())):
        img, gt = samples[i]
        pred = binarize(predict(model, img, ev["tile"], ev["overlap"], device, amp))
        err = img.copy()
        err[pred & ~gt] = (255, 0, 0)
        err[~pred & gt] = (0, 80, 255)
        d = image_scores(pred, gt)["dice"]
        for ax, im, title in zip(row, (img, gt, pred, err), (f"{t}\n{i[:10]}", "ground truth", f"pred dice={d:.3f}", "FP red, FN blue")):
            ax.imshow(im, cmap="gray")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def curves_plot(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5))
    for k, h in history.items():
        a.plot(h["train_loss"], label=f"fold {k}")
        b.plot(h["stop_dice"], label=f"fold {k}")
    a.set(title="train loss", xlabel="epoch")
    b.set(title="stop-set Dice", xlabel="epoch")
    b.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def train_fold(k, cfg, splits, data_root, device, smoke, out, wb):
    tr, dcfg, ev, nsl = cfg["train"], cfg["data"], cfg["eval"], cfg["nsl"]
    train_ids, stop_ids, val_ids = fold_ids(splits, k)
    assert not set(splits["test"]) & set(train_ids + stop_ids + val_ids), "test id in a CV split"
    if smoke:
        train_ids, stop_ids, val_ids = (train_ids[:SMOKE_LIMITS["train"]], stop_ids[:SMOKE_LIMITS["stop"]],
                                        val_ids[:SMOKE_LIMITS["val"]])
    torch.manual_seed(tr["seed"] + k)
    np.random.seed(tr["seed"] + k)

    t0 = time.time()
    train_s = load_samples(data_root, train_ids)
    stop_s = load_samples(data_root, stop_ids)
    val_s = load_samples(data_root, val_ids)
    log.info(f"fold {k}: train={len(train_s)} stop={len(stop_s)} val={len(val_s)} loaded in {time.time() - t0:.0f}s")

    ds = CropDataset(train_s, dcfg["crop"], dcfg["crops_per_image"], dcfg["scale"], dcfg["brightness"], dcfg["contrast"])
    loader = torch.utils.data.DataLoader(ds, batch_size=tr["batch_size"], shuffle=True, drop_last=True,
                                         num_workers=tr["num_workers"], pin_memory=device.type == "cuda",
                                         persistent_workers=tr["num_workers"] > 0)
    steps = SMOKE_LIMITS["batches"] if smoke else len(loader)
    epochs = SMOKE_LIMITS["epochs"] if smoke else tr["max_epochs"]

    model = build_model(cfg, smoke).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tr["warmup_epochs"] * steps, epochs * steps))
    amp = tr["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    alpha, eps = nsl["alpha"], nsl["eps_255"] / 255

    hist = {"train_loss": [], "stop_dice": []}
    best, best_epoch, best_state, bad = -1.0, -1, None, 0
    for epoch in range(epochs):
        te = time.time()
        model.train()
        losses = []
        for step, (x, y) in enumerate(loader):
            if step >= steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if alpha > 0:
                x.requires_grad_(True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = bce_dice(model(x), y)
            if alpha > 0:
                x_adv = adversarial_neighbour(x, loss, eps, scaler)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    loss = loss + alpha * bce_dice(model(x_adv), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        stop_dice = float(np.mean([s["dice"] for s in score(model, stop_s, ev, device, amp)]))
        hist["train_loss"].append(float(np.mean(losses)))
        hist["stop_dice"].append(stop_dice)
        if stop_dice > best:
            best, best_epoch, bad = stop_dice, epoch, 0
            best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
        else:
            bad += 1
        log.info(f"fold {k} epoch {epoch + 1}/{epochs} loss {hist['train_loss'][-1]:.4f} stop_dice {stop_dice:.4f} "
                 f"best {best:.4f}@{best_epoch + 1} lr {sched.get_last_lr()[0]:.2e} {time.time() - te:.0f}s")
        if wb:
            wb.log({f"fold{k}/epoch": epoch + 1, f"fold{k}/train_loss": hist["train_loss"][-1],
                    f"fold{k}/stop_dice": stop_dice, f"fold{k}/lr": sched.get_last_lr()[0]})
        if bad >= tr["patience"]:
            log.info(f"fold {k}: early stop, no stop-set gain for {bad} epochs")
            break

    model.load_state_dict(best_state)
    (out / "ckpt").mkdir(exist_ok=True)
    torch.save({n: v.half() if v.is_floating_point() else v for n, v in best_state.items()}, out / "ckpt" / f"fold{k}.pt")
    scores = score(model, val_s, ev, device, amp)
    summary = fold_summary(scores)
    summary["robust_dice"] = robust_dice(model, val_s, ev, device, amp)
    prediction_grid(model, val_s, splits["image_type"], ev, device, amp, out / "plots" / f"fold{k}_preds.png")
    fold = {"fold": k, "epochs_run": len(hist["stop_dice"]), "best_epoch": best_epoch + 1, "stop_dice": best,
            **summary, "minutes": round((time.time() - t0) / 60, 1)}
    log.info(f"fold {k}: val dice {summary['dice']:.4f} iou {summary['iou']:.4f} pixacc {summary['pixacc']:.4f} "
             f"robust_dice {summary['robust_dice']:.4f} ({fold['minutes']} min)")
    if wb:
        import wandb
        wb.log({f"fold{k}/val_{m}": summary[m] for m in ("dice", "iou", "pixacc", "dice_pooled", "robust_dice")}
               | {f"fold{k}/preds": wandb.Image(str(out / "plots" / f"fold{k}_preds.png"))})
    return fold, scores, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--data")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out or ROOT / "runs" / args.run_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")
    setup_logging(out)
    commit = os.environ.get("NUCSEG_COMMIT") or git("rev-parse", "HEAD") or "unknown"

    if args.smoke:
        device = torch.device("cpu")
        cfg["train"].update(batch_size=2, num_workers=0, folds=[0])
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        sys.exit("No GPU found. Real runs need a GPU; use --smoke for a CPU check.")

    data_root = find_data_root(args.data)
    splits = load_splits()
    log.info(f"run {args.run_id} commit {commit[:7]} device {device} data {data_root} smoke={args.smoke}")

    wb = None  # smoke skips W&B unless WANDB_MODE is set explicitly (to test the logging code)
    if os.environ.get("WANDB_MODE", "disabled" if args.smoke else "offline") != "disabled":
        import wandb
        wb = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=args.run_id, job_type="train",
                        config={**cfg, "commit": commit, "run_id": args.run_id})
        for k in cfg["train"]["folds"]:
            wb.define_metric(f"fold{k}/*", step_metric=f"fold{k}/epoch")

    t0 = time.time()
    folds, history, by_type = [], {}, defaultdict(list)
    for k in cfg["train"]["folds"]:
        fold, scores, hist = train_fold(k, cfg, splits, data_root, device, args.smoke, out, wb)
        folds.append(fold)
        history[k] = hist
        for s in scores:
            by_type[splits["image_type"][s["id"]]].append(s["dice"])
        curves_plot(history, out / "plots" / "curves.png")

    cv = cv_summary(folds)
    cv["robust_dice"] = float(np.mean([f["robust_dice"] for f in folds]))
    metrics = {
        "run_id": args.run_id, "commit": commit, "config": args.config, "smoke": args.smoke,
        "cv": cv, "dice_by_type": {t: float(np.mean(v)) for t, v in sorted(by_type.items())},
        "folds": folds, "minutes": round((time.time() - t0) / 60, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0,
        "versions": {"torch": torch.__version__, "timm": __import__("timm").__version__},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    summary = (f"RESULT run={args.run_id} dice={cv['dice']:.4f}±{cv['dice_std']:.4f} iou={cv['iou']:.4f} "
               f"pixacc={cv['pixacc']:.4f} dice_pooled={cv['dice_pooled']:.4f} robust_dice={cv['robust_dice']:.4f} "
               f"epochs={'/'.join(str(f['epochs_run']) for f in folds)} vram={metrics['peak_vram_gb']}GB "
               f"time={metrics['minutes']}m")
    (out / "summary.txt").write_text(summary + "\n")
    log.info(summary)
    if wb:
        wb.summary.update({f"cv/{m}": v for m, v in cv.items()})
        wb.log({"curves": __import__("wandb").Image(str(out / "plots" / "curves.png"))})
        wb.finish()

    # Smoke gate: launch/kaggle/push.sh only launches commits with a passing smoke run on a clean tree.
    if args.smoke and git("status", "--porcelain") == "":
        marker = ROOT / "runs" / ".smoke-ok" / commit
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        log.info(f"smoke passed on clean commit {commit[:7]}")


if __name__ == "__main__":
    main()
