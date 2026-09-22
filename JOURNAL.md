# Research journal

Append-only. Newest entries at the bottom. Read on demand, not every session.

## 2026-09-22 — repo setup
Project renamed to `nucseg` and pushed to github.com/lkx100/nucseg. Kaggle is the only GPU
backend; Modal dropped. uv project on Python 3.12 to match Kaggle's runtime; `kaggle` and
`jupytext` as uv tools. W&B chosen for live curves (entity `lkx100-kl-university`).
Next: Kaggle env probe kernel to pin dependency versions, then freeze `eval/`.

## 2026-09-22 — Kaggle env probe (kernel `luckyx100/nucseg-env-probe`, v2)
First push failed: jupytext writes no `kernelspec`, and Kaggle rejects that
("No kernel name found in notebook"). `launch/kaggle/push.sh` now patches it in.
v2 completed on a T4 (15.6 GB). Image versions: Python 3.12.13, torch 2.10.0+cu128,
torchvision 0.25.0, numpy 2.0.2, albumentations 2.0.8, scikit-image 0.25.2, wandb 0.26.1.
W&B failed with "ConnectionError ... communicate with service" — no secret attached yet;
the probe now reports the secret lookup and the `wandb.init` separately to tell them apart.
