# Research journal

Append-only. Newest entries at the bottom. Read on demand, not every session.

## 2026-09-22 — repo setup
Project renamed to `nucseg` and pushed to github.com/lkx100/nucseg. Kaggle is the only GPU
backend; Modal dropped. uv project on Python 3.12 to match Kaggle's runtime; `kaggle` and
`jupytext` as uv tools. W&B chosen for live curves (entity `lkx100-kl-university`).
Next: Kaggle env probe kernel to pin dependency versions, then freeze `eval/`.
