# nucseg agent instructions

Nuclei instance segmentation (DSB2018). Read `WORKFLOW.md` before implementing anything.

## Hard rules
- **No training on this machine.** It is low-end and has no GPU. Locally: edits, `--smoke` (2 batches, tiny model, seconds of CPU), and reading pulled results. All real runs go to Kaggle.
- **`eval/` is frozen.** Never edit the metric code or the split definitions. Ask instead.
- **Smoke before launch.** No remote launch unless `--smoke` passed on the current commit, and the tree is clean.
- **No secrets anywhere near the repo or the chat.** The Kaggle token lives in `~/.kaggle/credentials.json`. The W&B key exists only on this machine, from `wandb login`.
- **W&B runs offline on Kaggle.** Kaggle Secrets don't reach kernels pushed from the CLI. Training sets `WANDB_MODE=offline`, and `launch/kaggle/pull.sh` uploads the run with `wandb sync` after downloading the outputs.
- **Never hardcode dataset paths.** Kaggle now mounts datasets at `/kaggle/input/datasets/<owner>/<slug>/`, not `/kaggle/input/<slug>/`. Code finds the dataset folder by searching under `/kaggle/input/`.
- **Read summaries, not logs.** `summary.txt`, then `metrics.json`, then the plots. `train.log` only with `tail`/`grep`, and only after a crash.
- **No literature surveys** unless asked. The focus is experimentation and model building.
- **Keep the docs lean.** No stale or duplicated information. Each fact lives in one file.
- **Write plainly.** Docs, the journal, commit messages and replies to the user follow the `unslop` skill (`~/.agents/skills/unslop/SKILL.md`). Explain in plain words first and add technical detail only where it's needed.

## Layout
- `src/nucseg/`: code
- `configs/`: YAML configs
- `eval/`: the frozen metric and splits
- `specs/`: one file per experiment series
- `launch/`: backend scripts (`kaggle/push.sh`, `kaggle/pull.sh`)
- `notebooks/`: jupytext `# %%` `.py` files only
- `runs/<run_id>/`: pulled outputs

`data/` and `runs/` are gitignored.

## Commands
```bash
uv sync                                  # environment (Python 3.12, CPU-only torch)
uv run python -m nucseg.train --config configs/X.yaml --run-id <id> --smoke
DATASETS=sindhu9642/nuclei-seg-corrected GPU=1 launch/kaggle/push.sh notebooks/X.py <slug>
launch/kaggle/pull.sh <slug> runs/<run_id>   # wait, download, wandb sync
kaggle kernels status luckyx100/<slug>   # kaggle and jupytext are uv tools, no `uv run`
```

## After every run
Append a row to `results.tsv`, add 2–5 lines to `JOURNAL.md`, and record the outcome in the spec. Run IDs are `<date>-<short-commit>-<slug>` and are the join key across `results.tsv`, W&B and git.
