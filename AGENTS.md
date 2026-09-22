# nucseg — agent instructions

Nuclei instance segmentation (DSB2018). Read `WORKFLOW.md` before implementing anything.

## Hard rules
- **No training on this machine.** It is low-end and has no GPU. Locally: edits, `--smoke` (2 batches, tiny model, seconds of CPU), and reading pulled results. All real runs go to Kaggle.
- **`eval/` is frozen.** Never edit the metric code or the split definitions. Ask instead.
- **Smoke before launch.** No remote launch unless `--smoke` passed on the current commit, and the tree is clean.
- **No secrets anywhere near the repo or the chat.** Kaggle: `~/.kaggle/credentials.json`. W&B locally: `~/.netrc`. W&B on Kaggle: the `WANDB_API_KEY` Kaggle Secret.
- **Read summaries, not logs.** `summary.txt`, then `metrics.json`, then the plots. `train.log` only with `tail`/`grep`, and only after a crash.
- **No literature surveys** unless asked. The focus is experimentation and model building.
- **Keep the docs lean.** No stale or duplicated information: one fact lives in one file.

## Layout
`src/nucseg/` code · `configs/` YAML · `eval/` frozen metric + splits · `specs/` one per experiment series · `launch/` backend adapters · `notebooks/` jupytext `# %%` `.py` only · `runs/<run_id>/` pulled outputs · `data/` and `runs/` are gitignored.

## Commands
```bash
uv sync                                  # environment (Python 3.12, CPU-only torch)
uv run python -m nucseg.train --config configs/X.yaml --run-id <id> --smoke
launch/run.sh kaggle configs/X.yaml      # push, poll, pull into runs/<id>/
kaggle kernels status luckyx100/<slug>   # kaggle and jupytext are uv tools, no `uv run`
```

## After every run
Append a row to `results.tsv`, add 2–5 lines to `JOURNAL.md`, and record the outcome in the spec. Run IDs are `<date>-<short-commit>-<slug>` and are the join key across `results.tsv`, W&B and git.
