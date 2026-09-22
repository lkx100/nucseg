# AI-assisted research workflow

This is the working agreement for how Claude Code and I run ML research in this repo. It replaces the old loop of copying code into a Colab/Kaggle web UI and running cells by hand. Implementation sessions should read this file first.

---

## 1. The core idea

In web development, agents work well because the loop is **edit, run, read the result, edit again**, all on one machine. In ML research that loop breaks because the GPU is somewhere else and the only way to reach it is a browser notebook. The fix is:

> **Code lives locally, in git, as plain `.py` files. Remote GPUs are dumb executors that the agent reaches through a CLI or API. Results come back as small, structured files, not as scrollback.**

What that means in practice:

| Old way (friction) | New way |
|---|---|
| Code lives in notebook cells in a browser | Code lives in `src/` and `configs/` in a local git repo |
| Copy and paste between the LLM chat and Colab | The agent edits files directly and launches the job with one command |
| Run cells by hand and watch them | A job runs unattended, and the agent polls its status in the background |
| Paste logs back into the chat, which fills the context window | The job writes `metrics.json` plus a one-line summary, and the agent reads only those |
| Findings are lost when the tab closes | Findings go in `results.tsv`, `JOURNAL.md` and git commits |

## 2. Compute: where the jobs run

I have no local GPU, so every option below is remote. Each one differs in how easily an agent can drive it.

| Platform | Cost | Agent-drivable? | Best for |
|---|---|---|---|
| **Kaggle Kernels (CLI)** | Free, about 30 GPU-hours a week (T4 or P100) | ✅ `kaggle kernels push`, `status`, `output`. There is also an official remote MCP at `kaggle.com/mcp`. | **Default for batch training runs.** Free, runs unattended, and many datasets (including DSB2018 nuclei) are already on Kaggle. |

**Decision for this project: Kaggle is the only GPU backend.** The local machine runs `--smoke` only (2 batches, tiny model, seconds of CPU) and never trains. Paid or credit-based backends (Modal, Lightning, HF Jobs) were considered and dropped: one backend means one set of credentials and half the launcher code. `launch/run.sh` still takes a backend argument (`local`, `kaggle`), so a second backend is a new adapter file, not a rewrite. Revisit only if the 30 GPU-hours a week or the 12-hour session cap become the bottleneck.

**Kaggle limits to plan against:** about 30 GPU-hours a week, 12 hours maximum per session, a queue wait before a kernel starts, and no interactive access once it runs.

## 3. Repo layout

```
nucseg/                  # repo: github.com/lkx100/nucseg
├── AGENTS.md            # short: conventions, commands, hard rules (points here)
├── WORKFLOW.md          # this file
├── JOURNAL.md           # append-only research log (human + agent), read on demand
├── results.tsv          # one row per run, the agent's source of truth
├── specs/               # one file per experiment series: hypothesis, metric, budget
├── configs/             # YAML configs; every run is config + commit
├── src/nucseg/          # data, models, losses, train.py, eval.py
├── eval/                # FROZEN evaluation code + split definitions (agent must not edit)
├── notebooks/           # jupytext percent-format .py files only (no .ipynb in git)
├── launch/              # backend adapters: kaggle/, local
├── data/sample/         # a handful of images for local smoke tests (gitignored)
└── runs/<run_id>/       # pulled outputs: metrics.json, summary.txt, plots/, log tail
```

- **Notebooks:** write them as jupytext `# %%` percent-format `.py` files. The agent edits clean text instead of `.ipynb` JSON, and diffs stay readable. They open as notebooks in VS Code or Jupyter.
- **Dependencies:** `uv` with a lockfile, Python 3.12 to match Kaggle's runtime. Local installs are **CPU-only** (the `pytorch-cpu` index). Kaggle's image is the version reference: pins come from a `pip freeze` on Kaggle, and the kernel never reinstalls what the image already has.
- **Tools:** `kaggle` and `jupytext` are installed as uv tools, so they run directly, without `uv run`.

## 4. The job contract (the most important piece)

Every training entry point obeys the same rules, whatever backend runs it:

1. **Input:** `python -m nucseg.train --config configs/X.yaml --run-id <id> [--smoke]`
2. **Run ID:** `<date>-<short-commit>-<slug>`. Refuse to launch with uncommitted changes, so every result maps to exact code.
3. **Output directory**, always:
   - `metrics.json` holds the final and best metrics, epochs, wall time and peak VRAM.
   - `summary.txt` is **one line**, `RESULT run=… dice=0.812 aji=0.64 map=… time=…`, and is what the agent reads.
   - `train.log` is the full log. The agent only runs `tail` or `grep` on it and never reads it whole.
   - `plots/` holds loss curves and a grid of predictions against ground truth on fixed validation images.
   - Checkpoint: best only. Pulled on demand, not by default.
4. **No progress bars in logs** (`tqdm` disabled when not attached to a terminal). Print one line per epoch at most.
5. **`--smoke` mode:** 2 batches, a tiny model and CPU-OK. Must pass **locally** before any GPU launch. This single rule saves the most wasted GPU-hours and the most agent round trips.
6. **Seeds are fixed and logged.** The config is copied into the output directory.

**Launcher:** `launch/run.sh <backend> <config>` does the following:
- `local`: runs the smoke test.
- `kaggle`: generates `kernel-metadata.json` (GPU on, dataset sources, code bundled as a script or utility dataset), runs `kaggle kernels push`, then polls `kaggle kernels status` and fetches results with `kaggle kernels output` into `runs/<id>/`.

After every run a row is appended to `results.tsv` (tab-separated, because the description column contains commas):
`run_id  commit  config  backend  dice  aji  map  vram_gb  minutes  status(keep|discard|crash)  wandb  description`

**Tracking (W&B):** every non-smoke run logs to W&B under entity `lkx100-kl-university`, project `nucseg`, with the `run_id` as the W&B run name and the config and git commit attached. Smoke runs set `WANDB_MODE=disabled`. `results.tsv` stays the agent-readable source of truth. Kaggle Secrets don't reach kernels pushed from the CLI (tested 2026-09-23, see `JOURNAL.md`), so W&B runs **offline** on Kaggle and writes to `/kaggle/working/wandb/`. After a run, `launch/kaggle/pull.sh` downloads the outputs and uploads them with `wandb sync` from this machine. Curves appear in W&B once the run ends, not while it trains. The Kaggle notebook log is still visible during a run.

## 5. Experiment protocol (how a research step happens)

1. **Spec first** (`specs/NN-name.md`): hypothesis, the single variable changed, the metric and the threshold that counts as a win, and the compute budget (for example "≤ 3 Kaggle GPU-hours"). The agent does not move the goalposts once a run has started.
2. **Change one thing per run.** The config diff explains the run.
3. **Smoke test locally**, then commit, then launch remotely.
4. **Wait without watching.** The agent polls in the background (Monitor or a scheduled wake-up) instead of streaming logs into context.
5. **Read** `summary.txt`, then `metrics.json`, then the plots. Look at the prediction grid, not just the numbers.
6. **Record** the row in `results.tsv` and 2–5 lines in `JOURNAL.md`: what happened, anything surprising, what to try next.
7. **Keep or revert.** Winning changes merge into the baseline config. Losers stay in the log, because failures are the most useful data later.

## 6. Guardrails (research-specific failure modes of agents)

- **Frozen evaluation.** `eval/` (the metric code and the train/val/test split) is written once, reviewed by me, and then read-only for the agent. An agent that optimises a metric it can edit will "improve" the metric instead of the model.
- **The test set is used only at milestones and only when I ask.** Day-to-day decisions use the validation set.
- **Distrust surprisingly good results.** Any jump of more than a few points triggers a leakage check (overlap between splits, for example tiles from the same image) and a look at the plots before it goes in the journal.
- **Budget caps are stated in the spec.** The agent stops and asks when it hits one.
- **No silent fallbacks.** If the GPU is not found, the job fails loudly. It never trains quietly on CPU for 9 hours.
- **Secrets** (`~/.kaggle/credentials.json`, the local `wandb login`) stay on this machine. They never go to Kaggle, in the repo, in a notebook, or in a chat message.

## 7. Context hygiene (keeping the LLM effective)

- The agent reads **summaries, not logs**: `summary.txt`, `metrics.json`, `tail -50 train.log` only after a crash.
- `AGENTS.md` stays short: commands, rules and pointers. Detailed knowledge lives in `WORKFLOW.md`, `specs/` and `JOURNAL.md`, and is read on demand, never auto-loaded.
- **One session, one spec.** Start fresh sessions for new experiment series. `JOURNAL.md` and `results.tsv` carry the state between sessions, not chat history.
- **Retrospective at the end of a session:** the agent distils what worked and what broke (exact hyperparameters, error messages and fixes) into `JOURNAL.md` and, when a pattern repeats, into a `LEARNINGS.md` or a project skill.

## 8. Levels of autonomy

Start at L1 and move up only when the harness is trustworthy.

| Level | What the agent does | What I do |
|---|---|---|
| **L0: pair** | Writes and edits code, runs smoke tests | Launch jobs, read results |
| **L1: supervised runs** (default) | Writes the spec draft, launches and monitors, logs results, proposes the next step | Approve the spec, review plots, decide keep or revert |
| **L2: bounded loop** | Runs N experiments from an approved spec on its own, with a fixed time budget per run and keep/revert based on the validation metric (the autoresearch pattern) | Set the spec and budget, then review `results.tsv` and the journal afterwards |

L2 only makes sense once the eval is frozen, runs are short (a fixed number of minutes or epochs so results are comparable), and the search space is explicit in the spec.

## 9. Planning and the research record

**No literature surveys for now.** The focus is experimentation and model building. Don't propose paper comparisons or method surveys unless I ask.

- **Reproduce a baseline before changing anything.** The first spec is always "reproduce a known number".
- **Ablation planning:** draft the ablation grid and estimate GPU-hours per cell against the weekly Kaggle quota before anything runs.

**The record** has four layers, all joined by `run_id`, so a future report or paper can trace every number back to the code that produced it:

| Layer | Holds | Written by |
|---|---|---|
| `results.tsv` | one row per run: the numbers | the launcher, automatically |
| W&B | curves, config, system metrics, prediction grids | the training script |
| `specs/NN-*.md` | the hypothesis, plus an **Outcome** section once the series ends | the agent, I approve |
| `JOURNAL.md` | dated narrative: what we tried, why, surprises, dead ends | the agent after each run, and `/retro` at session end |

Milestone commits get a git tag, and figures worth keeping go in `reports/figures/`. When I ask for a write-up, the agent drafts it from `results.tsv` and `JOURNAL.md`, and I check every number against the source.

## 10. Small automations to build (in order of payoff)

1. `launch/run.sh` plus the job contract (§4). This is the main unlock.
2. A **smoke-test gate**: the launcher refuses a remote launch unless `--smoke` passed on the current commit.
3. **Background status polling** plus a desktop notification when a Kaggle job finishes or crashes.
4. A `results.tsv` appender plus a tiny `report.py` that prints the leaderboard and plots the metric against runs.
5. Project **slash commands or skills**: `/launch <config>`, `/status`, `/retro` (session retrospective into the journal), `/advise` (search the journal and learnings before starting a new spec).
6. W&B logging from the training script (live curves and prediction grids). `results.tsv` stays the agent-readable source of truth either way.

## 11. Setup status

Done:
- [x] `git init`, remote `github.com/lkx100/nucseg`, `uv init` with Python 3.12 (matching Kaggle's runtime)
- [x] `kaggle` and `jupytext` installed as uv tools; Kaggle CLI authenticated (`kaggle kernels list --mine`)
- [x] Kaggle env probe (`notebooks/00_kaggle_env_probe.py`): T4 GPU, dataset mount and W&B offline logging all confirmed

Next:
- [x] Repo skeleton: `AGENTS.md`, `JOURNAL.md`, `results.tsv`, `launch/kaggle/`
- [ ] Local `wandb login` on this machine, then confirm `wandb sync` of the probe run
- [ ] Pin dependencies from a Kaggle `pip freeze`; install CPU-only torch locally
- [ ] Pick the dataset (DSB2018 stage1) and **freeze the splits and metric in `eval/`**
- [ ] Spec 01: baseline U-Net, reproducing a reasonable Dice/AJI with the job contract end to end

---

### Sources
- Sionic AI, *Using Claude Code skills to run 1,000+ ML experiments a day* (TECHSPEC, `/retrospective`, `/advise`, and failures as the most useful data): https://huggingface.co/blog/sionic-ai/claude-code-skills-training
- Karpathy, *autoresearch* `program.md` (fixed time budget, `results.tsv`, keep or revert through git, output redirected to a file and read with grep): https://github.com/karpathy/autoresearch
- Eric Ma, *How to do agentic data science* (prescriptive goals, append-only journal, diagnostic plots, minimal version first): https://ericmjl.github.io/blog/2026/2/1/how-to-do-agentic-data-science/
- Martin Krasser, *Autonomous ML researcher with Claude Code workflows plus HF skills*: https://krasserm.github.io/2026/05/31/ml-research-workflow/
- Kaggle CLI kernels docs: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md and Kaggle MCP: https://www.kaggle.com/docs/mcp
- Jupytext for agents: https://www.zonca.dev/posts/2025-12-11-jupytext-ai-agents
