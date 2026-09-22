# %% [markdown]
# # Kaggle environment probe
#
# Run once on Kaggle to (1) pin dependency versions to the image, (2) confirm the GPU,
# (3) confirm the attached dataset, (4) confirm W&B offline logging works.
# Writes `env.json` + `summary.txt` to `/kaggle/working/`, which `kaggle kernels output` pulls.

# %%
import json, os, platform, subprocess, sys
from pathlib import Path

OUT = Path("/kaggle/working")
report = {"python": platform.python_version(), "platform": platform.platform()}

# %% [markdown]
# ## Versions of the libraries we care about

# %%
import importlib.metadata as md

PKGS = ["torch", "torchvision", "numpy", "pillow", "scikit-image", "opencv-python",
        "albumentations", "pyyaml", "matplotlib", "tifffile", "wandb", "pandas", "scipy"]
report["versions"] = {}
for p in PKGS:
    try:
        report["versions"][p] = md.version(p)
    except md.PackageNotFoundError:
        report["versions"][p] = None
print(json.dumps(report["versions"], indent=2))

# %% [markdown]
# ## GPU

# %%
import torch

report["gpu"] = {
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    if torch.cuda.is_available()
    else None,
    "torch_cuda": torch.version.cuda,
}
print(report["gpu"])
assert torch.cuda.is_available(), "No GPU: turn on the accelerator for this kernel"

# %% [markdown]
# ## Attached data
#
# Attach the DSB2018 dataset to this notebook in the Kaggle UI first (Add Input).

# %%
inputs = sorted(p.name for p in Path("/kaggle/input").glob("*")) if Path("/kaggle/input").exists() else []
report["inputs"] = inputs
for name in inputs:
    files = list(Path("/kaggle/input", name).rglob("*"))
    print(name, len(files), "entries; first few:", [str(f.relative_to(Path('/kaggle/input', name))) for f in files[:5]])

# %% [markdown]
# ## W&B (offline)
#
# Kaggle Secrets don't reach CLI-pushed runs, so W&B logs offline into /kaggle/working/wandb/.
# `launch/kaggle/pull.sh` downloads that folder and uploads it with `wandb sync` on the local machine.

# %%
import traceback

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = str(OUT)
report["wandb"] = {}
try:
    import wandb

    run = wandb.init(entity="lkx100-kl-university", project="nucseg", name="env-probe",
                     job_type="probe", config={"kind": "environment probe"})
    for step in range(3):
        wandb.log({"probe": step})
    report["wandb"].update(ok=True, run_id=run.id)
    run.finish()
except Exception:
    traceback.print_exc()
    report["wandb"].update(ok=False, error=traceback.format_exc(limit=1).strip().splitlines()[-1])
print(report["wandb"])

# %% [markdown]
# ## Write the report

# %%
(OUT / "env.json").write_text(json.dumps(report, indent=2))
summary = (
    f"RESULT probe python={report['python']} torch={report['versions'].get('torch')} "
    f"gpu={report['gpu']['device']} vram={report['gpu']['vram_gb']}GB "
    f"inputs={','.join(inputs) or 'none'} wandb_offline={report['wandb']['ok']}"
)
(OUT / "summary.txt").write_text(summary + "\n")
print(summary)
