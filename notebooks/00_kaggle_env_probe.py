# %% [markdown]
# # Kaggle environment probe
#
# Run once on Kaggle to (1) pin dependency versions to the image, (2) confirm the GPU,
# (3) confirm the attached dataset, (4) confirm the `WANDB_API_KEY` secret works.
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
# ## W&B secret
#
# Add a Kaggle Secret named `WANDB_API_KEY` to this notebook (Add-ons → Secrets) first.

# %%
import traceback

report["wandb"] = {}
try:
    from kaggle_secrets import UserSecretsClient

    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    report["wandb"]["secret"] = "ok"
except Exception:
    report["wandb"]["secret"] = traceback.format_exc(limit=1).strip().splitlines()[-1]

if report["wandb"].get("secret") == "ok":
    try:
        import wandb

        run = wandb.init(entity="lkx100-kl-university", project="nucseg", name="env-probe",
                         job_type="probe", config={"kind": "environment probe"})
        wandb.log({"probe": 1})
        report["wandb"].update(ok=True, url=run.url)
        run.finish()
    except Exception:
        traceback.print_exc()
        report["wandb"].update(ok=False, error=traceback.format_exc(limit=1).strip().splitlines()[-1])
else:
    report["wandb"]["ok"] = False

print(report["wandb"])

# %% [markdown]
# ## Write the report

# %%
(OUT / "env.json").write_text(json.dumps(report, indent=2))
summary = (
    f"RESULT probe python={report['python']} torch={report['versions'].get('torch')} "
    f"gpu={report['gpu']['device']} vram={report['gpu']['vram_gb']}GB "
    f"inputs={','.join(inputs) or 'none'} wandb={report['wandb']['ok']} secret={report['wandb']['secret']}"
)
(OUT / "summary.txt").write_text(summary + "\n")
print(summary)
