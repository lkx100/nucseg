# %% [markdown]
# # Training kernel
#
# Generic Kaggle runner for `nucseg.train`. Don't run it by hand. `launch/kaggle/push.sh` (with `CONFIG=` set)
# prepends a cell that defines `RUN_ID`, `CONFIG`, `COMMIT` and unpacks the committed code into `CODE`.
# Everything the job writes goes to `/kaggle/working/`, which `launch/kaggle/pull.sh` downloads.

# %%
import importlib.metadata as md
import os
import subprocess
import sys
import tomllib

# timm must match the local pin in pyproject.toml; the Kaggle image may carry another version or none.
pins = {d.split("==")[0]: d.split("==")[1] for d in tomllib.loads((CODE / "pyproject.toml").read_text())["project"]["dependencies"] if "==" in d}
try:
    have = md.version("timm")
except md.PackageNotFoundError:
    have = None
if have != pins["timm"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"timm=={pins['timm']}"], check=True)
print("timm", pins["timm"], "(image had", have, ")")

# %%
env = dict(os.environ, WANDB_MODE="offline", WANDB_DIR="/kaggle/working", PYTHONPATH=str(CODE / "src"),
           NUCSEG_COMMIT=COMMIT)
cmd = [sys.executable, "-m", "nucseg.train", "--config", CONFIG, "--run-id", RUN_ID, "--out", "/kaggle/working"]
proc = subprocess.Popen(cmd, cwd=CODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="")
assert proc.wait() == 0, f"training failed with exit code {proc.returncode}"
