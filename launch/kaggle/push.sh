#!/usr/bin/env bash
# Push a jupytext notebook to Kaggle as a kernel.
#   launch/kaggle/push.sh notebooks/00_kaggle_env_probe.py env-probe
# Env vars: GPU=1 (accelerator on), DATASETS="owner/slug,owner/slug2", TITLE="..."
#
# Training runs set CONFIG=configs/X.yaml and use notebooks/train_kernel.py. Then the push refuses
# unless the tree is clean and `--smoke` passed on HEAD (marker in runs/.smoke-ok/). The committed
# src/, eval/, configs/ and pyproject.toml go into the notebook as a bundle, so the kernel runs
# exactly HEAD. The run ID is <date>-<short-commit>-<slug>.
#
# NOTE: dataset_sources in the metadata is authoritative on push. Datasets attached in the
# web UI are dropped unless they are listed in DATASETS here. Kaggle Secrets are attached
# to the kernel in the UI and are not part of this metadata.
set -euo pipefail

nb=${1:?usage: push.sh <notebook.py> <slug>}
slug=${2:?usage: push.sh <notebook.py> <slug>}
user=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/credentials.json')))['username'])")
stage=".kaggle-stage/$slug"

rm -rf "$stage" "$stage.b64" && mkdir -p "$stage"

run_id=""
if [ -n "${CONFIG:-}" ]; then
  [ -f "$CONFIG" ] || { echo "no such config: $CONFIG" >&2; exit 1; }
  [ -z "$(git status --porcelain)" ] || { echo "refusing: uncommitted changes" >&2; exit 1; }
  commit=$(git rev-parse HEAD)
  [ -e "runs/.smoke-ok/$commit" ] || { echo "refusing: no passing --smoke on ${commit:0:7} (run it on the clean tree)" >&2; exit 1; }
  run_id="$(date +%Y%m%d)-${commit:0:7}-$slug"
  git archive --format=tar.gz HEAD src eval configs pyproject.toml | base64 -w0 > "$stage.b64"
fi

jupytext --to ipynb "$nb" -o "$stage/$slug.ipynb" >/dev/null
# jupytext writes no kernelspec, and Kaggle rejects a notebook without one.
# Training runs also get a first cell that defines the run and unpacks the code bundle.
python3 - "$stage/$slug.ipynb" "$run_id" "${CONFIG:-}" "${commit:-}" "$stage.b64" <<'PATCH'
import json, sys
from pathlib import Path
p, run_id, config, commit, bundle = sys.argv[1:]
nb = json.load(open(p))
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
if run_id:
    src = (f"# Injected by launch/kaggle/push.sh\nRUN_ID = {run_id!r}\nCONFIG = {config!r}\nCOMMIT = {commit!r}\n"
           "import base64, io, tarfile\nfrom pathlib import Path\nCODE = Path('/tmp/nucseg-code')\n"
           f"BUNDLE = {Path(bundle).read_text()!r}\n"
           "tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode='r:gz').extractall(CODE, filter='data')\n"
           "print(RUN_ID, CONFIG, COMMIT[:7])\n")
    nb["cells"].insert(0, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})
json.dump(nb, open(p, "w"))
PATCH

ds_json=$(python3 -c "import sys,json;print(json.dumps([d for d in sys.argv[1].split(',') if d]))" "${DATASETS:-}")
cat > "$stage/kernel-metadata.json" <<JSON
{
  "id": "$user/$slug",
  "title": "${TITLE:-$slug}",
  "code_file": "$slug.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": $([ "${GPU:-0}" = "1" ] && echo true || echo false),
  "enable_internet": true,
  "dataset_sources": $ds_json,
  "competition_sources": [],
  "kernel_sources": []
}
JSON

kaggle kernels push -p "$stage"
echo "watch: kaggle kernels status $user/$slug"
[ -z "$run_id" ] || echo "run_id: $run_id  (pull: launch/kaggle/pull.sh $slug runs/$run_id)"
