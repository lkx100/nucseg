#!/usr/bin/env bash
# Push a jupytext notebook to Kaggle as a kernel.
#   launch/kaggle/push.sh notebooks/00_kaggle_env_probe.py env-probe
# Env vars: GPU=1 (accelerator on), DATASETS="owner/slug,owner/slug2", TITLE="..."
#
# NOTE: dataset_sources in the metadata is authoritative on push. Datasets attached in the
# web UI are dropped unless they are listed in DATASETS here. Kaggle Secrets are attached
# to the kernel in the UI and are not part of this metadata.
set -euo pipefail

nb=${1:?usage: push.sh <notebook.py> <slug>}
slug=${2:?usage: push.sh <notebook.py> <slug>}
user=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/credentials.json')))['username'])")
stage=".kaggle-stage/$slug"

rm -rf "$stage" && mkdir -p "$stage"
jupytext --to ipynb "$nb" -o "$stage/$slug.ipynb" >/dev/null
# jupytext writes no kernelspec, and Kaggle rejects a notebook without one
python3 - "$stage/$slug.ipynb" <<'PATCH'
import json, sys
p = sys.argv[1]
nb = json.load(open(p))
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
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
