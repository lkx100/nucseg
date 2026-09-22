#!/usr/bin/env bash
# Wait for a Kaggle kernel to finish, download its outputs, and upload any offline W&B runs.
#   launch/kaggle/pull.sh <slug> <out_dir>
# Prints the kernel status and the W&B run URLs. The W&B login lives in the local wandb config,
# never on Kaggle.
set -euo pipefail

slug=${1:?usage: pull.sh <slug> <out_dir>}
out=${2:?usage: pull.sh <slug> <out_dir>}
user=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/credentials.json')))['username'])")

while true; do
  status=$(kaggle kernels status "$user/$slug" 2>&1)
  case "$status" in
    *COMPLETE*|*ERROR*|*CANCEL*) break ;;
  esac
  sleep 30
done
echo "$status"

rm -rf "$out" && mkdir -p "$out"
kaggle kernels output "$user/$slug" -p "$out" >/dev/null

shopt -s nullglob
for run_dir in "$out"/wandb/offline-run-*; do
  if ! log=$(wandb sync "$run_dir" 2>&1); then
    echo "W&B sync failed for $run_dir (retry: wandb sync $run_dir)" >&2
    echo "$log" | tail -2 >&2
    continue
  fi
  echo "$log" | grep -Eo "https://wandb.ai/[^ ]+" | tail -1 || true
done
