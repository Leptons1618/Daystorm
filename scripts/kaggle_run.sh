#!/usr/bin/env bash
# Push the distributed Stage B run to Kaggle and pull the logs back.
#
# One-time: open the kernel in the browser after the first push and set the
# accelerator to "GPU T4 x2". `enable_gpu` requests a GPU but not which one,
# and the kernel refuses to run on a single GPU rather than emit a fake number.
set -euo pipefail

ID="$(python -c 'import json;print(json.load(open("kaggle/kernel-metadata.json"))["id"])')"

echo "--> pushing ${ID}"
kaggle kernels push -p kaggle/

echo "--> waiting for completion (ctrl-c is safe; the run continues server-side)"
until kaggle kernels status "${ID}" 2>&1 | grep -qE "complete|error|cancel"; do
  sleep 30
done
kaggle kernels status "${ID}"

mkdir -p reports/kaggle
kaggle kernels output "${ID}" -p reports/kaggle
echo "--> logs and checkpoints in reports/kaggle/"
