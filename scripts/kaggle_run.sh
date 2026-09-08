#!/usr/bin/env bash
# Push the distributed Stage B run to Kaggle and pull the logs back.
#
# Not one-time: EVERY push resets the accelerator to Kaggle's default P100, and
# a P100 is sm_60 while Kaggle's own torch needs sm_70. `enable_gpu` requests a
# GPU but cannot say which one, and there is no metadata field that can.
#
# So after this script pushes, open the kernel, set Accelerator to "GPU T4 x2",
# and Save & Run All. The run this script starts will fail fast on the guard in
# the kernel - that failure is the reminder, not a bug.
#
# Source first: scripts/kaggle_push_src.sh, or the kernel runs the previous
# version of src/daystorm.
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
