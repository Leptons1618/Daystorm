#!/usr/bin/env bash
# Push the kernel to Kaggle, wait for it, and pull the outputs back.
#
#   scripts/kaggle_run.sh                 # run whatever SECTIONS says
#   scripts/kaggle_run.sh --only dist     # run just the distributed comparison
#   scripts/kaggle_run.sh --only all      # the whole pipeline, ~90 minutes
#
# Sections: smoke latency onnx stagea dist real ladder. They are independent - the
# Stage A checkpoint the distributed sections need ships in the source dataset -
# so re-running measured work to reach unmeasured work is never necessary. That
# mistake cost four separate ~90-minute runs.
#
# Source first: scripts/kaggle_push_src.sh, or the kernel runs the previous
# version of src/daystorm and a newly added flag comes back as "unrecognized
# arguments" from inside a subprocess whose output nobody is reading.
#
# The accelerator is pinned by "machine_shape": "NvidiaTeslaT4" in
# kernel-metadata.json. Without that field a push silently resets to Kaggle's
# default P100, which is sm_60 and below the sm_70 floor of Kaggle's own torch.
set -euo pipefail

KERNEL="kaggle/daystorm_fsdp_kernel.py"
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PUSH_DIR="kaggle"
if [ -n "${ONLY}" ]; then
  # Push from a copy. An earlier version edited the kernel in place and restored
  # it from a .bak on exit, which quietly reverted every source change made
  # while a run was in flight - a run takes minutes and that is exactly when the
  # next edit gets written. The section list is a per-run choice, so it belongs
  # in a throwaway directory, not in the file under version control.
  PUSH_DIR="$(mktemp -d)/kaggle"
  mkdir -p "${PUSH_DIR}"
  trap 'rm -rf "$(dirname "${PUSH_DIR}")"' EXIT
  cp kaggle/kernel-metadata.json "${KERNEL}" "${PUSH_DIR}/"
  sed -i "s|^SECTIONS = .*# RUN_SECTIONS$|SECTIONS = \"${ONLY}\"  # RUN_SECTIONS|" \
    "${PUSH_DIR}/$(basename "${KERNEL}")"
  grep -n "# RUN_SECTIONS$" "${PUSH_DIR}/$(basename "${KERNEL}")"
fi

# sed, not python: this script has to work from a bare shell, and `python` is
# not on PATH on Windows/Git Bash where the repo is developed.
ID="$(sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' kaggle/kernel-metadata.json | head -1)"

echo "--> pushing ${ID}"
kaggle kernels push -p "${PUSH_DIR}"

echo "--> waiting for completion (ctrl-c is safe; the run continues server-side)"
until kaggle kernels status "${ID}" 2>&1 | grep -qiE "complete|error|cancel"; do
  sleep 30
done
kaggle kernels status "${ID}"

OUT="reports/kaggle_run"
mkdir -p "${OUT}"
kaggle kernels output "${ID}" -p "${OUT}"
echo "--> outputs in ${OUT}/ ; read ${OUT}/console.log first"
