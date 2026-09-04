#!/usr/bin/env bash
# Run a training stage remotely and pull the artifacts back.
set -euo pipefail
STAGE="${1:?usage: colab_train.sh <stage_a|stage_b> [extra args...]}"
shift || true
REMOTE="${DAYSTORM_REMOTE:-/content/daystorm}"
DRIVE=/content/drive/MyDrive/daystorm

bash scripts/colab_sync.sh
colab exec "cd ${REMOTE} && python -m daystorm.train.${STAGE} --out ${DRIVE}/ckpt $*"
colab download "${DRIVE}/ckpt" ./ckpt
echo "--> stop the VM when you are done:  colab stop"
