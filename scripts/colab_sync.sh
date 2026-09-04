#!/usr/bin/env bash
# Push local source to a Colab session. Code is edited locally; the VM is disposable.
set -euo pipefail
REMOTE="${DAYSTORM_REMOTE:-/content/daystorm}"

colab sessions | grep -q . || { echo "no session; run: colab new"; exit 1; }
echo "--> syncing src/ scripts/ configs/ to ${REMOTE}"
colab upload src      "${REMOTE}/src"
colab upload scripts  "${REMOTE}/scripts"
colab upload configs  "${REMOTE}/configs"
colab upload pyproject.toml "${REMOTE}/pyproject.toml"
echo "--> done. install once per session with: colab install -e ${REMOTE}"
