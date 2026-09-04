#!/usr/bin/env bash
# One-time setup for a fresh Colab VM. Check flags with `colab help <cmd>`;
# the CLI is young and its options move.
set -euo pipefail
colab new
colab drivemount /content/drive
colab install torch transformers accelerate peft bitsandbytes datasets mlflow
mkdir -p /content/drive/MyDrive/daystorm/{ckpt,cache,norm}
echo "--> checkpoints will survive preemption at /content/drive/MyDrive/daystorm/ckpt"
