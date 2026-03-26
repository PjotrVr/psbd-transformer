#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Full training settings from each script (default TRAIN_MAX_EPOCHS=100)
unset TRAIN_MAX_EPOCHS || true

echo "[1/3] Training benign ResNet18-v2 (CIFAR10)..."
python benign_train_resnet.py

echo "[2/3] Training badnet ResNet18-v2 (CIFAR10)..."
python badnet_train_resnet.py

echo "[3/3] Training wanet ResNet18-v2 (CIFAR10)..."
python wanet_train_resnet.py

echo "Full training complete."
