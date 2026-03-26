#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export TRAIN_MAX_EPOCHS=3

echo "[1/3] Training benign ResNet18-v2 (CIFAR10) for smoke run..."
python benign_train_resnet.py

echo "[2/3] Training badnet ResNet18-v2 (CIFAR10) for smoke run..."
python badnet_train_resnet.py

echo "[3/3] Training wanet ResNet18-v2 (CIFAR10) for smoke run..."
python wanet_train_resnet.py

echo "Smoke run complete."
