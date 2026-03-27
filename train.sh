#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

datasets=(cifar10 cifar100 gtsrb)

total=$(( ${#datasets[@]} * 3 ))
step=1

for dataset in "${datasets[@]}"; do
	echo "[$step/$total] Training benign ResNet18-v2 ($dataset) with early stopping patience=5..."
	python benign_train_resnet.py \
		--dataset-name "$dataset" \
		--early-stopping-patience 5
	step=$((step + 1))

	echo "[$step/$total] Training badnet ResNet18-v2 ($dataset) with PSBD enabled and patience=5..."
	python badnet_train_resnet.py \
		--dataset-name "$dataset" \
		--early-stopping-patience 5 \
		--run-psbd
	step=$((step + 1))

	echo "[$step/$total] Training wanet ResNet18-v2 ($dataset) with PSBD enabled and patience=5..."
	python wanet_train_resnet.py \
		--dataset-name "$dataset" \
		--early-stopping-patience 5 \
		--run-psbd
	step=$((step + 1))
done

echo "Full training complete for: ${datasets[*]}"
