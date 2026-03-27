#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

datasets=(cifar10 cifar100 gtsrb)

total=$(( ${#datasets[@]} * 3 ))
step=1

for dataset in "${datasets[@]}"; do
	echo "[$step/$total] Smoke: benign ResNet18-v2 ($dataset), max_epochs=5, patience=5..."
	python benign_train_resnet.py \
		--dataset-name "$dataset" \
		--max-epochs 5 \
		--early-stopping-patience 5
	step=$((step + 1))

	echo "[$step/$total] Smoke: badnet ResNet18-v2 ($dataset), max_epochs=5, patience=5, PSBD on..."
	python badnet_train_resnet.py \
		--dataset-name "$dataset" \
		--max-epochs 5 \
		--early-stopping-patience 5 \
		--run-psbd
	step=$((step + 1))

	echo "[$step/$total] Smoke: wanet ResNet18-v2 ($dataset), max_epochs=5, patience=5, PSBD on..."
	python wanet_train_resnet.py \
		--dataset-name "$dataset" \
		--max-epochs 5 \
		--early-stopping-patience 5 \
		--run-psbd
	step=$((step + 1))
done

echo "Smoke run complete for: ${datasets[*]}"
