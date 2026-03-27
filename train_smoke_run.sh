#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

datasets=(cifar10 cifar100 gtsrb)
max_epochs=5
batch_size=256

total=$(( ${#datasets[@]} * 3 ))
step=1

for dataset in "${datasets[@]}"; do
	# echo "[$step/$total] Smoke: benign ResNet18-v2 ($dataset), max_epochs=$max_epochs, patience=5..."
	# python benign_train_resnet.py \
	# 	--dataset-name "$dataset" \
	# 	--max-epochs $max_epochs \
	# 	--early-stopping-patience 5
	#   --batch-size $batch_size
	# step=$((step + 1))

	# echo "[$step/$total] Smoke: badnet ResNet18-v2 ($dataset), max_epochs=$max_epochs, patience=5, PSBD on..."
	# python badnet_train_resnet.py \
	# 	--dataset-name "$dataset" \
	# 	--max-epochs $max_epochs \
	# 	--early-stopping-patience 5 \
	# 	--psbd-dropout-rates 0.8 \
	# 	--psbd-target-fprs 0.25 \
	# 	--psbd-selection-fpr 0.25 \
	# 	--batch-size $batch_size \
	# 	--run-psbd
	# step=$((step + 1))

	echo "[$step/$total] Smoke: wanet ResNet18-v2 ($dataset), max_epochs=$max_epochs, patience=5, PSBD on..."
	python wanet_train_resnet.py \
		--dataset-name "$dataset" \
		--max-epochs $max_epochs \
		--early-stopping-patience 5 \
		--psbd-dropout-rates 0.8 \
		--psbd-target-fprs 0.25 \
		--psbd-selection-fpr 0.25 \
		--batch-size $batch_size \
		--run-psbd

	step=$((step + 1))
done

echo "Smoke run complete for: ${datasets[*]}"
