python train_on_benign_set.py -dataset cifar10 -no_aug -no_normalize

python train_on_poison_set.py -dataset cifar10 -poison_type badnet -poisoning_ratio 0.1 -no_aug -no_normalize

python train_on_poison_set.py -dataset cifar10 -poison_type wanet -poisoning_ratio 0.1 -no_aug -no_normalize

python train_on_benign_set.py -dataset gtsrb -no_aug -no_normalize

python train_on_poison_set.py -dataset gtsrb -poison_type badnet -poisoning_ratio 0.1 -no_aug -no_normalize

python train_on_poison_set.py -dataset gtsrb -poison_type wanet -poisoning_ratio 0.1 -no_aug -no_normalize
