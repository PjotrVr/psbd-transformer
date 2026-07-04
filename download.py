from torchvision import datasets

datasets.CIFAR100(root="./raw_data", train=True, download=True)
datasets.CIFAR100(root="./raw_data", train=False, download=True)
datasets.CIFAR10(root="./raw_data", train=True, download=True)
datasets.CIFAR10(root="./raw_data", train=False, download=True)
datasets.GTSRB(root="./raw_data", split="train", download=True)
datasets.GTSRB(root="./raw_data", split="test", download=True)
