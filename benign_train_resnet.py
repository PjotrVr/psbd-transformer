from __future__ import annotations

import json
import os
from dataclasses import dataclass

import lightning as L
import torch
import torch.nn.functional as F
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from models import resnet18_v2
from utils import get_timestamp, load_tensor


@dataclass
class ResNetTrainConfig:
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    milestones: tuple[int, ...] = (50, 75)
    max_epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4
    early_stopping_patience: int = 15
    seed: int = 0
    precision: str = "16-mixed"


class LightningResNetV2(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        milestones: tuple[int, ...],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet18_v2(num_classes=num_classes)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.milestones = tuple(int(milestone) for milestone in milestones)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def shared_step(self, batch):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.shared_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(self.milestones)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def tensor_loader(
    data: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(data, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_trainer(
    run_dir: str, config: ResNetTrainConfig
) -> tuple[L.Trainer, ModelCheckpoint]:
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        monitor="val_acc",
        mode="max",
        filename="resnet18-v2-{epoch:02d}-{val_acc:.4f}",
        save_top_k=1,
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=int(config.early_stopping_patience),
        min_delta=0.0,
    )

    logger = CSVLogger(save_dir=run_dir, name="logs")

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        max_epochs=int(config.max_epochs),
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=25,
        enable_progress_bar=True,
    )
    return trainer, checkpoint_callback


def train_resnet_v2(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    val_data: torch.Tensor,
    val_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    run_name_prefix: str,
    num_classes: int = 10,
    config: ResNetTrainConfig | None = None,
) -> dict:
    if config is None:
        config = ResNetTrainConfig()

    seed_everything(config.seed, workers=True)

    run_name = f"{run_name_prefix}_{get_timestamp()}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_loader = tensor_loader(
        train_data,
        train_labels,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = tensor_loader(
        val_data,
        val_labels,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = tensor_loader(
        test_data,
        test_labels,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = LightningResNetV2(
        num_classes=num_classes,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        milestones=config.milestones,
    )

    trainer, checkpoint_callback = create_trainer(run_dir=run_dir, config=config)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_checkpoint_path = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )
    best_model = LightningResNetV2.load_from_checkpoint(best_checkpoint_path)

    test_metrics = trainer.test(best_model, dataloaders=test_loader, verbose=True)[0]

    result = {
        "run_dir": run_dir,
        "run_name": run_name,
        "best_checkpoint_path": best_checkpoint_path,
        "test_metrics": test_metrics,
        "config": {
            "learning_rate": config.learning_rate,
            "momentum": config.momentum,
            "weight_decay": config.weight_decay,
            "milestones": list(config.milestones),
            "max_epochs": config.max_epochs,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "early_stopping_patience": config.early_stopping_patience,
            "seed": config.seed,
            "precision": config.precision,
        },
    }

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    return result


def main() -> None:
    data_dir = os.path.join("preprocessed_data", "cifar10_benign")
    train_max_epochs = int(os.environ.get("TRAIN_MAX_EPOCHS", "100"))

    train_data = load_tensor(data_dir, "train_data.pt")
    train_labels = load_tensor(data_dir, "train_labels.pt")
    val_data = load_tensor(data_dir, "val_data.pt")
    val_labels = load_tensor(data_dir, "val_labels.pt")
    test_data = load_tensor(data_dir, "test_data.pt")
    test_labels = load_tensor(data_dir, "test_labels.pt")

    config = ResNetTrainConfig(
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        milestones=(50, 75),
        max_epochs=train_max_epochs,
        batch_size=128,
        num_workers=4,
        early_stopping_patience=15,
        seed=0,
        precision="16-mixed",
    )

    result = train_resnet_v2(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        test_data=test_data,
        test_labels=test_labels,
        run_name_prefix="benign_resnet18v2_cifar10",
        num_classes=10,
        config=config,
    )

    print("Run dir:", result["run_dir"])
    print("Best checkpoint:", result["best_checkpoint_path"])
    print("Test metrics:", result["test_metrics"])


if __name__ == "__main__":
    main()
