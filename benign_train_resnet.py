from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import lightning as L
import torch
import torch.nn.functional as F
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from models import resnet18_v2
from utils import get_timestamp, load_tensor, tensor_loader


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


class LResNetV2(L.LightningModule):
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
        self._clean_val_labels: list[torch.Tensor] = []
        self._backdoor_val_labels: list[torch.Tensor] = []
        self._backdoor_val_predictions: list[torch.Tensor] = []

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

    def on_validation_epoch_start(self):
        self._clean_val_labels = []
        self._backdoor_val_labels = []
        self._backdoor_val_predictions = []

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        loss, accuracy = self.shared_step(batch)

        images, labels = batch
        logits = self(images)
        predictions = torch.argmax(logits, dim=1)

        if int(dataloader_idx) == 0:
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )
            self.log(
                "val_acc",
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )
            self._clean_val_labels.append(labels.detach().cpu())
        else:
            self._backdoor_val_labels.append(labels.detach().cpu())
            self._backdoor_val_predictions.append(predictions.detach().cpu())

    def on_validation_epoch_end(self):
        if len(self._backdoor_val_labels) == 0:
            return

        if len(self._clean_val_labels) == 0:
            return

        clean_labels = torch.cat(self._clean_val_labels, dim=0)
        backdoor_labels = torch.cat(self._backdoor_val_labels, dim=0)
        backdoor_predictions = torch.cat(self._backdoor_val_predictions, dim=0)

        asr = (backdoor_predictions == backdoor_labels).float().mean()

        changed_mask = backdoor_labels != clean_labels
        if int(changed_mask.sum().item()) == 0:
            targeted_asr = torch.tensor(0.0)
        else:
            targeted_asr = (
                (backdoor_predictions[changed_mask] == backdoor_labels[changed_mask])
                .float()
                .mean()
            )

        self.log("asr", asr, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "targeted_asr",
            targeted_asr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

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


def create_trainer(
    run_dir: str,
    config: ResNetTrainConfig,
    monitor_metric: str = "val_acc",
) -> tuple[L.Trainer, ModelCheckpoint]:
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        monitor=monitor_metric,
        mode="max",
        filename="resnet18-v2-{epoch:02d}",
        save_top_k=1,
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,
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
    backdoor_val_data: torch.Tensor | None = None,
    backdoor_val_labels: torch.Tensor | None = None,
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

    model = LResNetV2(
        num_classes=num_classes,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        milestones=config.milestones,
    )

    backdoor_val_loader = None
    if (backdoor_val_data is not None) and (backdoor_val_labels is not None):
        backdoor_val_loader = tensor_loader(
            backdoor_val_data,
            backdoor_val_labels,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

    monitor_metric = "val_acc"

    trainer, checkpoint_callback = create_trainer(
        run_dir=run_dir,
        config=config,
        monitor_metric=monitor_metric,
    )
    val_dataloaders = val_loader
    if backdoor_val_loader is not None:
        val_dataloaders = [val_loader, backdoor_val_loader]

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_dataloaders,
    )

    best_checkpoint_path = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )
    best_model = LResNetV2.load_from_checkpoint(best_checkpoint_path)

    test_metrics = trainer.test(best_model, dataloaders=test_loader, verbose=True)[0]

    result = {
        "run_dir": run_dir,
        "run_name": run_name,
        "best_checkpoint_path": best_checkpoint_path,
        "test_metrics": test_metrics,
        "monitor_metric": monitor_metric,
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


def main():
    parser = argparse.ArgumentParser(description="Train ResNet18V2 on benign data")
    parser.add_argument("--dataset-name", default="cifar10", type=str)
    parser.add_argument("--data-root", default="preprocessed_data", type=str)
    parser.add_argument("--max-epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--early-stopping-patience", default=15, type=int)
    args = parser.parse_args()

    data_dir = os.path.join(args.data_root, f"{args.dataset_name}_benign")

    train_data = load_tensor(data_dir, "train_data.pt")
    train_labels = load_tensor(data_dir, "train_labels.pt")
    val_data = load_tensor(data_dir, "val_data.pt")
    val_labels = load_tensor(data_dir, "val_labels.pt")
    test_data = load_tensor(data_dir, "test_data.pt")
    test_labels = load_tensor(data_dir, "test_labels.pt")
    num_classes = len(torch.unique(train_labels))
    config = ResNetTrainConfig(
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        milestones=(50, 75),
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
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
        run_name_prefix=f"benign_resnet18v2_{args.dataset_name}",
        num_classes=num_classes,
        config=config,
    )

    print("Run dir:", result["run_dir"])
    print("Best checkpoint:", result["best_checkpoint_path"])
    print("Test metrics:", result["test_metrics"])


if __name__ == "__main__":
    main()
