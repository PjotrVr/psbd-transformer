from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Literal

import lightning as L
import torch
import torch.nn.functional as F
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from models import vit_tiny
from utils import get_timestamp, load_tensor, tensor_loader


@dataclass
class ViTTrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    max_epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4
    early_stopping_patience: int = 15
    seed: int = 0
    precision: Literal["16-mixed", "32-true"] = "16-mixed"


class LViT(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = vit_tiny(num_classes=num_classes)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)

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
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_acc", accuracy, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.shared_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.trainer.max_epochs or 1),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def train_vit(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    val_data: torch.Tensor,
    val_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    run_name_prefix: str,
    num_classes: int,
    config: ViTTrainConfig,
) -> dict:
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

    model = LViT(
        num_classes=num_classes,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        monitor="val_acc",
        mode="max",
        filename="vit-tiny-{epoch:02d}",
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
        precision=config.precision,
        max_epochs=int(config.max_epochs),
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=25,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_checkpoint_path = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )
    best_model = LViT.load_from_checkpoint(best_checkpoint_path)
    test_metrics = trainer.test(best_model, dataloaders=test_loader, verbose=True)[0]

    result = {
        "run_dir": run_dir,
        "run_name": run_name,
        "best_checkpoint_path": best_checkpoint_path,
        "test_metrics": test_metrics,
        "config": {
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
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
    parser = argparse.ArgumentParser(description="Train ViT-Tiny on benign data")
    parser.add_argument("--dataset-name", default="cifar10", type=str)
    parser.add_argument("--max-epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--early-stopping-patience", default=15, type=int)
    parser.add_argument("--learning-rate", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=0.05, type=float)
    args = parser.parse_args()

    data_dir = os.path.join("preprocessed_data", f"{args.dataset_name}_benign")
    train_data = load_tensor(data_dir, "train_data.pt")
    train_labels = load_tensor(data_dir, "train_labels.pt")
    val_data = load_tensor(data_dir, "val_data.pt")
    val_labels = load_tensor(data_dir, "val_labels.pt")
    test_data = load_tensor(data_dir, "test_data.pt")
    test_labels = load_tensor(data_dir, "test_labels.pt")

    num_classes = int(torch.unique(train_labels).numel())
    config = ViTTrainConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
    )

    result = train_vit(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        test_data=test_data,
        test_labels=test_labels,
        run_name_prefix=f"benign_vit_tiny_{args.dataset_name}",
        num_classes=num_classes,
        config=config,
    )

    print("Run dir:", result["run_dir"])
    print("Best checkpoint:", result["best_checkpoint_path"])
    print("Test metrics:", result["test_metrics"])


if __name__ == "__main__":
    main()
