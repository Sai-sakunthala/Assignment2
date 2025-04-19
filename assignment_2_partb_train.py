import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import random
from collections import defaultdict


class FineTunedModel(pl.LightningModule):
    def __init__(self, num_classes=10, freeze_k=2, unfreeze_every=2, dropout_prob=0.4):
        super().__init__()
        self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.freeze_k = freeze_k
        self.unfreeze_every = unfreeze_every
        self.total_blocks = len(self.model.features)

        for i, block in enumerate(self.model.features):
            if i < freeze_k:
                for param in block.parameters():
                    param.requires_grad = False

        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def on_train_epoch_start(self):
        if self.current_epoch % self.unfreeze_every == 0:
            new_k = self.freeze_k + self.current_epoch // self.unfreeze_every
            if new_k > self.freeze_k and new_k < self.total_blocks:
                for i in range(self.freeze_k, new_k + 1):
                    for param in self.model.features[i].parameters():
                        param.requires_grad = True
                self.freeze_k = new_k + 1


def train(data_dir, batch_size, max_epochs, run_name):
    random.seed(42)
    torch.manual_seed(42)

    wandb.init(project="inaturalist_finetune", name=run_name)
    wandb_logger = WandbLogger(project="inaturalist_finetune", name=run_name)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir)
    num_classes = len(full_dataset.classes)

    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_to_indices[label].append(idx)

    train_indices, val_indices = [], []
    for label, indices in class_to_indices.items():
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

    train_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=train_transform), train_indices)
    val_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=val_transform), val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = FineTunedModel(num_classes=num_classes)

    checkpoint_cb = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision=16,
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_cb],
        gradient_clip_val=0.5
    )

    try:
        trainer.fit(model, train_loader, val_loader)
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/inaturalist_12K/train", help="Path to training dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training/validation")
    parser.add_argument("--max_epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--run_name", type=str, default="efficient_net_4", help="wandb run name")

    args = parser.parse_args()
    train(args.data_dir, args.batch_size, args.max_epochs, args.run_name)