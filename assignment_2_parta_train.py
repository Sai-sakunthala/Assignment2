import argparse
import os
import random
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb


def get_activation(name):
    return {
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "SiLU": nn.SiLU,
        "Mish": nn.Mish
    }[name]


class CNN(pl.LightningModule):
    def __init__(self, initial_in_channels=3, num_classes=10, num_conv_layers=5, num_filters=64, kernel_size=3,
                 activation_fn=nn.SiLU, dense_neurons=256, learning_rate=1e-3, use_batchnorm=True,
                 dropout_rate=0.3, filter_organization='same', data_augmentation=True):
        super().__init__()
        self.save_hyperparameters()

        layers_conv = []
        input_channels = initial_in_channels
        current_filters = num_filters

        for i in range(num_conv_layers):
            out_channels = current_filters
            layers_conv.append(nn.Conv2d(input_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            if use_batchnorm:
                layers_conv.append(nn.BatchNorm2d(out_channels))
            layers_conv.append(activation_fn())
            if dropout_rate > 0:
                layers_conv.append(nn.Dropout(dropout_rate))
            layers_conv.append(nn.MaxPool2d(kernel_size=2, stride=2))
            input_channels = out_channels

            if filter_organization == 'double':
                current_filters *= 2
            elif filter_organization == 'half':
                current_filters = max(4, current_filters // 2)

        self.conv_block = nn.Sequential(*layers_conv)
        self.fc1 = nn.LazyLinear(dense_neurons)
        self.bn_fc1 = nn.BatchNorm1d(dense_neurons) if use_batchnorm else None
        self.activation_dense = activation_fn()
        self.dropout_fc1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.fc2 = nn.Linear(dense_neurons, num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.hparams.use_batchnorm and self.bn_fc1 is not None:
            x = self.bn_fc1(x)
        x = self.activation_dense(x)
        if self.hparams.dropout_rate > 0 and self.dropout_fc1 is not None:
            x = self.dropout_fc1(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        return [optimizer], [scheduler]


def main(args):
    random.seed(42)
    torch.manual_seed(42)

    wandb_logger = WandbLogger(project="cnn-sweep", log_model='all') if args.use_wandb else None

    transform_list = [
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ] if args.data_augmentation else [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform = transforms.Compose(transform_list)
    full_dataset = datasets.ImageFolder(root=args.data_dir)
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

    train_dataset = Subset(datasets.ImageFolder(root=args.data_dir, transform=transform), train_indices)
    val_dataset = Subset(datasets.ImageFolder(root=args.data_dir, transform=val_transform), val_indices)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = CNN(
        num_classes=num_classes,
        num_conv_layers=args.num_conv_layers,
        num_filters=args.num_filters,
        kernel_size=args.kernel_size,
        activation_fn=get_activation(args.activation_fn),
        dense_neurons=args.dense_neurons,
        learning_rate=args.learning_rate,
        use_batchnorm=args.use_batchnorm,
        dropout_rate=args.dropout_rate,
        filter_organization=args.filter_organization,
        data_augmentation=args.data_augmentation
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=16,
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)],
        gradient_clip_val=0.5
    )

    trainer.fit(model, train_loader, val_loader)
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/inaturalist_12K/train")
    parser.add_argument("--num_conv_layers", type=int, default=5)
    parser.add_argument("--num_filters", type=int, default=64)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--activation_fn", type=str, default="SiLU", choices=["ReLU", "GELU", "SiLU", "Mish"])
    parser.add_argument("--dense_neurons", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--use_batchnorm", type=bool, default=True)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--filter_organization", type=str, default="same", choices=["same", "double", "half"])
    parser.add_argument("--data_augmentation", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--use_wandb", type=bool, default=False)

    args = parser.parse_args()
    main(args)