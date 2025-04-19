import os
import argparse
import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from collections import defaultdict
import cv2

class CNN(pl.LightningModule):
    def __init__(self, initial_in_channels=3, num_classes=10, num_conv_layers=5, num_filters=64, kernel_size=3, activation_fn=nn.SiLU,
                 dense_neurons=256, learning_rate=1e-3, use_batchnorm=True, dropout_rate=0.3, filter_organization='same', data_augmentation = True):

        super().__init__()
        self.save_hyperparameters()
        layers_conv = []
        input_channels = initial_in_channels
        current_filters = num_filters
        for i in range(num_conv_layers):
            out_channels = current_filters
            layers_conv.append(nn.Conv2d(input_channels, out_channels, kernel_size = kernel_size, padding = kernel_size//2))
            if use_batchnorm:
                layers_conv.append(nn.BatchNorm2d(out_channels))
            layers_conv.append(activation_fn())
            if dropout_rate == 0:
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
        self.dropout_fc1 = nn.Dropout(dropout_rate) if dropout_rate == 0 else None
        self.fc2 = nn.Linear(dense_neurons, num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.hparams.use_batchnorm:
            x = self.bn_fc1(x)
        x = self.activation_dense(x)
        if self.hparams.dropout_rate == 0:
            x = self.dropout_fc1(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay = 5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        return [optimizer], [scheduler]

def evaluate_model(args):
    if args.use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            job_type="evaluation"
        )

    if args.use_wandb and args.wandb_sweep_id:
        try:
            api = wandb.Api()
            sweep_path = f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_sweep_id}"
            sweep = api.sweep(sweep_path)
            best_run = max(sweep.runs, key=lambda r: r.summary.get('val_acc', 0))
            artifact = best_run.logged_artifacts()[0]
            artifact_dir = artifact.download()
            ckpt_path = os.path.join(artifact_dir, "model.ckpt")
        except Exception as e:
            raise ValueError(f"Failed to load model from wandb: {str(e)}")
    else:
        if not os.path.exists(args.model_checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.model_checkpoint}")
        ckpt_path = args.model_checkpoint

    model = CNN.load_from_checkpoint(ckpt_path)
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(args.test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=2)
    class_names = test_dataset.classes

    trainer = pl.Trainer(accelerator='auto', logger=False)
    test_results = trainer.test(model, dataloaders=test_loader)

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            preds = outputs.argmax(dim=1)

            for label, pred in zip(y, preds):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1

    class_wise_accuracy = {
        class_names[i]: 100 * class_correct[i] / class_total[i]
        if class_total[i] > 0 else 0.0
        for i in range(len(class_names))
    }

    if args.use_wandb:
        wandb.log({"test_loss": test_results[0]['test_loss'],
                  "test_acc": test_results[0]['test_acc']})

        wandb.log({"class_wise_accuracy": class_wise_accuracy})

        table = wandb.Table(columns=["Class", "Accuracy"])
        for class_name, acc in class_wise_accuracy.items():
            table.add_data(class_name, acc)
        wandb.log({"class_accuracy_table": table})

    if args.visualize_samples and args.use_wandb:
        visualize_predictions(model, test_dataset, class_names, project=args.wandb_project)

    if args.use_wandb:
        wandb.finish()

def visualize_predictions(model, dataset, class_names, project="cnn-sweep"):
    wandb.init(project=project, name="sample_predictions", job_type="visualization")

    samples_per_class = {i: [] for i in range(len(class_names))}

    def add_border(img, correct):
        img = (img * 255).astype(np.uint8)
        color = (0, 255, 0) if correct else (255, 0, 0)
        return cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=color)

    with torch.no_grad():
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            input_tensor = img.unsqueeze(0)
            pred = model(input_tensor).argmax(dim=1).item()

            img_disp = img.permute(1, 2, 0).cpu().numpy()
            img_disp = img_disp * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            img_disp = np.clip(img_disp, 0, 1)
            bordered = add_border(img_disp, pred == label)
            caption = f"True: {class_names[label]} | Pred: {class_names[pred]}"
            wandb_img = wandb.Image(bordered, caption=caption)

            if len(samples_per_class[label]) < 3:
                samples_per_class[label].append(wandb_img)

            if all(len(imgs) >= 3 for imgs in samples_per_class.values()):
                break

    all_imgs = [img for imgs in samples_per_class.values() for img in imgs]
    wandb.log({"sample_predictions": all_imgs})
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN model on test data")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test dataset directory")
    parser.add_argument("--model_checkpoint", type=str, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for testing")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="cnn-sweep", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, help="W&B username/entity")
    parser.add_argument("--wandb_sweep_id", type=str, help="Sweep ID to pull best model")
    parser.add_argument("--run_name", type=str, default="test_run", help="W&B run name")
    parser.add_argument("--visualize_samples", action="store_true", help="Visualize predictions")

    args = parser.parse_args()
    
    if args.use_wandb and not args.wandb_entity:
        raise ValueError("wandb_entity must be specified when use_wandb is True")
    
    evaluate_model(args)