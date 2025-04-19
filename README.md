# Assignment2
# Part A
## Overview

This repository contains a configurable CNN image classification pipeline built using **PyTorch Lightning**. It supports dynamic model architecture, data augmentation, and integrates with **Weights & Biases (W&B)** for experiment tracking.

## Features

- Modular CNN with customizable layers and filters
- Supports activation functions: `ReLU`, `GELU`, `SiLU`, `Mish`
- Batch normalization and dropout options
- Filter scaling: `same`, `double`, `half`
- 16-bit mixed precision training
- GPU acceleration
- Optional W&B logging

## Getting Started

### Install Dependencies
pip install torch torchvision pytorch-lightning wandb
### Dataset Format
Place your dataset in the following structure (uses torchvision.datasets.ImageFolder):
<pre>
dataset/
├── class_1/
│   ├── image1.jpg
│   └── image2.jpg
├── class_2/
│   ├── image3.jpg
│   └── image4.jpg
</pre>

## Running the Code

### Basic Training
python train_cnn.py --data_dir /path/to/dataset

python train_cnn.py \
  --data_dir ./dataset \
  --num_conv_layers 4 \
  --num_filters 32 \
  --kernel_size 5 \
  --activation_fn GELU \
  --dropout_rate 0.2 \
  --filter_organization double \
  --epochs 20 \
  --use_batchnorm True \
  --data_augmentation True \
  --use_wandb True
