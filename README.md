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

### Training

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
  
### full commands
--data_dir             Path to dataset (ImageFolder format)
--num_conv_layers      Number of convolutional layers [default: 5]
--num_filters          Number of filters in first conv layer [default: 64]
--kernel_size          Convolution kernel size [default: 3]
--activation_fn        Activation function: ReLU, GELU, SiLU, Mish [default: SiLU]
--dense_neurons        Neurons in fully connected layer [default: 256]
--learning_rate        Learning rate [default: 0.001]
--use_batchnorm        Enable batch normalization [default: True]
--dropout_rate         Dropout rate [default: 0.3]
--filter_organization  Filter scaling: same, double, half [default: same]
--data_augmentation    Apply data augmentation [default: True]
--batch_size           Batch size [default: 64]
--epochs               Number of training epochs [default: 15]
--use_wandb            Enable W&B logging [default: False]
