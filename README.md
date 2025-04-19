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

python-repl
Copy
Edit
data_dir/
├── class_1/
│   ├── img1.jpg
│   ├── img2.jpg
├── class_2/
│   ├── img3.jpg
│   ├── img4.jpg
...
