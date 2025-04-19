# Assignment2
wandb link: https://wandb.ai/sai-sakunthala-indian-institute-of-technology-madras/cnn-sweep/reports/Assignment-2-report--VmlldzoxMjM2MDMxOQ

github link: https://github.com/Sai-sakunthala/Assignment2

This repository 6 files, 3 .py files and 3 .ipynb files. Below is explanation of .py files, the .ipynb files are the same version of this with wandb sweeps included and ready to run versions on colab notebook
# Part A : from scratch implementation
## 1) assignment_2_parta_train.py 
## Overview

This file contains a configurable CNN image classification pipeline built using **PyTorch Lightning** for inaturalist_data. It supports dynamic model architecture, data augmentation, and integrates with **Weights & Biases (W&B)** for experiment tracking.

## Features

- Modular CNN with customizable layers and filters
- Supports activation functions: `ReLU`, `GELU`, `SiLU`, `Mish`
- Batch normalization and dropout options
- Filter scaling: `same`, `double`, `half`
- 16-bit mixed precision training
- GPU acceleration
- Optional W&B logging

# CNN module
The CNN class is a convolutional neural network built using PyTorch Lightning. Here’s an overview of its components:

Convolutional Layers: The model dynamically adds multiple convolution layers based on num_conv_layers. Each layer is followed by batch normalization (optional), an activation function (default SiLU), dropout, and max pooling. The number of filters is adjustable with options to either double or halve filters between layers.

Dense Layers: After the convolution block, the output is flattened and passed through a fully connected layer (fc1) with batch normalization and activation. A dropout layer is applied if needed.

Output Layer: The final layer (fc2) outputs class predictions for classification.

Optimizer and Scheduler: The optimizer is Adam, and a cosine annealing learning rate scheduler is used.

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

| Argument               | Description                                           | Default       |
|------------------------|-------------------------------------------------------|---------------|
| `--data_dir`           | Path to dataset (ImageFolder format)                 | `/root/inaturalist_12K/train` |
| `--num_conv_layers`    | Number of convolutional layers                       | `5`           |
| `--num_filters`        | Number of filters in first conv layer                | `64`          |
| `--kernel_size`        | Convolution kernel size                              | `3`           |
| `--activation_fn`      | Activation function: `ReLU`, `GELU`, `SiLU`, `Mish`  | `SiLU`        |
| `--dense_neurons`      | Neurons in fully connected layer                     | `256`         |
| `--learning_rate`      | Learning rate                                        | `0.001`       |
| `--use_batchnorm`      | Enable batch normalization                           | `True`        |
| `--dropout_rate`       | Dropout rate                                         | `0.3`         |
| `--filter_organization`| Filter scaling: `same`, `double`, `half`            | `same`        |
| `--data_augmentation`  | Apply data augmentation                              | `True`        |
| `--batch_size`         | Batch size                                           | `64`          |
| `--epochs`             | Number of training epochs                            | `15`          |
| `--use_wandb`          | Enable W&B logging                                   | `False`       |

## 2) test_data_partA.py

This script evaluates a pre-trained CNN model on a test dataset of inaturalist using PyTorch Lightning. It supports optional integration with **Weights and Biases (W&B)** for logging and visualization.
## W&B Integration
When enabled, logs include:

Test Loss

Test Accuracy

Class-wise Accuracy

Sample Predictions (True vs Predicted labels)
## sample run
python test.py --test_dir /path/to/test_data --model_checkpoint /path/to/model.ckpt --batch_size 32 --use_wandb --wandb_project cnn-sweep --wandb_entity your_wandb_username --run_name test_run_1
## Arguments

| Argument              | Description                                                                | Default Value    |
|-----------------------|----------------------------------------------------------------------------|------------------|
| `--test_dir`           | Path to the test dataset directory (ImageFolder format)                    | N/A              |
| `--model_checkpoint`   | Path to model checkpoint (.ckpt)                                           | N/A              |
| `--batch_size`         | Batch size for testing                                                      | 64               |
| `--use_wandb`          | Enable W&B logging                                                          | False            |
| `--wandb_project`      | W&B project name                                                           | cnn-sweep        |
| `--wandb_entity`       | W&B username/entity (required if using WandB)                              | N/A              |
| `--wandb_sweep_id`     | Sweep ID to pull the best model                                            | None             |
| `--run_name`           | W&B run name                                                              | test_run         |
| `--visualize_samples`  | Visualize predictions using W&B                                            | False            |

# Part B: Fine-Tuning EfficientNetV2
## assignment_2_partb_train.py
This repository contains code for fine-tuning the EfficientNetV2 model using PyTorch Lightning and Weights and Biases for the iNaturalist dataset.

## FineTunedModel module
#### Model Initialization:
- **EfficientNet-V2-M** is loaded with pre-trained **ImageNet** weights.
- Blocks of the model are initially frozen based on `freeze_k`.
- The final classifier is replaced with a **Dropout** layer and a **Linear** layer, tailored for `num_classes`.

#### Training Strategy:
- Freezes the first `freeze_k` layers.
- Gradually unfreezes layers every `unfreeze_every` epochs to adapt progressively.

#### Optimization:
- Uses **Adam** optimizer with **Cosine Annealing** LR scheduler for better convergence.

#### Training & Validation:
- Uses standard **Cross-Entropy Loss** and logs **accuracy** metrics for both training and validation.

#### Dynamic Layer Freezing:
- Gradual unfreezing of layers reduces overfitting, enabling more refined feature learning.
  
This model provides a **progressive fine-tuning** approach, ideal for **transfer learning** scenarios.
## Weights and Biases Integration
WandB is used for logging training progress, loss, accuracy, and model checkpoints.

## Checkpoints
The best model will be saved based on the highest validation accuracy.
## sample code
python train.py --data_dir /path/to/inaturalist/train --batch_size 64 --max_epochs 25 --run_name "efficient_net_finetune_run"

## arguments
| Argument         | Description                                                      | Default                              |
|------------------|------------------------------------------------------------------|--------------------------------------|
| `--data_dir`     | Path to the training dataset (ImageFolder format)                | `/root/inaturalist_12K/train`       |
| `--batch_size`   | Batch size for training/validation                               | 64                                   |
| `--max_epochs`   | Number of training epochs                                        | 25                                   |
| `--run_name`     | The name of the run for WandB logging                            | `efficient_net_4`                   |

