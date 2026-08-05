# Conditional Wasserstein BiGAN for Image Generation, Latent Interpolation, Similarity Search, and Anomaly Detection

This repository contains code for constructing and training a conditional Wasserstein BiGAN on the CelebA
dataset. Below is a breif summary of the repo layout:

- `config`: This folder contains config files that specify model architecture and training hyperparameters.
- `torch_models`: This folder contains the definitions of the 3 BiGAN models: the encoder, the generator and
  the Wasserstein critic.
- `dataset_utils.py`: This module contains utility functions for pre-processing and constructing dataloaders.
- `utils.py`: This module contains general utility functions used throughout training and analysis.
- `trainer.py`: This module contains the Trainer class which performs all training operations and
  checkpointing.
- `runner.py`: This module provides easy-to-use functions for running pre-training, training, and
  post-training.
- `environment.yml` - This file outlines the requirements of the conda env used to run the experiments of this
  project.

## Abstract

This project presents the implementation and training of a conditional Wasserstein Bidirectional Generative
Adversarial Network (BiGAN) consisting of a generator, encoder, and Wasserstein critic, trained on
approximately 200,000 RGB headshot images (64 × 64 × 3) from the CelebA dataset. Unlike conventional GANs,
which are often difficult to train due to instability and uninformative gradients, the Wasserstein critic
approximates the Earth Mover (Wasserstein-1) distance and incorporates a gradient penalty to enforce the
1-Lipschitz constraint. This formulation provides more stable and informative training signals for the
generator and encoder, resulting in improved convergence and training stability. By learning a bidirectional
mapping between images and latent representations, the trained BiGAN supports a variety of applications,
including realistic face synthesis, image compression and reconstruction, smooth latent-space interpolation,
image similarity search, and anomaly detection.