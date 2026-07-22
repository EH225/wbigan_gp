"""
This module defines the discriminator (critic) model architecture.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_models.shared_components import SelfAttention
import numpy as np


class ResDownBlock(nn.Module):
    """
    Down-sampling convolution block with residual connections.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 16):
        """
        Down-sampling convolution block with residual connections.

        :param in_channels: The number of channels of the tensor expected as input.
        :param out_channels: The number of channels to include in the output tensor.
        :param groups: The number of groups for group norm.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.resid_skip_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),  # Re-shape channels (B, out_channels, H, W)
            nn.AvgPool2d(2),  # Down-sample by a factor of 2 (B, out_channels, H/2, W/2)
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.activation = nn.LeakyReLU(0.2)

        # Define a trainable scaling factor for the residual connection to improve stability
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the down-sampling convolution residual block.

        :param x: An input tensor of size (B, in_channels, H, W).
        :returns: An output tensor of size (B, out_channels, H/2, W/2).
        """
        # Downsample x for the residual connection to match shapes for the output
        x_resid = self.resid_skip_conv(x)  # (B, in_channels, H, W) -> (B, out_channels, H/2, W/2)

        # Process the input tensor x through the down-sampling block, use a pre-norm layout
        x = self.conv1(x)  # (B, in_channels, H, W) -> (B, out_channels, H/2, W/2)
        x = self.activation(x)  # Apply SiLU (Sigmoid Linear Unit) activation

        x = self.conv2(x)  # (B, out_channels, H/2, W/2) -> (B, out_channels, H/2, W/2)
        x = self.activation(x)  # Apply SiLU (Sigmoid Linear Unit) activation

        x = x * self.res_scale + x_resid  # Link with x_resid to form a residual connection
        return x


class Discriminator(nn.Module):
    """
    Conditional discriminator (critic) model for a Wasserstein Bi-GAN.

    This model operates on:
        - An input image tensor of size (B, 3, image_dim, image_dim)
        - A latent z tensor of size (B, z_dim)
        - A class ID which is internally converted to an embedding tensor of size (B, z_dim)

    and jointly reasons about the image features and the latent vector. It outputs a single real-valued
    score: D(x, z) ∈ R denoting how similar the inputs are to the observed data distribution. During training
    the discriminator tries to maximize: E[D(x_real, E(x_real))] - E[D(G(z), z)]
    with real image examples (x_real, E(x_real)) and fake image examples ((G(z), z)).
    It judges whether an (image, latent) pair comes from the joint data distribution or the joint model
    distribution. The critic never sees mismatched pairs i.e. only (real image, encoded latent z) or
    (generated image, sampled latent z). The encoder and generator are trained together so that these two
    joint distributions become indistinguishable.

    Discriminator(image, z) -> float
    """

    def __init__(self, z_dim: int = 64, image_dim: int = 64, num_classes: int = 0):
        """
        A conditional discriminator (critic) model for a Wasserstein Bi-GAN.

        :param z_dim: The dimension of the output latent noise vector, z. This is also the dimension of the
            embedding vectors used to represent each class. The default is 64.
        :param image_dim: The dimension of the input images used during training and also the output images
            produced by the Bi-GAN model. Must be one of: [128, 64, 32].
        :param num_classes: The number of classes expected i.e. how many unique conditional inputs. Pass in
            zero for no conditional classes.
        """
        super().__init__()
        self.name = "discriminator"
        self.z_dim = z_dim
        assert image_dim in [128, 64, 32], "image_dim must be one of: [128, 64, 32]"
        self.image_dim = image_dim
        self.num_classes = num_classes
        assert isinstance(num_classes, int) and num_classes >= 1, "num_classes must be an int >= 1"

        if num_classes > 1:  # Create a class embedding layer if needed
            self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=z_dim)

        # An initial convolution before the residual down-sampling conv blocks
        # All examples throughout this module will be using the image_dim=64 default
        self.input_conv = nn.Conv2d(3, 64, 3, padding=1)  # Out: (B, 64, 64, 64)

        # Define the discriminator CNN encoder backbone as a series of down-sampling residual conv blocks
        channel_schedule = [(64, 128), (128, 256), (256, 512)]
        channel_schedule += [(512, 512)] * (int(np.log2(self.image_dim)) - 3)  # End with (B, 512, 1, 1)
        # E.g. for image_size = 64
        # (B, 64, 64, 64) -> (B, 128, 32, 32)
        # (B, 128, 32, 32) -> (B, 256, 16, 16)
        # (B, 256, 16, 16) -> (B, 512, 8, 8)
        # (B, 512, 8, 8) -> (B, 512, 4, 4)
        # (B, 512, 4, 4) -> (B, 512, 2, 2)
        # (B, 512, 2, 2) -> (B, 512, 1, 1)
        self.blocks = nn.Sequential(*[ResDownBlock(*channels) for channels in channel_schedule])

        # Define a small MLP to process the input latent z vector
        self.mlp_z = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
        )

        # Define a small MLP to process the class embedding vector
        self.mlp_c = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.z_dim * 2, self.z_dim * 2),
        )

        # Define the final MLP that will operate on the fusion of all 3 inputs (x, z, class_embed)
        self.mlp = nn.Sequential(
            nn.Linear(512 * 3 + (self.z_dim * 2 if self.num_classes > 1 else 0), 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

        # Initialize the final layer's weights at zero
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor, class_id: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the discriminator model. Discriminator(image, z, class_id) -> float

        :param x: An input image tensor of size (B, 3, image_dim, image_dim) with pixel values [-1, +1].
        :param z: An input latent tensor of size (B, z_dim).
        :param class_id: An input tensor of class ID integers of size (B,).
        :returns: A tensor of size (B, 1) containing the Wasserstein scores for each input pair.
        """
        msg = f"x must be (B, 3, {self.image_dim}, {self.image_dim})"
        assert x.shape == (len(x), 3, self.image_dim, self.image_dim), msg
        assert len(x) == len(z), "Inputs x and z must be the same length"
        assert z.shape[-1] == self.z_dim, "z must be (B, z_dim)"
        if self.num_classes > 1:
            assert class_id is not None, "class_id must not be None if num_classes > 1"
            assert len(z) == len(class_id), "Inputs z and class_id must the same length"

        # 1). Process the input images, pass it through the CNN backbone to generate a deep latent rep.
        x = self.input_conv(x)  # Apply an initial conv (B, 3, 64, 64) -> (B, 64, 64, 64)
        x = self.blocks(x)  # Pass through the residual CNN encoder blocks (B, 512, 1, 1)
        # x = F.adaptive_avg_pool2d(x, 1)  # Downsample further (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # Flatten (B, 512, 1, 1) -> (B, 512)

        # 2). Apply an MLP processing step to the input z-vector
        z = self.mlp_z(z)  # (B, z_dim) -> (B, 512)

        # 3). Apply an MLP processing step to the input class embedding if num_classes > 0 and feature
        # fusion by concatenating all inputs together and then passing to an MLP
        # With class conditioning: (B, 512) + (B, 512) + (B, 2*z_dim) = (B, 1024 + 2*z_dim)
        # Otherwise: (B, 512) + (B, 512 + 512) = (B, 1024)
        if self.num_classes > 1:
            class_embed = self.mlp_c(self.class_embedding(class_id))  # (B,) -> (B, z_dim) -> (B, 2*z_dim)
            x = torch.cat([x, z, x * z, class_embed], dim=1)  # (B, 512 * 3 + 2*z_dim)
        else:
            x = torch.cat([x, z, x * z], dim=1)  # (B, 512 * 3)
        return self.mlp(x)  # (B, 1)
