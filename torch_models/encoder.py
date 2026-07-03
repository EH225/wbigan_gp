"""
This module defines the encoder model architecture, which approximates an inverse mapping of the generator.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_models.shared_components import SelfAttention


class ResDownBlock(nn.Module):
    """
    Down-sampling convolution block with residual connections.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 32):
        """
        Down-sampling convolution block with residual connections.

        :param in_channels: The number of channels of the tensor expected as input.
        :param out_channels: The number of channels to include in the output tensor.
        :param groups: The group size for group norm.
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

        self.norm1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)

        self.activation = nn.SiLU()
        # Define a trainable scaling factor for the residual connection to improve stability
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the down-sampling convolution residual block.

        :param x: An input tensor of size (B, in_channels, H, W).
        :returns: An output tensor of size (B, out_channels, H/2, W/2).
        """
        # Downsample x for the residual connection to match shapes for the output
        x_resid = self.resid_skip_conv(x)  # (B, in_channels, H, W) -> (B, out_channels, H/2, W/2)

        # Process the input tensor x through the down-sampling block, use a pre-norm layout
        x = self.norm1(x)  # Apply group-norm (B, in_channels, H, W)
        x = self.activation(x)  # Apply SiLU (Sigmoid Linear Unit) activation
        x = self.conv1(x)  # (B, in_channels, H, W) -> (B, out_channels, H/2, W/2)

        x = self.norm2(x)  # Apply group-norm (B, out_channels, H/2, W/2)
        x = self.activation(x)  # Apply SiLU (Sigmoid Linear Unit) activation
        x = self.conv2(x)  # (B, out_channels, H/2, W/2) -> (B, out_channels, H/2, W/2)

        x = x * self.res_scale + x_resid  # Link with x_resid to form a residual connection
        return x


class Encoder(nn.Module):
    """
    Conditional image encoder model for a Wasserstein Bi-GAN.

    Starting with an input image (B, 3, 128, 128) of pixel values [-1, +1] and a conditional class embedding
    vector of size (B, z_dim), this model model returns predicted latent z-vectors of size (B, z_dim).

    This model approximates an inverse mapping of the generator.

    Encoder(real_image, class_embed) -> z_hat
    """

    def __init__(self, z_dim: int = 128):
        """
        Conditional image encoder model for a Wasserstein Bi-GAN.

        :param z_zim: The dimension of the output latent noise vector, z. This is also the dimension of the
            embedding vectors used to represent each class. The default is 128.
        """
        super().__init__()
        self.name = "encoder"
        self.z_dim = z_dim

        # An initial convolution before the residual down-sampling conv blocks
        self.input_conv = nn.Conv2d(3, 64, 3, padding=1)  # Out: (B, 64, 128, 128)

        # Define the encoder backbone as a series of down-sampling residual conv blocks
        self.blocks = nn.Sequential(
            ResDownBlock(64, 128),  # (B, 64, 128, 128) -> (B, 128, 64, 64)
            ResDownBlock(128, 256),  # (B, 128, 64, 64) -> (B, 256, 32, 32)
            SelfAttention(256),  # (B, 256, 32, 32) -> (B, 256, 32, 32)
            ResDownBlock(256, 512),  # (B, 256, 32, 32) -> (B, 512, 16, 16)
            ResDownBlock(512, 512),  # (B, 512, 16, 16) -> (B, 512, 8, 8)
            ResDownBlock(512, 512),  # (B, 512, 8, 8) -> (B, 512, 4, 4)
        )

        # This fully connected layer takes the deep latent feature representation of the input
        # image concatenated with the class_embed vector and outputs a predicted latent z-vector
        self.norm = nn.LayerNorm(512)
        self.fc = nn.Linear(512 + z_dim, z_dim)  # (B, 512 + z_dim) -> (B, z_dim)

    def forward(self, x: torch.Tensor, class_embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder model. Encoder(real_image, class_embed) -> z_hat

        :param x: An input image tensor of size (B, 3, 128, 128) with pixel values [-1, +1].
        :param class_embed: An input tensor of class label embeddings of size (B, z_dim).
        :returns: A tensor of size (B, z_dim) estimated latent z-vectors.
        """
        assert class_embed.shape[-1] == self.z_dim, "class_embed have a size of (B, z_dim)"
        x = self.input_conv(x)  # Apply an initial conv (B, 3, 128, 128) -> (B, 64, 128, 128)
        x = self.blocks(x)  # Pass through the residual CNN encoder blocks (B, 512, 4, 4)
        x = F.adaptive_avg_pool2d(x, 1)  # Downsample further (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # Flatten (B, 512, 1, 1) -> (B, 512)
        x = F.silu(self.norm(x))  # Norm and pass through a non-linearity before the last FCNN layer
        # Add in the class conditional information by concatenating class_embed
        x = torch.cat([x, class_embed], dim=1)  # (B, 512) + (B, z_dim) = (B, 512 + z_dim)
        z_hat = self.fc(x)  # Compute a predicted latent representation (B, z_dim)
        return z_hat
