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

    def __init__(self, in_channels: int, out_channels: int, groups: int = 16, cond_dim: int = 128):
        """
        Down-sampling convolution block with residual connections.

        :param in_channels: The number of channels of the tensor expected as input.
        :param out_channels: The number of channels to include in the output tensor.
        :param groups: The number of groups for group norm.
        :param cond_dim: The size of the condition input vector i.e. the class_embed vector.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.cond_dim = cond_dim

        self.resid_skip_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),  # Re-shape channels (B, out_channels, H, W)
            nn.AvgPool2d(2),  # Down-sample by a factor of 2 (B, out_channels, H/2, W/2)
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)

        if self.cond_dim > 0:  # Init only if cond_dim > 0 i.e. there is conditioning to be applied
            self.gamma1 = nn.Linear(cond_dim, in_channels)
            self.beta1 = nn.Linear(cond_dim, in_channels)

            self.gamma2 = nn.Linear(cond_dim, out_channels)
            self.beta2 = nn.Linear(cond_dim, out_channels)

            # Initialize FiLM layer parameters at mu=0 and sigma^2=1 for each
            nn.init.zeros_(self.gamma1.weight)
            nn.init.ones_(self.gamma1.bias)

            nn.init.zeros_(self.beta1.weight)
            nn.init.zeros_(self.beta1.bias)

            nn.init.zeros_(self.gamma2.weight)
            nn.init.ones_(self.gamma2.bias)

            nn.init.zeros_(self.beta2.weight)
            nn.init.zeros_(self.beta2.bias)

        self.activation = nn.SiLU()

        # Define a trainable scaling factor for the residual connection to improve stability
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the down-sampling convolution residual block.

        :param x: An input tensor of size (B, in_channels, H, W).
        :param cond_vec: A conditioning vector i.e. the class embedding vector that injects class conditional
            information at each down-sampling residual block layer.
        :returns: An output tensor of size (B, out_channels, H/2, W/2).
        """
        if self.cond_dim > 0:
            assert cond_vec is not None, "cond_vec cannot be None if self.cond_dim > 0"
        else:
            assert cond_vec is None, "cond_vec must be None if self.cond_dim == 0"
        # Downsample x for the residual connection to match shapes for the output
        x_resid = self.resid_skip_conv(x)  # (B, in_channels, H, W) -> (B, out_channels, H/2, W/2)

        # Process the input tensor x through the down-sampling block, use a pre-norm layout
        x = self.norm1(x)  # Apply group-norm (B, in_channels, H, W)
        if self.cond_dim > 0:
            # Pass the (B, cond_dim = z_dim) conditioning vector through the linear layers to allow the
            # conditioning vector to modulate every feature channel independently
            gamma1 = self.gamma1(cond_vec).unsqueeze(-1).unsqueeze(-1)  # (B, in_channels, 1, 1)
            beta1 = self.beta1(cond_vec).unsqueeze(-1).unsqueeze(-1)  # (B, in_channels, 1, 1)
            x = gamma1 * x + beta1  # Apply FiLM/AdaIN-style modulation to incorporate conditional info
        x = self.activation(x)  # Apply SiLU (Sigmoid Linear Unit) activation
        x = self.conv1(x)  # (B, in_channels, H, W) -> (B, out_channels, H/2, W/2)

        x = self.norm2(x)  # Apply group-norm (B, out_channels, H/2, W/2)
        if self.cond_dim > 0:
            # Pass the (B, cond_dim = z_dim) conditioning vector through the linear layers to allow the
            # conditioning vector to modulate every feature channel independently
            gamma2 = self.gamma2(cond_vec).unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)
            beta2 = self.beta2(cond_vec).unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)
            x = gamma2 * x + beta2  # Apply FiLM/AdaIN-style modulation to incorporate conditional info
        x = self.activation(x)  # Apply SiLU (Sigmoid Linear Unit) activation
        x = self.conv2(x)  # (B, out_channels, H/2, W/2) -> (B, out_channels, H/2, W/2)

        x = x * self.res_scale + x_resid  # Link with x_resid to form a residual connection
        return x


class Encoder(nn.Module):
    """
    Conditional image encoder model for a Wasserstein Bi-GAN.

    Starting with an input image (B, 3, 64, 64) of pixel values [-1, +1] and a conditional class id
    vector of size (B, ), this model returns predicted latent z-vectors of size (B, z_dim).

    This model approximates an inverse mapping of the generator.

    Encoder(real_image, class_id) -> z_hat
    """

    def __init__(self, z_dim: int = 64, image_dim: int = 64, num_classes: int = 1):
        """
        Conditional image encoder model for a Wasserstein Bi-GAN.

        :param z_dim: The dimension of the output latent noise vector, z. This is also the dimension of the
            embedding vectors used to represent each class. The default is 64.
        :param image_dim: The dimension of the input images used during training and also the output images
            produced by the Bi-GAN model. Must be one of: [128, 64, 32].
        :param num_classes: The number of classes expected i.e. how many unique conditional inputs. Pass in
            zero for no conditional classes.
        """
        super().__init__()
        self.name = "encoder"
        self.z_dim = z_dim
        assert image_dim in [128, 64, 32], "image_dim must be one of: [128, 64, 32]"
        self.image_dim = image_dim
        self.num_classes = num_classes
        assert isinstance(num_classes, int) and num_classes >= 1, "num_classes must be an int >= 1"
        cond_dim = self.z_dim if num_classes > 1 else 0

        if num_classes > 1:  # Create a class embedding layer if needed
            self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=z_dim)

        # An initial convolution before the residual down-sampling conv blocks
        # All examples throughout this module will be using the image_dim=64 default
        self.input_conv = nn.Conv2d(3, 64, 3, padding=1)  # Out: (B, 64, 64, 64)

        # Define the encoder backbone as a series of down-sampling residual conv blocks
        self.blocks = nn.ModuleList([
            ResDownBlock(64, 128, cond_dim=cond_dim),  # (B, 64, 64, 64) -> (B, 128, 32, 32)
            ResDownBlock(128, 256, cond_dim=cond_dim),  # (B, 128, 32, 32) -> (B, 256, 16, 16)
            SelfAttention(256),  # (B, 256, 16, 16) -> (B, 256, 16, 16)
            ResDownBlock(256, 512, cond_dim=cond_dim),  # (B, 256, 16, 16) -> (B, 512, 8, 8)
            ResDownBlock(512, 512, cond_dim=cond_dim),  # (B, 512, 8, 8) -> (B, 512, 4, 4)
        ])

        # Add a final convolution to mix spatial information before compressing
        self.final_conv = nn.Sequential(
            nn.GroupNorm(16, 512),
            nn.SiLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
        )  # (B, 512, 4, 4) -> (B, 512, 4, 4)

        # This fully connected layer takes the deep latent feature representation of the input
        # image concatenated with the class_embed vector and outputs a predicted latent z-vector
        self.norm = nn.LayerNorm(512)
        # (B, 512 + z_dim) -> (B, z_dim) if classes are provided, else (B, 512) -> (B, z_dim)
        self.fc = nn.Linear(512 + (z_dim if self.num_classes > 1 else 0), z_dim)

    def forward(self, x: torch.Tensor, class_id: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the encoder model. Encoder(real_image, class_id) -> z_hat

        :param x: An input image tensor of size (B, 3, image_dim, image_dim) with pixel values [-1, +1].
        :param class_id: An input tensor of class ID integers of size (B,).
        :returns: A tensor of size (B, z_dim) estimated latent z-vectors.
        """
        msg = f"x must be (B, 3, {self.image_dim}, {self.image_dim})"
        assert x.shape == (len(x), 3, self.image_dim, self.image_dim), msg
        if self.num_classes > 1:
            assert class_id is not None, "class_id must not be None if num_classes > 0"
            assert len(x) == len(class_id), "Inputs x and class_id must the same length"

        class_embed = self.class_embedding(class_id) if self.num_classes > 1 else None

        x = self.input_conv(x)  # Apply an initial conv (B, 3, 64, 64) -> (B, 64, 64, 64)
        # Pass through the residual CNN encoder blocks (B, 512, 4, 4) with self-attention
        for block in self.blocks:
            x = block(x, class_embed)

        x = self.final_conv(x)  # Pass through a final conv to max information along the spatial dimension
        x = F.adaptive_avg_pool2d(x, 1)  # Downsample further (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # Flatten (B, 512, 1, 1) -> (B, 512)
        z_hat = self.fc(self.norm(x))  # Compute a predicted latent representation (B, z_dim)
        return z_hat
