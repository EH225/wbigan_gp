"""
This module defines the generator model architecture.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.models.shared_components import SelfAttention


class ResUpBlock(nn.Module):
    """
    Up-sampling convolution block with residual connections.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 32, cond_dim: int = 256):
        """
        Up-sampling convolution block with residual connections.

        :param in_channels: The number of channels of the tensor expected as input.
        :param out_channels: The number of channels to include in the output tensor.
        :param groups: The group size for group norm.
        :param cond_dim: The size of the conditionl input vector i.e. z and the class_embed concatenated.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.cond_dim = cond_dim

        self.resid_skip_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)

        self.gamma1 = nn.Linear(cond_dim, out_channels)
        self.beta1 = nn.Linear(cond_dim, out_channels)

        self.gamma2 = nn.Linear(cond_dim, out_channels)
        self.beta2 = nn.Linear(cond_dim, out_channels)

        self.activation = nn.SiLU()
        # Define a trainable scaling factor for the residual connection to improve stability
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the up-sampling convolution residual block.

        :param x: An input tensor of size (B, in_channels, H, W).
        :param cond_vec: A conditioning vector that is the concatentation of the latent z vector with
            the class embedding. Injecting this conditional information into each residual blocks gives
            better steerability of the outputs.
        :returns: An output tensor of size (B, out_channels, 2*H, 2*W).
        """
        # Use nearest-neighbor interpolation to avoid checkerboard artifacts often
        # produced by transpose convolutions to up-size the inputs by a factor of 2x
        # (B, in_channels, H, W) -> (B, in_channels, 2*H, 2*W)
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        # Upsample the number of channels for the residual connection to match shapes for the output
        x_resid = self.resid_skip_conv(x)  # (B, in_channels, 2*H, 2*W) -> (B, out_channels, 2*H, 2*W)

        # Pass the (B, cond_dim = 2*z_dim) conditioning vector through the linear layers to allow the
        # conditioning vector to modulate every feature channel independently
        gamma1 = self.gamma1(cond_vec).unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)
        beta1 = self.beta1(cond_vec).unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)

        gamma2 = self.gamma2(cond_vec).unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)
        beta2 = self.beta2(cond_vec).unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)

        # Process the input tensor x through the up-sampling block, use a pre-norm layout
        x = self.norm1(x)  # Apply group-norm (B, out_channels, 2*H, 2*W)
        x = gamma1 * x + beta1  # Apply FiLM/AdaIN-style modulation to incorporate conditional info
        x = self.activation(x)  # Apply SiLU (Sigmoid Linear Unit) activation
        x = self.conv1(x)  # (B, in_channels, 2*H, 2*W) -> (B, out_channels, 2*H, 2*W)

        x = self.norm2(x)  # Apply group-norm (B, out_channels, 2*H, 2*W)
        x = gamma2 * x + beta2  # Apply FiLM/AdaIN-style modulation to incorporate conditional info
        x = self.activation(x)  # Apply SiLU (Sigmoid Linear Unit) activation
        x = self.conv2(x)  # (B, out_channels, 2*H, 2*W) -> (B, out_channels, 2*H, 2*W)

        x = x * self.res_scale + x_resid  # Link with x_resid to form a residual connection
        return x


class Generator(nn.Module):
    """
    Conditional image generator model for a Wasserstein Bi-GAN.

    Starting with an input latent noise vector z of size (B, z_dim), this model returns a collection
    of up-sampled RGB images of size (B, 3, 128, 128) of pixel values [-1, +1].

    Generator(z, class_embed) -> fake_images
    """

    def __init__(self, z_dim: int = 128):
        """
        Conditional image generator model for a Wasserstein Bi-GAN.

        :param z_zim: The dimension of the input latent noise vector, z. This is also the dimension of the
            embedding vectors used to represent each class. The default is 128.
        """
        super().__init__()
        self.name = "Generator"
        self.z_dim = z_dim

        # This fully connected layer maps from the latent noise vector to a larger tensor that
        # gets reshaped for convolution operations (z_dim, ) -> (512 * 8 * 8) -> (512, 8, 8)
        self.fc = nn.Linear(z_dim, 512 * 8 * 8)

        # An initial convolution immediately after the linear layer but before the residual
        # blocks to help the network organize the initial learned feature map before upsampling
        self.input_conv = nn.Sequential(
            nn.GroupNorm(32, 512),  # (B, 512, 8, 8) -> (B, 512, 8, 8)
            nn.SiLU(),
            nn.Conv2d(512, 512, 3, padding=1),  # Out: (B, 512, 8, 8)
        )

        # The dimension of the conditional context vector will always be 2x z_dim since it is
        # the concatenation of the z-vector and the class conditional embedding which is also
        # of size z_dim, it will be (B, 2*z_dim)
        cond_dim = z_dim * 2

        # Define the generator backbone as a series of up-sampling residual conv blocks
        self.blocks = nn.ModuleList([
            ResUpBlock(512, 512, cond_dim=cond_dim),  # (B, 512, 8, 8) -> (B, 512, 16, 16)
            ResUpBlock(512, 256, cond_dim=cond_dim),  # (B, 512, 16, 16) -> (B, 256, 32, 32)
            SelfAttention(256),  # (B, 256, 32, 32) -> (B, 256, 32, 32)
            ResUpBlock(256, 128, cond_dim=cond_dim),  # (B, 256, 32, 32) -> (B, 128, 64, 64)
            ResUpBlock(128, 64, cond_dim=cond_dim),  # (B, 128, 64, 64) -> (B, 64, 128, 128)
        ])

        # Final conv mapping from (B, 64, 128, 128) to (B, 3, 128, 128) i.e. 3 channel RGB with
        # final pixel values [-1, +1]
        self.to_rgb = nn.Sequential(
            nn.GroupNorm(32, 64),  # Apply a final group norm
            nn.SiLU(),  # And SiLU (Sigmoid Linear Unit) activation
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Re-shape the output channels down to 3
            nn.Tanh()  # Constrain to [-1, +1] pixel values
        )

    def forward(self, z: torch.Tensor, class_embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator model. Generator(z, class_embed) -> fake_images

        :param z: An input latent tensor of size (B, z_dim) to seed the generation process.
        :param class_embed: An input tensor of class label embeddings of size (B, z_dim).
        :returns: A tensor of size (B, 3, 128, 128) generated images.
        """
        assert z.shape == class_embed.shape, "z.shape must equal class_embed.shape"
        # Concatenate the z-vector with the class embedding vector to create a conditioning vector
        cond_vec = torch.cat([z, class_embed], dim=1)  # (B, 2*z_dim)

        x = self.fc(z)  # (B, z_dim) -> (B, 512 * 8 * 8) = (B, 32768)
        x = x.view(-1, 512, 8, 8)  # Reshape into a 2d image (B, 512 * 8 * 8) -> (B, 512, 8, 8)
        x = self.input_conv(x)  # Apply the initial conv layer

        # Pass through the residual block backbone with self-attention
        for block in self.blocks:
            if isinstance(block, ResUpBlock):
                x = block(x, cond_vec)
            else:  # Multi-headed self-attention doesn't require the input cond_vec
                x = block(x)

        x = self.to_rgb(x)  # Convert final output to 3 channel RGB outputs [-1, +1]

        return x
