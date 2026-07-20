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
from torch_models.shared_components import SelfAttention


class ResUpBlock(nn.Module):
    """
    Up-sampling convolution block with residual connections.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 16, cond_dim: int = 256):
        """
        Up-sampling convolution block with residual connections.

        :param in_channels: The number of channels of the tensor expected as input.
        :param out_channels: The number of channels to include in the output tensor.
        :param groups: The number of groups for group norm.
        :param cond_dim: The size of the condition input vector i.e. z and the class_embed concatenated if
            num_classes > 1, else just the latent z vector.
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

        # Allow each block to transform the input cond_vec separately as is done in BigGAN
        self.mlp_cond_vec = nn.Sequential(
            nn.Linear(cond_dim, 2 * cond_dim),
            nn.SiLU(),
            nn.Linear(2 * cond_dim, cond_dim),
        )

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

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the up-sampling convolution residual block.

        :param x: An input tensor of size (B, in_channels, H, W).
        :param cond_vec: A conditioning vector that is the concatenation of the latent z vector with
            the class embedding. Injecting this conditional information into each residual blocks gives
            better steerability of the outputs.
        :returns: An output tensor of size (B, out_channels, 2*H, 2*W).
        """
        # Use nearest neighbor interpolation to avoid checkerboard artifacts often produced by transpose
        # convolutions to up-size the inputs by a factor of 2x
        # (B, in_channels, H, W) -> (B, in_channels, 2*H, 2*W)
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        # Upsample the number of channels for the residual connection to match shapes for the output
        x_resid = self.resid_skip_conv(x)  # (B, in_channels, 2*H, 2*W) -> (B, out_channels, 2*H, 2*W)

        cond_vec = self.mlp_cond_vec(cond_vec)  # (B, cond_dim) -> (B, cond_dim)

        # Process the input tensor x through the up-sampling block, use a pre-norm layout
        x = self.norm1(x)  # Apply group-norm (B, out_channels, 2*H, 2*W)
        # Pass the (B, cond_dim) conditioning vector through the linear layers to allow the
        # conditioning vector to modulate every feature channel independently
        gamma1 = self.gamma1(cond_vec).unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)
        beta1 = self.beta1(cond_vec).unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)
        x = gamma1 * x + beta1  # Apply FiLM/AdaIN-style modulation to incorporate conditional info
        x = self.activation(x)  # Apply SiLU (Sigmoid Linear Unit) activation
        x = self.conv1(x)  # (B, in_channels, 2*H, 2*W) -> (B, out_channels, 2*H, 2*W)

        x = self.norm2(x)  # Apply group-norm (B, out_channels, 2*H, 2*W)
        # Pass the (B, cond_dim) conditioning vector through the linear layers to allow the
        # conditioning vector to modulate every feature channel independently
        gamma2 = self.gamma2(cond_vec).unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)
        beta2 = self.beta2(cond_vec).unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)
        x = gamma2 * x + beta2  # Apply FiLM/AdaIN-style modulation to incorporate conditional info
        x = self.activation(x)  # Apply SiLU (Sigmoid Linear Unit) activation
        x = self.conv2(x)  # (B, out_channels, 2*H, 2*W) -> (B, out_channels, 2*H, 2*W)

        x = x * self.res_scale + x_resid  # Link with x_resid to form a residual connection
        return x


class Generator(nn.Module):
    """
    Conditional image generator model for a Wasserstein Bi-GAN.

    Starting with an input latent noise vector z of size (B, z_dim), this model returns a collection
    of up-sampled RGB images of size (B, 3, image_dim, image_dim) of pixel values [-1, +1].

    Generator(z, class_id) -> fake_images
    """

    def __init__(self, z_dim: int = 64, image_dim: int = 64, num_classes: int = 1):
        """
        Conditional image generator model for a Wasserstein Bi-GAN.

        :param z_dim: The dimension of the input latent noise vector, z. This is also the dimension of the
            embedding vectors used to represent each class. The default is 64.
        :param image_dim: The dimension of the input images used during training and also the output images
            produced by the Bi-GAN model. Must be one of: [128, 64, 32].
        :param num_classes: The number of classes expected i.e. how many unique conditional inputs. Pass in
            zero for no conditional classes.
        """
        super().__init__()
        self.name = "generator"
        self.z_dim = z_dim
        assert image_dim in [128, 64, 32], "image_dim must be one of: [128, 64, 32]"
        self.image_dim = image_dim
        self.num_classes = num_classes
        assert isinstance(num_classes, int) and num_classes >= 1, "num_classes must be an int >= 1"

        if num_classes > 1:  # Create a class embedding layer if there are multiple classes
            self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=z_dim)

        # The dimension of the conditional context vector will be either z_dim or 2 * z_dim if there are
        # conditional class embedding as well
        cond_dim = z_dim * (2 if self.num_classes > 1 else 1)

        # This fully connected layer maps from the conditioned latent noise vector, to a larger tensor that
        # gets reshaped for convolution operations (z_dim, ) -> e.g. 512 * 8 * 8) -> (512, 8, 8)
        # for image_dim = 128, otherwise (512, 4, 4) for image_dim = 64 and (512, 2, 2) for image_dim = 32
        # All examples throughout this module will be using the image_dim=64 default
        self.fc = nn.Linear(cond_dim, 512 * (self.image_dim // 16) ** 2)

        # An initial convolution immediately after the linear layer but before the residual
        # blocks to help the network organize the initial learned feature map before upsampling
        self.input_conv = nn.Sequential(
            nn.GroupNorm(16, 512),  # (B, 512, 4, 4) -> (B, 512, 4, 4)
            nn.SiLU(),
            nn.Conv2d(512, 512, 3, padding=1),  # Out: (B, 512, 4, 4)
        )

        # Define the generator backbone as a series of up-sampling residual conv blocks
        self.blocks = nn.ModuleList([
            ResUpBlock(512, 512, cond_dim=cond_dim),  # (B, 512, 4, 4) -> (B, 512, 8, 8)
            ResUpBlock(512, 256, cond_dim=cond_dim),  # (B, 512, 8, 8) -> (B, 256, 16, 16)
            SelfAttention(256),  # (B, 256, 16, 16) -> (B, 256, 16, 16)
            ResUpBlock(256, 128, cond_dim=cond_dim),  # (B, 256, 16, 16) -> (B, 128, 32, 32)
            ResUpBlock(128, 64, cond_dim=cond_dim),  # (B, 128, 32, 32) -> (B, 64, 64, 64)
        ])

        # Final conv mapping from (B, 64, 64, 64) to (B, 3, 64, 64) i.e. 3 channel RGB with
        # final pixel values [-1, +1]
        self.to_rgb = nn.Sequential(
            nn.SiLU(),  # And SiLU (Sigmoid Linear Unit) activation
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Re-shape the output channels down to 3
            nn.Tanh()  # Constrain to [-1, +1] pixel values
        )

    def forward(self, z: torch.Tensor, class_id: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the generator model. Generator(z, class_embed) -> fake_images

        :param z: An input latent tensor of size (B, z_dim) to seed the generation process.
        :param class_id: An input tensor of class ID integers of size (B,).
        :returns: A tensor of size (B, 3, image_dim, image_dim) generated images.
        """
        assert z.shape[-1] == self.z_dim, "z must be (B, z_dim)"
        if self.num_classes > 1:
            assert class_id is not None, "class_id must not be None if num_classes > 1"
            assert len(z) == len(class_id), "Inputs z and class_id must the same length"
            # Concatenate the z-vector with the class embedding vector to create a conditioning vector
            class_embed = self.class_embedding(class_id)  # (B,) -> (B, z_dim)
            cond_vec = torch.cat([z, class_embed], dim=1)  # (B, 2*z_dim)
        else:  # Otherwise, no concatenation needed, cond_vec is just z (B, z_dim)
            cond_vec = z

        x = self.fc(cond_vec)  # (B, cond_dim) -> (B, 512 * 4 * 4) = (B, 8192)
        # Reshape into a 2d image (B, 512 * 4 * 4) -> (B, 512, 4, 4)
        x = x.view(-1, 512, (self.image_dim // 16), (self.image_dim // 16))
        x = self.input_conv(x)  # Apply the initial conv layer

        # Pass through the residual block backbone with self-attention
        for block in self.blocks:
            if isinstance(block, ResUpBlock):
                x = block(x, cond_vec)
            else:  # Multi-headed self-attention doesn't require the input cond_vec
                x = block(x)

        x = self.to_rgb(x)  # Convert final output to 3 channel RGB outputs [-1, +1]
        return x
