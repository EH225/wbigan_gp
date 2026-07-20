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
from torch_models.encoder import ResDownBlock
from torch_models.shared_components import SelfAttention


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
        cond_dim = self.z_dim if num_classes > 1 else 0

        if num_classes > 1:  # Create a class embedding layer if needed
            self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=z_dim)

        # An initial convolution before the residual down-sampling conv blocks
        # All examples throughout this module will be using the image_dim=64 default
        self.input_conv = nn.Conv2d(3, 64, 3, padding=1)  # Out: (B, 64, 64, 64)

        # Define the discriminator CNN encoder backbone as a series of down-sampling residual conv blocks
        self.blocks = nn.Sequential(
            ResDownBlock(64, 128, cond_dim=0),  # (B, 64, 64, 64) -> (B, 128, 32, 32)
            ResDownBlock(128, 256, cond_dim=0),  # (B, 128, 32, 32) -> (B, 256, 16, 16)
            SelfAttention(256),  # (B, 256, 16, 16) -> (B, 256, 16, 16)
            ResDownBlock(256, 512, cond_dim=0),  # (B, 256, 16, 16) -> (B, 512, 8, 8)
            ResDownBlock(512, 512, cond_dim=0),  # (B, 512, 8, 8) -> (B, 512, 4, 4)
        )

        # Add a final convolution to mix spatial information before compressing
        self.final_conv = nn.Sequential(
            nn.GroupNorm(16, 512),
            nn.SiLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
        )  # (B, 512, 4, 4) -> (B, 512, 4, 4)

        # Define a small MLP to process the input latent z vector
        self.mlp_z = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim * 2),
            nn.SiLU(),
            nn.Linear(self.z_dim * 2, self.z_dim * 2),
        )

        # Define a small MLP to process the class embedding vector
        self.mlp_c = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim * 2),
            nn.SiLU(),
            nn.Linear(self.z_dim * 2, self.z_dim * 2),
        )

        # Define the final MLP that will operate on the fusion of all 3 inputs (x, z, class_embed)
        self.mlp = nn.Sequential(
            nn.Linear(512 + (4 if self.num_classes > 1 else 2) * self.z_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

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
        x = self.blocks(x)  # Pass through the residual CNN encoder blocks (B, 512, 4, 4)
        x = self.final_conv(x)  # Pass through a final conv to max information along the spatial dimension
        x = F.adaptive_avg_pool2d(x, 1)  # Downsample further (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # Flatten (B, 512, 1, 1) -> (B, 512)

        # 2). Apply an MLP processing step to the input z-vector
        z = self.mlp_z(z)  # (B, z_dim) -> (B, 2*z_dim)
        z = F.layer_norm(z, z.shape[-1:])  # Add layer norm to normalize values to avoid scale diffs

        # 3). Apply an MLP processing step to the input class embedding if num_classes > 0 and feature
        # fusion by concatenating all inputs together and then passing to an MLP
        # With class conditioning: (B, 512) + (B, 2*z_dim) + (B, 2*z_dim) = (B, 512 + 4*z_dim)
        # Otherwise: (B, 512) + (B, 2*z_dim) = (B, 512 + 2*z_dim)
        if self.num_classes > 1:
            class_embed = self.mlp_c(self.class_embedding(class_id))  # (B,) -> (B, z_dim) -> (B, 2*z_dim)
            class_embed = F.layer_norm(class_embed, class_embed.shape[-1:])
            x = torch.cat([x, z, class_embed], dim=1)
        else:
            x = torch.cat([x, z], dim=1)
        return self.mlp(x)  # (B, 1)
