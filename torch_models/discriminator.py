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
    Conditional discriminator model for a Wasserstein Bi-GAN.

    This model operates on:
        - An input image tensor of size (B, 3, 128, 128)
        - A latent z tensor of size (B, z_dim)
        - A class embedding tensor of size (B, z_dim)

    and jointly reasons about the image features and the latent vector. It outputs a single real-valued
    score: D(x, z) ∈ R denoting how similar the inputs are to the observed data distribution. During training
    the discriminator tries to maximize: E[D(x_real, E(x_real))] - E[D(G(z), z)]
    with real image examples (x_real, E(x_real)) and fake image examples ((G(z), z)).
    It judges whether an (image, latent) pair comes from the joint data distribution or the joint model
    distribution. The critic never sees mismatched pairs i.e. only (real image, encoded latent z) or
    (generated image, sampled latent z). The encoder and generator are trained together so that these two
    joint distributions become indistinguishable.
    """

    def __init__(self, z_dim: int = 128):
        """
        A conditional discriminator (critic) model for a Wasserstein Bi-GAN.

        :param z_zim: The dimension of the output latent noise vector, z. This is also the dimension of the
            embedding vectors used to represent each class. The default is 128.
        """
        super().__init__()
        self.name = "discriminator"
        self.z_dim = z_dim

        # An initial convolution before the residual down-sampling conv blocks
        self.input_conv = nn.Conv2d(3, 64, 3, padding=1)  # Out: (B, 64, 128, 128)

        # Define the discriminator CNN encoder backbone as a series of down-sampling residual conv blocks
        self.blocks = nn.Sequential(
            ResDownBlock(64, 128),  # (B, 64, 128, 128) -> (B, 128, 64, 64)
            ResDownBlock(128, 256),  # (B, 128, 64, 64) -> (B, 256, 32, 32)
            SelfAttention(256),  # (B, 256, 32, 32) -> (B, 256, 32, 32)
            ResDownBlock(256, 512),  # (B, 256, 32, 32) -> (B, 512, 16, 16)
            ResDownBlock(512, 512),  # (B, 512, 16, 16) -> (B, 512, 8, 8)
            ResDownBlock(512, 512),  # (B, 512, 8, 8) -> (B, 512, 4, 4)
        )

        # Define a small MLP to process the input latent z vector
        self.mlp_z = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim * 2),
            nn.SiLU(),
            nn.Linear(self.z_dim * 2, self.z_dim * 2),
        )

        # Define a small MLP to process the input class embedding vector
        self.mlp_c = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim * 2),
            nn.SiLU(),
            nn.Linear(self.z_dim * 2, self.z_dim * 2),
        )

        # Define the final MLP that will operate on the fusion of all 3 inputs (x, z, class_embed)
        self.mlp = nn.Sequential(
            nn.Linear(512 + 4 * self.z_dim, 768),
            nn.SiLU(),
            nn.Linear(768, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor, class_embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator model. Discriminator(image, z, class_embed) -> float

        :param x: An input image tensor of size (B, 3, 128, 128) with pixel values [-1, +1].
        :param z: An input latent tensor of size (B, z_dim).
        :param class_embed: An input tensor of class label embeddings of size (B, z_dim).
        :returns: A tensor of size (B, 1) containing the Wasserstein scores for each input pair.
        """
        assert z.shape == class_embed.shape, "z.shape must equal class_embed.shape"
        assert z.shape[-1] == self.z_dim, "z must be (B, z_dim)"
        assert class_embed.shape[-1] == self.z_dim, "class_embed must be (B, z_dim)"

        # 1). Process the input images, pass it through the CNN backbone to generate a deep latent rep.
        x = self.input_conv(x)  # Apply an initial conv (B, 3, 128, 128) -> (B, 64, 128, 128)
        x = self.blocks(x)  # Pass through the residual CNN encoder blocks (B, 512, 4, 4)
        x = F.adaptive_avg_pool2d(x, 1)  # Downsample further (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # Flatten (B, 512, 1, 1) -> (B, 512)

        # 2). Apply an MLP processing step to the input z-vector
        z = self.mlp_z(z)

        # 3). Apply an MLP processing step to the input class embedding
        class_embed = self.mlp_c(class_embed)

        # 4). Perform feature fusion by concatenating all inputs together and then passing to an MLP
        # (B, 512) + (B, z_dim*2) + (B, z_dim*2) = (B, 512 + z_dim*4)
        x = torch.cat([x, z, class_embed], dim=1)
        return self.mlp(x)  # (B, 1)
