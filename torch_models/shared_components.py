"""
This module defines model components that are shared.
"""

import torch
import torch.nn as nn


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes: int = 37, embed_dim: int = 128):
        """
        Class embedding layer which maps class_id integers (B, 1) to (B, embed_dim) tensors.

        :param num_classes: The number of classes expected i.e. how many unique conditional inputs.
        :param embed_dim: The size of the output embedding vectors.
        """
        super().__init__()
        self.name = "class_embedding"
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        # Create an embedding layer to map class_id to embedding vectors of size (B, embed_dim)
        self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embed_dim)

    def forward(self, class_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the class embedding layer mapping class_id integers (B, 1) to (B, embed_dim)
        tensors.

        :param class_id: A torch.Tensor or integers [0, num_classes-1] denoting the class ID of each obs.
        """
        return self.class_embedding(class_id)


class SelfAttention(nn.Module):
    """
    Multi-headed self-attention block.
    """

    def __init__(self, channels: int):
        """
        Multi-headed self-attention block between input channels.

        :param channels: The number of input channels expected.
        """
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.self_attention = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-headed self-attention block between input channels.

        :param x: An input tensor of size (B, channels, H, W).
        :returns: An output tensor of size (B, channels, H, W).
        """
        B, C, H, W = x.shape  # Extract the shape of the input x tensor
        h = self.norm(x)  # Apply group norm
        h = h.flatten(2).transpose(1, 2)  # Reshape (B, C, H, W) -> (B, H*W, C)
        # Use each channel as a separate token, apply multi-headed self-attention between channels
        h, _ = self.self_attention(h, h, h)
        h = h.transpose(1, 2).reshape(B, C, H, W)  # Reshape (B, H*W, C) -> (B, C, H, W)
        return x + h  # Add a residual connection to the original input
