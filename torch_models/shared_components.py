"""
This module defines model components that are shared.
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Multi-headed self-attention block.
    """

    def __init__(self, channels: int, groups: int = 16):
        """
        Multi-headed self-attention block between input channels.

        :param channels: The number of input channels expected.
        :param groups: The number of groups for group norm.
        """
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.self_attention = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        # Define a trainable scaling factor for the residual connection to improve stability
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-headed self-attention block between input channels.

        :param x: An input tensor of size (B, channels, H, W).
        :returns: An output tensor of size (B, channels, H, W).
        """
        B, C, H, W = x.shape  # Extract the shape of the input x tensor
        h = self.norm(x)  # Apply group norm
        h = h.flatten(2).transpose(1, 2)  # Reshape (B, C, H, W) -> (B, H*W, C) -> (B, T, E)
        # Use each spacial location as a separate token, apply multi-headed self-attention to tokens
        h, _ = self.self_attention(h, h, h)
        h = h.transpose(1, 2).reshape(B, C, H, W)  # Reshape (B, H*W, C) -> (B, C, H, W)
        return h * self.res_scale + x  # Add a residual connection to the original input
