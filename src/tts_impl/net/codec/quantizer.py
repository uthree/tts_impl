import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, channels: int, codebook_size: int, beta=0.25):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codebook_size, channels))
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass for training

        Args:
            x: (batch_size, channels, length)
        Returns:
            code: Discrete code (batch_size, length)
            embeddings: (batch_size, channels, length)
            loss: () quantization loss
        """
        codebook = self.codebook.unsqueeze(0)
        dists = torch.cdist(
            x.transpose(1, 2), codebook
        )  # (batch, length, codebook_size)
        code = torch.argmax(dists, dim=2)  # (batch, length)
        emb = F.embedding(code, self.codebook).transpose(
            1, 2
        )  # (batch, channels, length)
        loss = ((x.detach() - emb) ** 2 + self.beta * (emb.detach() - emb) ** 2).mean
        return code, emb, loss

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        quantize features

        Args:
            x: (batch_size, channels, length)
        Returns:
            code: Discrete code (batch_size, length)
        """
        code, _, _ = self.forward(x)
        return code

    def dequantize(self, code: torch.Tensor) -> torch.Tensor:
        """
        dequantize features

        Args:
            code: (batch_size, length)
        Returns:
            embeddings: (batch_size, channels, length)
        """
        emb = F.embedding(code, self.codebook).transpose(1, 2)
        return emb
