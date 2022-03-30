import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class GlobalAveragePooling(nn.Module):

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class Classifier(nn.Module):

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Linear(input_dim, num_classes)
        nn.init.zeros_(self.model.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLPBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MixerBlock(nn.Module):

    def __init__(
        self,
        num_patches: int,
        num_channels: int,
        tokens_hidden_dim: int,
        channels_hidden_dim: int
    ):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b p c -> b c p"),
            MLPBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPBlock(num_channels, channels_hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x


class MLPMixer(nn.Module):

    def __init__(
        self,
        font_dim=100,
        word_size=5,
        num_classes=26,
        hidden_dim: int = 100,
        num_layers: int = 8,
        tokens_hidden_dim: int = 256,
        channels_hidden_dim: int = 2048
    ):
        super().__init__()
        self.embed = nn.Linear(word_size, hidden_dim)
        layers = [
            MixerBlock(
                num_patches=font_dim,
                num_channels=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )
            for _ in range(num_layers)
        ]
        self.layers = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = GlobalAveragePooling(dim=1)
        self.classifier = Classifier(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.embed(x)           # [b, p, c]
        x = self.layers(x)          # [b, p, c]
        x = self.norm(x)            # [b, p, c]
        x = self.pool(x)            # [b, c]
        x = self.classifier(x)      # [b, num_classes]
        return x
