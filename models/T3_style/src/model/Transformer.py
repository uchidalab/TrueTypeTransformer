import torch
from torch import nn, Tensor
from torchinfo import summary


class TransformerModel(nn.Module):
    def __init__(
        self,
        *,
        font_dim: int = 100,
        word_size: int = 5,
        num_classes: int = 26,
        embed_dim: int = 100,
        nhead: int = 5,
        dim_feedforward: int = 1024,
        depth: int = 3,
        dropout: float = 0.2
    ) -> None:
        super(TransformerModel, self).__init__()

        self.input_shape = (font_dim, word_size)
        self.word_size = word_size
        self.embed_dim = embed_dim
        self.font_size = font_dim * embed_dim

        self.pos_embedding = nn.Linear(self.word_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=depth
        )

        self.mlp_head = nn.Linear(self.font_size, num_classes)

    def forward(self, font: Tensor) -> Tensor:
        batch_size = len(font)
        src_mask = (font == -1)[:, :, 0]
        x = self.pos_embedding(font)
        x = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        x = x.reshape(batch_size, -1)
        verify_shape = torch.Size([batch_size, self.font_size])
        assert x.shape == verify_shape, f'{x.shape}, {verify_shape}'

        x = self.mlp_head(x)
        return x


if __name__ == '__main__':
    font_dim = 100
    word_size = 5
    num_classes = 26

    model = TransformerModel(
        font_dim=font_dim,
        word_size=word_size,
        num_classes=num_classes)

    image_batch = torch.rand(256, *model.input_shape)
    print(image_batch.shape)
    image_batch = torch.arange(image_batch.numel()).reshape(image_batch.shape).float()
    model(image_batch)
    summary(model, (256, *model.input_shape), device='cpu')
