import torch
from torch import nn
from torchinfo import summary

from einops import repeat

# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0., batch_first=True):
        super(Attention, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=batch_first)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x, weight = self.self_attn(src, src, src, attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., batch_first=True):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        attn = nn.Sequential(nn.LayerNorm(dim),
                             Attention(dim=dim, heads=heads, dropout=dropout, batch_first=batch_first))
        ff = nn.Sequential(nn.LayerNorm(dim),
                           FeedForward(dim, mlp_dim, dropout=dropout))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                attn,
                ff]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViTmodel(nn.Module):
    def __init__(self, *, font_dim, word_size, num_classes, embed_dim, heads, mlp_dim, depth,
                 pool='cls', dropout=0., emb_dropout=0., batch_first=True):
        super(ViTmodel, self).__init__()
        self.input_shape = (font_dim, word_size)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.embedding = nn.Linear(word_size, embed_dim, bias=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(embed_dim, depth, heads, mlp_dim, dropout, batch_first)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, font):
        x = self.embedding(font)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == '__main__':
    font_dim = 100
    word_size = 5
    num_classes = 26
    model = ViTmodel(
        font_dim=font_dim,
        word_size=word_size,
        num_classes=num_classes,
        embed_dim=5,
        depth=6,
        heads=5,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    )

    image_batch = torch.rand(256, *model.input_shape)
    print(image_batch.shape)
    image_batch = torch.arange(image_batch.numel()).reshape(image_batch.shape).float()
    model(image_batch)
    summary(model, (256, *model.input_shape), device='cpu')
