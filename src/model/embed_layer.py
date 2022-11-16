import torch
import torch.nn as nn
import math


class PositionalEncodingSinCos(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncodingSinCos, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class queryEmbedding(nn.Module):
    def __init__(self, seq_len, n_contours, n_orders,
                 n_args=2, args_dim=257, n_flag=2, d_model=64,
                 use_contours=False, group_len=None):
        super().__init__()

        self.contour_embed = nn.Embedding(n_contours + 1, d_model)
        self.order_embed = nn.Embedding(n_orders + 1, d_model)
        self.flag_embed = nn.Embedding(n_flag + 1, d_model)

        args_dim = 2 * args_dim

        self.arg_embed = nn.Embedding(args_dim, 64)
        self.embed_fcn = nn.Linear(64 * n_args, d_model)

        self.use_contours = use_contours
        if use_contours:
            if group_len is None:
                group_len = n_contours
            self.group_embed = nn.Embedding(group_len+2, d_model)

        self.pos_encoding = PositionalEncodingSinCos(d_model, max_len=seq_len+2)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.order_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.arg_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.embed_fcn.weight, mode="fan_in")

        if self.use_contours:
            nn.init.kaiming_normal_(self.group_embed.weight, mode="fan_in")

    def forward(self, fonts, groups=None):
        b, S, EN = fonts.shape

        src = self.flag_embed((fonts[:, :, 2]+1).long()) +\
            self.contour_embed((fonts[:, :, 3]+1).long()) + \
            self.order_embed((fonts[:, :, 4]+1).long()) +\
            self.embed_fcn(self.arg_embed((fonts[:, :, :2] + 1).long()).view(b, S, -1))  # shift due to -1 PAD_VAL

        if self.use_contours:
            src = src + self.group_embed(groups.long())

        # src = self.pos_encoding(src)

        return src
