import copy
from typing import Optional

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, encoder_block: nn.Module, num_encoder_blocks: int = 6):
        super(Encoder, self).__init__()

        self._encoder = nn.ModuleList()
        for _ in range(num_encoder_blocks):
            self._encoder.append(copy.deepcopy(encoder_block))

        self._num_enc = num_encoder_blocks
        self._pos_scale = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
        )
        self.norm = nn.LayerNorm(256)

    def forward(self, inputs, mask, pos_embed):
        # inputs shape:     (batch_size, channels, height, width)
        # reshape inputs to (height * width, batch_size, channels)
        batch_size, channels, height, width = inputs.shape

        x = inputs.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        for enc_blk in self._encoder:
            scale = self._pos_scale(x)
            tmp_x = enc_blk(
                x,
                key_mask=mask,
                pos_embed=pos_embed * scale,
            )
            x = x + tmp_x

        x = self.norm(x)
        x = x.permute(1, 2, 0).view(batch_size, channels, height, width)

        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        heads_num: int = 8,
        d_k: int = 256,
        d_v: int = 256,
    ):
        super(EncoderBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads_num,
            dropout=0.3,
            kdim=hidden_dim,
            vdim=hidden_dim,
        )
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=hidden_dim)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # self._input_shape = input_shape
        self._heads_num = heads_num
        self._hidden_dim = hidden_dim
        self._proj_to_q = nn.Linear(
            in_features=self._hidden_dim, out_features=d_k, bias=False
        )
        self._proj_to_k = nn.Linear(
            in_features=self._hidden_dim, out_features=d_k, bias=False
        )
        self._proj_to_v = nn.Linear(in_features=d_k, out_features=d_v, bias=False)

    @property
    def heads_num(self):
        return self._heads_num

    def forward(
        self,
        inputs,
        mask: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
    ):
        to_q_k = inputs + pos_embed

        tmp_x, _ = self.self_attn(
            query=to_q_k,
            key=to_q_k,
            value=inputs,
            attn_mask=mask,
            key_padding_mask=key_mask,
        )
        x = inputs + self.dropout1(tmp_x)

        x = self.norm1(x)
        res_x = self.dropout2(self.fc1(x).relu())
        res_x = self.dropout3(self.fc2(res_x))
        x = x + res_x
        x = self.norm2(x)

        return x


def build_encoder(args):
    encoder = Encoder(
        encoder_block=EncoderBlock(
            d_k=args.hidden_dim,
            d_v=args.hidden_dim,
        ),
        num_encoder_blocks=args.num_encoder_blocks,
    )

    return encoder
