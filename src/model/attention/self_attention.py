import math
from typing import Optional

import torch
from torch import nn


class SelfAttention(torch.nn.Module):
    def __init__(
        self, heads_num: int = 8, dropout_prob: float = 0.3, hidden_dim: int = 256
    ):
        super(SelfAttention, self).__init__()

        self._num_heads = heads_num
        self._dropout_prob = dropout_prob
        self._hidden_dim = hidden_dim

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        attn_sc = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.size(-1))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_sc = attn_sc.masked_fill(attn_mask, float("-inf"))
            else:
                attn_sc += attn_mask

        if key_padding_mask is not None:
            attn_sc = attn_sc.masked_fill(
                key_padding_mask.bool().unsqueeze(1).unsqueeze(1), float("-inf")
            )

        attn_sc = torch.softmax(input=attn_sc, dim=-1)
        attn_sc = nn.Dropout(self._dropout_prob)(attn_sc)
        output = torch.matmul(
            attn_sc, value
        )  # shape of output (batch_size, num_heads, num_objects, d_v)

        output = output.transpose(1, 2).flatten(2).contiguous()

        return output
