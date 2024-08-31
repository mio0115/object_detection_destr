import torch
from torch import nn
from torch.nn import Linear, Dropout, LayerNorm, ReLU

from ..attention.self_attention import SelfAttention
from ...utils.positional_embedding import gen_sineembed_for_position


class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        block_idx: int,
        position_index_2d: torch.Tensor,
        d_model: int = 256,
        input_shape: tuple[int, int] = (49, 256),
        heads_num: int = 8,
        d_k: int = 256,
        d_v: int = 256,
    ):
        super(EncoderBlock, self).__init__()

        self.self_attn = SelfAttention(
            heads_num=heads_num, input_shape=input_shape, output_shape=input_shape
        )
        self.fc1 = Linear(in_features=d_model, out_features=2048)
        self.fc2 = Linear(in_features=2048, out_features=d_model)

        self.dropout1 = Dropout(0.3)
        self.dropout2 = Dropout(0.3)
        self.dropout3 = Dropout(0.3)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self._input_shape = input_shape
        self._heads_num = heads_num
        self._hidden_dim = d_model
        self._pos_embed_2d = gen_sineembed_for_position(
            position_index_2d, d_model=d_model
        )
        self._proj_to_q = Linear(
            in_features=self.embedding_dim, out_features=d_k, bias=False
        )
        self._proj_to_k = Linear(
            in_features=self.embedding_dim, out_features=d_k, bias=False
        )
        self._proj_to_v = Linear(in_features=d_k, out_features=d_v, bias=False)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def sequence_length(self):
        return self._input_shape[0]

    @property
    def embedding_dim(self):
        return self._input_shape[1]

    @property
    def heads_num(self):
        return self._heads_num

    def _split_heads(self, tensor: torch.Tensor, batch_size: int):
        tensor = tensor.view(
            size=(
                batch_size,
                self.sequence_length,
                self.heads_num,
                self.embedding_dim // self.heads_num,
            )
        ).transpose(1, 2)

        return tensor

    def forward(self, inputs):
        batch_size = inputs.size(0)

        to_q_k = self._split_heads(
            inputs + self._pos_embed_2d.flatten(1, 2), batch_size=batch_size
        )
        q = self._proj_to_q(to_q_k)
        k = self._proj_to_k(to_q_k)

        to_v = self._split_heads(inputs, batch_size)
        v = self._proj_to_v(to_v)

        res_x = self.self_attn(query=q, key=k, value=v)
        x = x + self.dropout1(res_x)

        x = self.norm1(x)
        res_x = self.dropout2(self.fc1(x).relu())
        res_x = self.dropout3(self.fc2(res_x))
        x = x + res_x
        x = self.norm2(x)

        return x


def build_encoder(
    position_index_2d, hidden_dim: int = 256, num_encoder_blocks: int = 6
):
    encoder = nn.ModuleList(
        [
            EncoderBlock(
                block_idx=idx,
                position_index_2d=position_index_2d,
                input_shape=(49, hidden_dim),
                heads_num=8,
                d_k=hidden_dim,
                d_v=hidden_dim,
            )
            for idx in range(num_encoder_blocks)
        ]
    )

    return encoder
