import torch
from torch.nn import Linear, Dropout, LayerNorm, Module, ReLU

from ..attention.self_attention import SelfAttention
from ..attention.pair_self_attention import PairSelfAttention
from ...utils.positional_embedding import gen_sineembed_for_position


class DecoderBlock(Module):
    def __init__(
        self,
        object_queries_shape,
        position_index_2d,
        lambda_: float = 0.5,
        hidden_dim: int = 256,
    ) -> None:
        super(DecoderBlock, self).__init__()

        self._object_queries_shape = object_queries_shape
        self._heads_num = 8
        self._channels = hidden_dim
        self._pos_embed_2d = gen_sineembed_for_position(position_index_2d)

        self._lambda = lambda_
        self._self_attn = SelfAttention(
            heads_num=8,
            input_shape=object_queries_shape,
            output_shape=object_queries_shape,
        )
        self._pair_attn = PairSelfAttention(
            heads_num=8, input_shape=object_queries_shape
        )
        self._cls_branch = ClsRegBranch(
            attn_input_shape=object_queries_shape,
            attn_output_shape=(object_queries_shape[0], object_queries_shape[1] // 2),
            hidden_dim=self._channels,
        )
        self._reg_branch = ClsRegBranch(
            attn_input_shape=object_queries_shape,
            attn_output_shape=(object_queries_shape[0], object_queries_shape[1] // 2),
            hidden_dim=self._channels,
        )

        # projection to query, key, value for self-attention
        self._sa_proj_to_q_obj = Linear(
            in_features=self._channels * 2, out_features=self._channels * 2, bias=False
        )
        self._sa_proj_to_q_pos = Linear(
            in_features=self._channels, out_features=self._channels, bias=False
        )
        self._sa_proj_to_k_obj = Linear(
            in_features=self._channels * 2, out_features=self._channels * 2, bias=False
        )
        self._sa_proj_to_k_pos = Linear(
            in_features=self._channels, out_features=self._channels, bias=False
        )
        self._sa_proj_to_v_obj = Linear(
            in_features=self._channels * 2, out_features=self._channels * 2, bias=False
        )

        # projection to query, key, value for cross-attention
        self._ca_proj_to_q_obj = Linear(
            in_features=self._channels * 2, out_features=self._channels * 2, bias=False
        )
        self._ca_proj_to_q_pos = Linear(
            in_features=self._channels, out_features=self._channels, bias=False
        )
        self._ca_proj_to_k_enc = Linear(
            in_features=self._channels, out_features=self._channels, bias=False
        )
        self._ca_proj_to_k_pos = Linear(
            in_features=self._channels, out_features=self._channels, bias=False
        )
        self._ca_proj_to_v_enc = Linear(
            in_features=self._channels, out_features=self._channels, bias=False
        )

        self.norm1 = LayerNorm(self._channels * 2)
        self.norm2 = LayerNorm(self._channels * 2)
        self.dropout1 = Dropout(0.3)

    def _split_heads(self, tensor: torch.Tensor):
        batch_size, sequence_length, embed_dim, *_ = tensor.shape

        tensor = tensor.view(
            shape=(
                batch_size,
                sequence_length,
                self._heads_num,
                embed_dim // self._heads_num,
            )
        ).transpose(1, 2)

        return tensor

    def _combine_heads(self, tensor: torch.Tensor):
        batch_size, _, sequence_length, embed_dim = tensor.shape

        tensor = tensor.transpose(1, 2).view(
            shape=(batch_size, sequence_length, self._heads_num * embed_dim)
        )

        return tensor

    def forward(
        self,
        object_queries: torch.Tensor,
        encoder_output: torch.Tensor,
        obj_coords: torch.Tensor,
        obj_pos_embed: torch.Tensor,
        obj_sin_embed: torch.Tensor,
    ):
        q_obj = self._sa_proj_to_q_obj(object_queries)
        q_pos = self._sa_proj_to_q_pos(obj_pos_embed)
        q_pos = torch.concat([q_pos, q_pos], dim=-1)

        k_obj = self._sa_proj_to_k_obj(object_queries)
        k_pos = self._sa_proj_to_k_pos(obj_pos_embed)
        k_pos = torch.concat([k_pos, k_pos], dim=-1)

        v = self._split_heads(self._sa_proj_to_v_obj(object_queries))
        q = self._split_heads(q_obj + q_pos)
        k = self._split_heads(k_obj + k_pos)

        o1 = self._self_attn(query=q, key=k, value=v)
        o2 = self._pair_attn(query=q, key=k, value=v, top_k_centers=obj_coords)

        o = self._lambda * self.norm1(object_queries + self.dropout1(o1)) + (
            1 - self._lambda
        ) * self.norm2(object_queries + self.dropout1(o2))
        o_cls, o_reg = torch.split(o, split_size_or_sections=2, dim=-1)

        q_obj = self._ca_proj_to_q_obj(o)
        q_pos = self._ca_proj_to_q_pos(obj_sin_embed)
        k_enc = self._ca_proj_to_k_enc(encoder_output)
        k_pos = self._ca_proj_to_k_pos(self._pos_embed_2d.flatten(1, 2))
        v2 = self._ca_proj_to_v_enc(encoder_output)

        q_cls, q_reg = torch.split(q_obj, split_size_or_sections=2, dim=-1)
        q_cls = self._split_heads(q_cls)
        q_reg = self._split_heads(q_reg)
        q_pos = self._split_heads(q_pos)
        q_cls = torch.concat([q_cls, q_pos], dim=-1)
        q_reg = torch.concat([q_reg, q_pos], dim=-1)
        q_cls = self._combine_heads(q_cls)
        q_reg = self._combine_heads(q_reg)

        k_enc = self._split_heads(k_enc)
        k_pos = self._split_heads(k_pos)
        k = torch.concat([k_enc, k_pos], dim=-1)
        k = self._combine_heads(k)

        cls_output = self._cls_branch(inputs=o_cls, query=q_cls, key=k, value=v2)
        reg_output = self._reg_branch(inputs=o_reg, query=q_reg, key=k, value=v2)
        o = torch.concat([cls_output, reg_output], dim=-1)

        return o


class ClsRegBranch(Module):
    def __init__(
        self,
        attn_input_shape: tuple[int, int],
        attn_output_shape: tuple[int, int],
        hidden_dim: int = 256,
    ):
        super(ClsRegBranch, self).__init__()

        self.cross_attn = SelfAttention(
            heads_num=1, input_shape=attn_input_shape, output_shape=attn_output_shape
        )

        self.fc1 = Linear(
            in_features=attn_output_shape[-1], out_features=hidden_dim * 4
        )
        self.fc2 = Linear(in_features=hidden_dim * 4, out_features=hidden_dim)
        self.dropout = Dropout(0.3)
        self.relu = ReLU()
        self.norm1 = LayerNorm(attn_output_shape[-1])
        self.norm2 = LayerNorm(hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        ca = self.cross_attn(
            query=query.unsqueeze(0), key=key.unsqueeze(0), value=value.unsqueeze(0)
        )

        x = inputs + self.dropout(ca)
        x = self.norm1(x)
        res_x = self.dropout(self.relu(self.fc1(x)))
        res_x = self.dropout(self.fc2(res_x))
        x = x + res_x
        x = self.norm2(x)

        return x
