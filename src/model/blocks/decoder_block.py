import copy

import torch
from torch import nn

from ..attention.self_attention import SelfAttention
from ..attention.pair_self_attention import PairSelfAttention
from ...utils.positional_embedding import gen_sineembed_for_position
from ...utils.misc import inverse_sigmoid


class Decoder(nn.Module):
    def __init__(self, decoder_block: nn.Module, num_decoder_blocks: int):
        super(Decoder, self).__init__()

        self._decoder = nn.ModuleList()
        for _ in range(num_decoder_blocks):
            self._decoder.append(copy.deepcopy(decoder_block))

        self._num_dec = num_decoder_blocks
        self._pos_scale = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
        )

    def forward(
        self,
        selected_objects: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: torch.Tensor,
        fine_pos: torch.Tensor,
        selected_objects_pos_embed: torch.Tensor,
        selected_centers: torch.Tensor,
        bbox_embed: nn.Module,
    ):
        x = selected_objects
        d_model2 = x.size(-1) // 2

        selected_centers_before_sigmoid = inverse_sigmoid(selected_centers)

        for dec_blk in self._decoder:
            obj_pos_trans = self._pos_scale(x[..., d_model2:])
            selected_objects_sin_embed = gen_sineembed_for_position(
                selected_centers, d_model=d_model2
            )

            selected_objects_sin_embed = selected_objects_sin_embed * obj_pos_trans

            tmp_bbox = bbox_embed(x[..., d_model2:])
            tmp_bbox[..., :2] += selected_centers_before_sigmoid

            selected_objects_coords = tmp_bbox.sigmoid()

            x = dec_blk(
                x,
                encoder_output,
                enc_pos_embed=fine_pos,
                enc_key_mask=mask,
                obj_coords=selected_objects_coords,
                obj_pos_embed=selected_objects_pos_embed,
                obj_sin_embed=selected_objects_sin_embed,
            )

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        lambda_: float = 0.5,
        hidden_dim: int = 256,
        heads_num: int = 8,
    ) -> None:
        super(DecoderBlock, self).__init__()

        self._heads_num = heads_num
        self._channels = hidden_dim

        self._lambda = lambda_
        self._self_attn = SelfAttention(
            heads_num=heads_num,
            hidden_dim=hidden_dim,
            dropout_prob=0.3,
        )
        self._pair_attn = PairSelfAttention(heads_num=heads_num)
        self._cls_branch = ClsRegBranch(
            hidden_dim=self._channels,
        )
        self._reg_branch = ClsRegBranch(
            hidden_dim=self._channels,
        )

        # projection to query, key, value for self-attention
        self._sa_proj_to_q_obj = nn.Linear(
            in_features=self._channels * 2, out_features=self._channels * 2, bias=False
        )
        self._sa_proj_to_q_pos = nn.Linear(
            in_features=self._channels, out_features=self._channels, bias=False
        )
        self._sa_proj_to_k_obj = nn.Linear(
            in_features=self._channels * 2, out_features=self._channels * 2, bias=False
        )
        self._sa_proj_to_k_pos = nn.Linear(
            in_features=self._channels, out_features=self._channels, bias=False
        )
        self._sa_proj_to_v_obj = nn.Linear(
            in_features=self._channels * 2, out_features=self._channels * 2, bias=False
        )

        # projection to query, key, value for cross-attention
        self._ca_proj_to_q_obj = nn.Linear(
            in_features=self._channels * 2, out_features=self._channels * 2, bias=False
        )
        self._ca_proj_to_q_pos = nn.Linear(
            in_features=self._channels, out_features=self._channels, bias=False
        )
        self._ca_proj_to_k_enc = nn.Linear(
            in_features=self._channels, out_features=self._channels, bias=False
        )
        self._ca_proj_to_k_pos = nn.Linear(
            in_features=self._channels, out_features=self._channels, bias=False
        )
        self._ca_proj_to_v_enc = nn.Linear(
            in_features=self._channels, out_features=self._channels, bias=False
        )

        self.norm1 = nn.LayerNorm(self._channels * 2)
        self.norm2 = nn.LayerNorm(self._channels * 2)
        self.dropout1 = nn.Dropout(0.3)

    def _split_heads(self, tensor: torch.Tensor):
        batch_size, sequence_length, embed_dim, *_ = tensor.shape

        tensor = (
            tensor.view(
                size=(
                    batch_size,
                    sequence_length,
                    self._heads_num,
                    embed_dim // self._heads_num,
                )
            )
            .transpose(1, 2)
            .contiguous()
        )

        return tensor

    def _combine_heads(self, tensor: torch.Tensor):
        tensor = tensor.transpose(1, 2).flatten(2)

        return tensor

    def forward(
        self,
        obj_selected: torch.Tensor,
        enc_output: torch.Tensor,
        obj_coords: torch.Tensor,
        obj_pos_embed: torch.Tensor,
        obj_sin_embed: torch.Tensor,
        enc_pos_embed: torch.Tensor,
        enc_key_mask: torch.Tensor,
    ):
        q_obj = self._sa_proj_to_q_obj(obj_selected)
        q_pos = self._sa_proj_to_q_pos(obj_pos_embed)
        q_pos = torch.concat([q_pos, q_pos], dim=-1)

        k_obj = self._sa_proj_to_k_obj(obj_selected)
        k_pos = self._sa_proj_to_k_pos(obj_pos_embed)
        k_pos = torch.concat([k_pos, k_pos], dim=-1)

        v = self._split_heads(self._sa_proj_to_v_obj(obj_selected))
        q = self._split_heads(q_obj + q_pos)
        k = self._split_heads(k_obj + k_pos)

        o1 = self._self_attn(query=q, key=k, value=v)
        o2 = self._pair_attn(query=q, key=k, value=v, top_k_centers=obj_coords)

        o = self._lambda * self.norm1(obj_selected + self.dropout1(o1)) + (
            1 - self._lambda
        ) * self.norm2(obj_selected + self.dropout1(o2))
        o_cls, o_reg = torch.split(
            o, split_size_or_sections=[self._channels, self._channels], dim=-1
        )

        q_obj = self._ca_proj_to_q_obj(o)
        q_pos = self._ca_proj_to_q_pos(obj_sin_embed)
        k_enc = self._ca_proj_to_k_enc(enc_output)
        k_pos = self._ca_proj_to_k_pos(enc_pos_embed)
        v2 = self._ca_proj_to_v_enc(enc_output)

        q_cls, q_reg = torch.split(
            q_obj, split_size_or_sections=[self._channels, self._channels], dim=-1
        )
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

        cls_output = self._cls_branch(
            inputs=o_cls, query=q_cls, key=k, value=v2, key_mask=enc_key_mask
        )
        reg_output = self._reg_branch(
            inputs=o_reg, query=q_reg, key=k, value=v2, key_mask=enc_key_mask
        )
        o = torch.concat([cls_output, reg_output], dim=-1)

        return o


class ClsRegBranch(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
    ):
        super(ClsRegBranch, self).__init__()

        self.cross_attn = SelfAttention(heads_num=1)

        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim * 4)
        self.fc2 = nn.Linear(in_features=hidden_dim * 4, out_features=hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: torch.Tensor,
    ):
        ca = self.cross_attn(
            query=query.unsqueeze(1),
            key=key.unsqueeze(1),
            value=value.unsqueeze(1),
            key_padding_mask=key_mask,
        )

        x = inputs + self.dropout(ca)
        x = self.norm1(x)
        res_x = self.dropout(self.fc1(x).relu())
        res_x = self.dropout(self.fc2(res_x))
        x = x + res_x
        x = self.norm2(x)

        return x


def build_decoder(
    args,
):

    decoder = Decoder(
        decoder_block=DecoderBlock(
            hidden_dim=args.hidden_dim,
        ),
        num_decoder_blocks=args.num_decoder_blocks,
    )

    return decoder
