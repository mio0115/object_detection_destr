import torchvision as tv
import torch
from torch.nn import LayerNorm, Conv2d, Linear, Module, Sequential, Sigmoid, Softmax

from .blocks.encoder_block import EncoderBlock
from .blocks.decoder_block import DecoderBlock
from .blocks.mini_detector import MiniDetector
from ..utils.positional_embedding import gen_sineembed_for_position


class ObjDetSplitTransformer(Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_cls: int,
        hidden_dim: int,
        num_encoder_blocks: int = 6,
        num_decoder_blocks: int = 6,
        top_k: int = 500,
    ):
        super(ObjDetSplitTransformer, self).__init__()

        self._backbone = tv.models.resnet50()

        self._hidden_dim = hidden_dim
        self._cls_ffn = Sequential(
            [Linear(in_features=512, out_features=num_cls), Softmax()]
        )
        self._reg_ffn = Sequential(
            [
                Linear(in_features=512, out_features=256),
                Linear(in_features=256, out_features=256),
                Linear(in_features=256, out_features=4),
            ]
        )
        self._pos_scale = Sequential(
            [
                Linear(in_features=512, out_features=256),
                Linear(in_features=256, out_features=2),
            ]
        )

        self._reduce_dim = Conv2d(
            in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=1
        )
        self._mini_detector = MiniDetector(
            input_shape=(7, 7, hidden_dim),
            cls_num=num_cls,
            top_k=top_k,
            reg_ffn=self._reg_ffn,
        )
        self._position_index_2d = None

        self._num_encoders = num_encoder_blocks
        self._num_decoders = num_decoder_blocks

        self._encoder_blocks = Sequential(
            [
                EncoderBlock(
                    block_idx=idx,
                    position_index_2d=self._position_index_2d,
                    input_shape=(49, hidden_dim),
                    heads_num=8,
                    d_k=hidden_dim,
                    d_v=hidden_dim,
                )
                for idx in range(num_encoder_blocks)
            ]
        )
        self._decoder_blocks = [
            DecoderBlock(object_queries_shape=(top_k, hidden_dim), lambda_=0.5)
            for idx in range(num_decoder_blocks)
        ]

    def forward(self, inputs):
        batch_size = inputs.size(0)

        x = self._backbone(inputs)
        x = self._reduce_dim(x).flatten(1, 2)

        x = self._encoder_blocks(x)
        encoder_output = x

        x = x.view(shape=(batch_size, 7, 7, self._hidden_dim))
        top_k_proposals, top_k_coords, all_proposals = self._mini_detector(x)

        top_k_proposals = top_k_proposals.detach()
        top_k_coords = top_k_coords.detach()

        top_k_centers = top_k_coords[..., :2]
        top_k_centers_before_sigmoid = -1 * (top_k_centers.pow(-1) - 1).log()

        top_k_centers_before_sigmoid = torch.concat(
            [
                top_k_centers_before_sigmoid,
                torch.zeros_like(top_k_centers_before_sigmoid, dtype=torch.float32),
            ],
            dim=-1,
        )

        obj_pos_embed = gen_sineembed_for_position(pos_tensor=top_k_coords, d_model=256)

        x = top_k_proposals

        for idx in range(self._num_decoders):
            _, reg_x = torch.split(x, split_size_or_sections=2, dim=-1)
            obj_pos_trans = self._pos_scale(reg_x)
            obj_pos_embed = top_k_centers * obj_pos_trans

            tmp_bbox = self._reg_ffn(reg_x)
            tmp_bbox = tmp_bbox + top_k_centers_before_sigmoid
            top_k_coords = Sigmoid()(tmp_bbox)

            x = self._decoder_blocks[idx](
                x,
                encoder_output,
                obj_coords=top_k_coords,
                obj_pos_embed=obj_pos_embed,
                obj_sin_embed=obj_pos_embed,
            )

        cls_x, reg_x = torch.split(x, split_size_or_sections=2, dim=-1)

        cls_output = self._cls_ffn(cls_x)
        bbox_output = Sigmoid()(self._reg_ffn(reg_x) + top_k_centers_before_sigmoid)

        idx = torch.argmax(torch.max(cls_output, dim=-1), dim=-1)
        cls_output = torch.gather(cls_output, dim=1, index=idx)
        bbox_output = torch.gather(bbox_output, dim=1, index=idx)

        return cls_output, bbox_output, all_proposals
