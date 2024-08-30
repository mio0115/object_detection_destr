import torchvision as tv
import torch
from torch.nn import Conv2d, Linear, Module, Sequential, Identity, Softmax, ModuleList

from .blocks.encoder_block import EncoderBlock
from .blocks.decoder_block import DecoderBlock
from .blocks.mini_detector import MiniDetector
from ..utils.positional_embedding import gen_sineembed_for_position


class ObjDetSplitTransformer(Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_cls: int = 2,
        hidden_dim: int = 256,
        loaded_model: bool = False,  # if we have loaded the model, we do not need to load resnet again.
        num_encoder_blocks: int = 6,
        num_decoder_blocks: int = 6,
        top_k: int = 500,
    ):
        super(ObjDetSplitTransformer, self).__init__()

        # we use resnet50 as feature extractor
        self._backbone = tv.models.resnet50()
        if not loaded_model:
            self._backbone.load_state_dict(
                torch.load("/workspace/checkpoints/resnet50-0676ba61.pth"),
                weights_only=True,
            )
        for param in self._backbone.parameters():
            param.requires_grad = False
        self._backbone.fc = Identity()

        self._hidden_dim = hidden_dim
        self._cls_ffn = Sequential(
            Linear(in_features=self._hidden_dim, out_features=num_cls), Softmax()
        )
        self._reg_ffn = Sequential(
            Linear(in_features=self._hidden_dim, out_features=self._hidden_dim),
            Linear(in_features=self._hidden_dim, out_features=self._hidden_dim),
            Linear(in_features=self._hidden_dim, out_features=4),
        )
        self._pos_scale = Sequential(
            Linear(in_features=self._hidden_dim, out_features=self._hidden_dim),
            Linear(in_features=self._hidden_dim, out_features=2),
        )

        # the output channels from ResNet50 is 2048
        self._reduce_dim = Conv2d(
            in_channels=2048,
            out_channels=self._hidden_dim,
            kernel_size=(1, 1),
            stride=1,
        )
        self._mini_detector = MiniDetector(
            input_shape=(7, 7, self._hidden_dim),
            cls_num=num_cls,
            top_k=top_k,
            reg_ffn=self._reg_ffn,
        )

        # position index to be embeded
        # Ex: [[[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]], [[2, 0], [2, 1], [2, 2]]]
        idx_tensor = torch.arange(0, 7, dtype=torch.float32)
        idx_tensor1 = idx_tensor.unsqueeze(0).broadcast_to((7, 7))
        idx_tensor2 = idx_tensor.unsqueeze(1).broadcast_to((7, 7))
        self._position_index_2d = torch.stack([idx_tensor2, idx_tensor1], dim=-1)

        self._num_encoders = num_encoder_blocks
        self._num_decoders = num_decoder_blocks

        self._encoder_blocks = ModuleList(
            [
                EncoderBlock(
                    block_idx=idx,
                    position_index_2d=self._position_index_2d,
                    input_shape=(49, self._hidden_dim),
                    heads_num=8,
                    d_k=self._hidden_dim,
                    d_v=self._hidden_dim,
                )
                for idx in range(num_encoder_blocks)
            ]
        )
        self._decoder_blocks = ModuleList(
            [
                DecoderBlock(
                    object_queries_shape=(top_k, self._hidden_dim),
                    lambda_=0.5,
                    position_index_2d=self._position_index_2d,
                    hidden_dim=self._hidden_dim,
                )
                for idx in range(num_decoder_blocks)
            ]
        )

    def forward(self, inputs):
        batch_size = inputs.size(0)

        x = self._backbone(inputs)
        x = self._reduce_dim(x).flatten(1, 2)

        for encoder_block in self._encoder_blocks:
            x = encoder_block(x)
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

        for decoder_block in self._decoder_blocks:
            reg_x = x[..., self._hidden_dim :]
            obj_pos_trans = self._pos_scale(reg_x)
            obj_pos_embed = top_k_centers * obj_pos_trans

            tmp_bbox = self._reg_ffn(reg_x)
            tmp_bbox = tmp_bbox + top_k_centers_before_sigmoid
            top_k_coords = tmp_bbox.sigmoid()

            x = decoder_block(
                x,
                encoder_output,
                obj_coords=top_k_coords,
                obj_pos_embed=obj_pos_embed,
                obj_sin_embed=obj_pos_embed,
            )

        cls_x, reg_x = torch.split(x, [self._hidden_dim, self._hidden_dim], dim=-1)

        cls_output = self._cls_ffn(cls_x)
        bbox_output = torch.sigmoid(self._reg_ffn(reg_x) + top_k_centers_before_sigmoid)

        idx = torch.argmax(torch.max(cls_output, dim=-1), dim=-1)
        cls_output = torch.gather(cls_output, dim=1, index=idx)
        bbox_output = torch.gather(bbox_output, dim=1, index=idx)

        return cls_output, bbox_output, all_proposals
