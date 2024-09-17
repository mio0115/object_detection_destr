import os

import torch
from torch import nn

from .blocks.encoder_block import build_encoder
from .blocks.decoder_block import build_decoder
from .blocks.backbone import build_backbone_customized
from .blocks.mini_detector import MiniDetector
from ..utils.misc import nested_tensor_from_tensor_list, inverse_sigmoid
from ..utils.positional_embedding import gen_sineembed_for_position


class ObjDetSplitTransformer(nn.Module):
    def __init__(
        self,
        args,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super(ObjDetSplitTransformer, self).__init__()

        # we use resnet50 as backbone
        self._backbone = backbone
        self._encoder = encoder
        self._decoder = decoder

        self._hidden_dim = args.hidden_dim
        self._cls_embed = nn.Linear(
            in_features=self._hidden_dim, out_features=args.num_cls
        )
        self._bbox_embed = nn.Sequential(
            nn.Linear(
                in_features=self._hidden_dim, out_features=self._hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(in_features=self._hidden_dim, out_features=4),
        )
        self._reg_ffn = nn.Sequential(
            nn.Linear(
                in_features=self._hidden_dim, out_features=self._hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self._hidden_dim, out_features=self._hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(in_features=self._hidden_dim, out_features=2),
        )
        self._pos_scale = nn.Sequential(
            nn.Linear(
                in_features=self._hidden_dim, out_features=self._hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(in_features=self._hidden_dim, out_features=2),
        )

        # the output channels from ResNet50 is 2048
        self._reduce_dim = nn.Conv2d(
            in_channels=2048,
            out_channels=self._hidden_dim,
            kernel_size=(1, 1),
            stride=1,
        )
        self._mini_detector = MiniDetector(
            top_k=args.top_k,
            reg_ffn=self._reg_ffn,
            class_embed=self._cls_embed,
            bbox_embed=self._bbox_embed,
        )

    def forward(self, inputs):
        if isinstance(inputs, (list, torch.Tensor)):
            inputs = nested_tensor_from_tensor_list(inputs)

        features, pos = self._backbone(inputs)

        x, mask = features[-1].decompose()
        batch_size, channels, height, width = x.shape

        x = self._reduce_dim(x)

        x = self._encoder(x, mask, pos[-1])
        encoder_output = x

        # shape of pos -> (batch_size, channels, height, width)
        #              -> (height*width, batch_size, channels)
        fine_pos = pos[-1].flatten(2).permute(2, 0, 1)
        fine_pos = fine_pos * self._encoder._pos_scale(
            x.flatten(2).permute(2, 0, 1).contiguous()
        )
        fine_pos = (
            fine_pos.view(height, width, batch_size, -1)
            .permute(2, 3, 0, 1)
            .contiguous()
        )
        fine_mask = mask

        selected_objects, selected_centers, det_output = self._mini_detector(
            x, fine_pos, mask
        )

        selected_objects_pos_embed = gen_sineembed_for_position(
            selected_centers, self._hidden_dim
        )

        x = self._decoder(
            selected_objects=selected_objects,
            encoder_output=encoder_output.flatten(2)
            .transpose(1, 2)
            .contiguous(),
            mask=fine_mask.flatten(1).contiguous(),
            fine_pos=fine_pos.flatten(2).transpose(1, 2).contiguous(),
            selected_objects_pos_embed=selected_objects_pos_embed,
            selected_centers=selected_centers,
            bbox_embed=self._bbox_embed,
        )

        cls_x, reg_x = torch.split(
            x, [self._hidden_dim, self._hidden_dim], dim=-1
        )
        center_offset_before_sigmoid = inverse_sigmoid(selected_centers)

        cls_output = self._cls_embed(cls_x)

        tmp = self._bbox_embed(reg_x)
        tmp[..., :2] += center_offset_before_sigmoid
        bbox_output = tmp.sigmoid()

        model_output = {"pred_class": cls_output, "pred_boxes": bbox_output}

        return model_output, det_output


class SingleShotDetector(nn.Module):
    def __init__(
        self, backbone: nn.Module, non_max_sup: nn.Module, num_class: int
    ):
        super().__init__()

        self._backbone = backbone
        self._feature_maps = self._build_feature_maps(
            embed_dim=[1024, 512, 512, 256, 256, 256],
            hidden_dim=[256, 256, 128, 128, 128],
        )
        self._detectors = self._build_detectors(
            input_dim=[1024, 512, 512, 256, 256, 256],
            default_boxes=[4, 6, 6, 6, 4, 4],
        )
        self._num_class = num_class  # exclude dummy class for background
        self._num_boxes = [4, 6, 6, 6, 4, 4]
        self._nmp = non_max_sup

    def _build_detectors(
        self, input_dim: list[int], default_boxes: list[int]
    ) -> nn.ModuleList:
        detectors = []

        for in_channel, num_boxes in zip(input_dim, default_boxes):
            bbox_embed = nn.Conv2d(
                in_channels=in_channel,
                out_channels=num_boxes * 4,
                kernel_size=3,
                padding="same",
            )
            conf_embed = nn.Conv2d(
                in_channels=in_channel,
                out_channels=num_boxes * (self._num_class + 1),
                kernel_size=3,
                padding="same",
            )
            detectors.append([bbox_embed, conf_embed])

        return nn.ModuleList(detectors)

    def _build_feature_maps(
        self, embed_dim: list[int], hidden_dim: list[int]
    ) -> nn.ModuleList:
        blocks = []
        for idx, (in_channels, out_channels, inter_channels) in enumerate(
            zip(embed_dim[:-1], embed_dim[1:], hidden_dim)
        ):
            if idx < 3:
                block = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=inter_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=inter_channels),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=inter_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=2,
                        padding="same",
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=inter_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(inter_channels),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=inter_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            blocks.append(block)

        return nn.ModuleList(blocks)

    def forward(self, inputs):
        x = self._backbone(inputs)

        features = [x]

        for block in self._feature_maps:
            x = block(x)
            features.append(x)

        outputs = []
        for ft, det, num_boxes in zip(
            features, self._detectors, self._num_boxes
        ):
            bs, _, h, w = ft.shape
            bbox_embed = (
                det[0](ft)
                .reshape(bs, num_boxes, -1, h, w)
                .permute(0, 3, 4, 1, 2)
                .contiguous()
            )
            conf_embed = (
                det[1](ft)
                .reshape(bs, num_boxes, -1, h, w)
                .permute(0, 3, 4, 1, 2)
                .contiguous()
            )

            # shape after transformed (batch_size, height, width, boxes, embed)
            # note that both height and width are not constant
            # for each boxes, there are coordinate (cx, cy, h, w)
            # and the rest are confidence score (scr1, scr2, ..., scrn)
            outputs.append(
                torch.cat([bbox_embed, conf_embed], -1).contiguous()
            )

        outputs = self.nmp(outputs)


def build_model(args):
    encoder = build_encoder(args=args)
    decoder = build_decoder(args=args)
    backbone = build_backbone_customized(args=args)

    model = ObjDetSplitTransformer(
        args=args, backbone=backbone, encoder=encoder, decoder=decoder
    )
    if args.resume:
        model.load_state_dict(
            torch.load(
                os.path.join(
                    "/workspace", "checkpoints", args.model_weight_name
                ),
                weights_only=True,
            )
        )

    return model
