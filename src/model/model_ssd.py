import torch
from torch import nn
from torchvision import models as tv_models


class SingleShotDetector(nn.Module):
    def __init__(self, backbone: nn.Module, num_class: int):
        super().__init__()

        self._num_class = num_class + 1  # exclude dummy class for background
        self._num_boxes = [4, 6, 6, 6, 4, 4]

        self._backbone = backbone
        self._feature_maps = self._build_feature_maps(
            embed_dim=[512, 1024, 512, 256, 256, 256],
            hidden_dim=[1024, 256, 128, 128, 128],
        )
        self._build_detectors(
            input_dim=[512, 1024, 512, 256, 256, 256],
            default_boxes=self._num_boxes,
        )

    def _build_detectors(self, input_dim: list[int], default_boxes: list[int]) -> None:
        bbox_detectors, conf_detectors = [], []

        for in_channel, num_boxes in zip(input_dim, default_boxes):
            bbox_embed = nn.Conv2d(
                in_channels=in_channel,
                out_channels=num_boxes * 4,
                kernel_size=3,
                padding=1,
            )
            conf_embed = nn.Conv2d(
                in_channels=in_channel,
                out_channels=num_boxes * (self._num_class + 1),
                kernel_size=3,
                padding=1,
            )
            bbox_detectors.append(bbox_embed)
            conf_detectors.append(conf_embed)

        self._detectors = nn.ModuleDict(
            {
                "boxes": nn.ModuleList(bbox_detectors),
                "conf": nn.ModuleList(conf_detectors),
            }
        )

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
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=inter_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
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
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=inter_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            blocks.append(block)

        return nn.ModuleList(blocks)

    def forward(self, inputs: torch.Tensor) -> dict[str : list[torch.Tensor]]:
        x = self._backbone(inputs)

        features = [x]

        for block in self._feature_maps:
            x = block(x)
            features.append(x)

        outputs = {"boxes": [], "conf": []}
        for ft, box_det, conf_det, num_boxes in zip(
            features, self._detectors["boxes"], self._detectors["conf"], self._num_boxes
        ):
            bs, _, h, w = ft.shape
            bbox_embed = (
                box_det(ft)
                .reshape(bs, num_boxes, -1, h, w)
                .permute(0, 3, 4, 1, 2)
                .contiguous()
            )
            conf_embed = (
                conf_det(ft)
                .reshape(bs, num_boxes, -1, h, w)
                .permute(0, 3, 4, 1, 2)
                .contiguous()
            )

            # shape after transformed (batch_size, height, width, boxes, embed)
            # note that both height and width are not constant
            # for each boxes, there are coordinate (cx, cy, h, w)
            # and the rest are confidence score (scr1, scr2, ..., scrn)
            outputs["boxes"].append(bbox_embed)
            outputs["conf"].append(conf_embed)

        return outputs


class Backbone(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        vgg16_model = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
        self._layers = nn.ModuleList(list(vgg16_model.features)[:23])

    def forward(self, inputs):
        x = inputs
        for layer in self._layers:
            x = layer(x)

        return x


def build_model(args):
    backbone = Backbone()

    return SingleShotDetector(backbone=backbone, num_class=args.num_cls)
