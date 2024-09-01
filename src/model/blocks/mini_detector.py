import torch
from torch import nn

from ...utils.positional_embedding import gen_sineembed_for_position


class MiniDetector(nn.Module):
    def __init__(
        self,
        reg_ffn: nn.Module,
        class_embed: nn.Module,
        bbox_embed: nn.Module,
        top_k: int,
        # position_index_2d: torch.Tensor,
        hidden_dim: int = 256,
    ) -> None:
        super(MiniDetector, self).__init__()

        self._top_k = top_k
        self._hidden_dim = hidden_dim

        self._cls_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=(3, 3),
                    stride=1,
                    padding="same",
                )
                for _ in range(4)
            ]
        )
        self._reg_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=(3, 3),
                    stride=1,
                    padding="same",
                )
                for _ in range(4)
            ]
        )
        self._pos_conv = nn.ModuleList()
        for _ in range(3):
            self._pos_conv.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=(3, 3),
                    stride=1,
                    padding="same",
                )
            )
            self._pos_conv.append(nn.ReLU())

        self._cls_embed = class_embed
        self._pos_head = reg_ffn
        self._bbox_embed = bbox_embed

    def remove_zero_padding(self, tensor, mask, height, width, padding_value=float(0)):
        batch_size, sequence_len, channels = tensor.shape

    def get_topk_index(self):
        pass

    def forward(self, inputs, pos_embed, mask):
        batch_size, channels, height, width = inputs.shape

        cls_x = inputs
        for conv in self._cls_conv:
            cls_x = conv(cls_x)  # bs, ch, h, w
        cls_x = cls_x.flatten(2).permute(0, 2, 1).contiguous()  # bs, h*w, ch
        cls_x = self.remove_zero_padding(cls_x, mask=mask, height=height, width=width)

        cls_features = cls_x
        det_output_class = self._cls_embed(cls_x)

        pos_query = pos_embed
        for conv in self._pos_conv:
            pos_query = conv(pos_query)
        pos_query = pos_query.flatten(2).permute(0, 2, 1).contiguous()
        pos_query = self.remove_zero_padding(
            pos_query, mask=mask, height=height, width=width
        )

        bbox_center_offset = self._pos_head(pos_query)

        reg_x = inputs
        for conv in self._reg_conv:
            reg_x = conv(reg_x)  # bs, channels, h, w
        reg_x = reg_x.flatten(2).permute(0, 2, 1).contiguous()  # reg_x should be 4
        reg_x = self.remove_zero_padding(reg_x, mask=mask, height=height, width=width)

        reg_features = reg_x
        bbox_coord = self._bbox_embed(reg_x)
        bbox_coord[..., :2] += bbox_center_offset
        det_output_coord = bbox_coord.sigmoid()

        det_output = {
            "pred_class": torch.clone(det_output_class),
            "pred_boxes": torch.clone(det_output_coord),
        }

        object_features = torch.concat([cls_features, reg_features], dim=-1)
        object_features = object_features.permute(1, 0, 2).contiguous()

        det_output_coord = self.remove_zero_padding(
            det_output_coord, mask=mask, height=height, width=width
        ).reshape(batch_size, height * width, 4)
        det_output_class = self.remove_zero_padding(
            det_output_class.sigmoid(), mask=mask, height=height, width=width
        ).reshape(batch_size, height * width, -1)

        valid_cand_per_img = mask.flatten(1).eq(0).sum(dim=-1, keepdim=False)[0].min()
        avail_k = min(self._top_k, height * width, valid_cand_per_img)

        idx = self.get_topk_index(
            det_output_class,
            k=avail_k,
            padding_mask=mask.flatten(1),
            training=self.training,
        )

        # TODO: check if the reshape is needed; use transpose instead of permute
        selected_objects = (
            object_features[idx]
            .reshape(batch_size, avail_k, -1)
            .permute(1, 0, 2)
            .contiguous()
            .detach()
        )
        selected_objects_center = (
            det_output_coord[..., :2]
            .transpose(0, 1)[idx]
            .reshape(batch_size, avail_k, -1)
            .permute(1, 0, 2)
            .detach()
        )

        return selected_objects, selected_objects_center, det_output
