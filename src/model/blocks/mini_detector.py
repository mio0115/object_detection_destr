from typing import Optional

import torch
from torch import nn


class MiniDetector(nn.Module):
    def __init__(
        self,
        reg_ffn: nn.Module,
        class_embed: nn.Module,
        bbox_embed: nn.Module,
        top_k: int,
        hidden_dim: int = 256,
    ) -> None:
        super(MiniDetector, self).__init__()

        self._top_k = top_k
        self._hidden_dim = hidden_dim

        self._cls_conv = nn.ModuleList()
        for _ in range(4):
            self._cls_conv.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=(3, 3),
                    stride=1,
                    padding="same",
                )
            )

        self._reg_conv = nn.ModuleList()
        for _ in range(4):
            self._reg_conv.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=(3, 3),
                    stride=1,
                    padding="same",
                )
            )

        self._pos_conv = nn.ModuleList()
        for _ in range(4):
            self._pos_conv.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=(3, 3),
                    stride=1,
                    padding="same",
                )
            )

        self._cls_embed = class_embed
        self._pos_head = reg_ffn
        self._bbox_embed = bbox_embed

    def mask_invalid_features(self, tensor, mask, padding_value=float(0)):
        """We mask invalid(padded) features in tensor by 0"""
        tensor = tensor.masked_fill(mask.flatten(1).unsqueeze(-1), padding_value)

        return tensor

    def get_topk_index(
        self,
        scores: torch.Tensor,
        k: int,
        padding_mask: Optional[torch.Tensor],
    ):
        batch_size, num_objects, _ = scores.shape

        cls_scores = scores.sigmoid()
        max_cls_scores, _ = cls_scores.max(dim=-1, keepdim=False)

        _, topk_idx = torch.topk(max_cls_scores, k=k, dim=1, largest=True)
        if padding_mask is not None:
            # count number of valid features
            valid_nums = (1 - padding_mask.to(dtype=torch.float32)).sum(dim=-1)
            new_topk = []

            for idx, valid in zip(topk_idx, valid_nums):
                if valid > num_objects:
                    new_topk.append(idx)
                    continue
                tmp = torch.flip(idx[: valid.int()], dims=(0,)).repeat(
                    k // valid.int() + 1
                )
                tmp = tmp[:k]
                tmp = torch.concat((idx[: valid.int()], tmp[valid.int() :]), dim=0)
                new_topk.append(tmp)
            topk_idx = torch.stack(new_topk, dim=0)
        batch_idx = [(torch.ones(k) * idx).int() for idx in range(batch_size)]
        batch_idx = torch.concat(batch_idx)

        idx = topk_idx.flatten()
        selected_idx = (batch_idx, idx)

        return selected_idx

    def forward(self, inputs, pos_embed, mask):
        batch_size, _, height, width = inputs.shape

        cls_x = inputs
        for conv in self._cls_conv:
            cls_x = conv(cls_x)  # bs, ch, h, w
        cls_x = cls_x.flatten(2).transpose(1, 2).contiguous()  # bs, h*w, ch
        cls_x = self.mask_invalid_features(cls_x, mask=mask)

        cls_features = cls_x
        det_output_class = self._cls_embed(cls_x)

        pos_query = pos_embed
        for conv in self._pos_conv:
            pos_query = conv(pos_query)
        pos_query = pos_query.flatten(2).transpose(1, 2).contiguous()
        pos_query = self.mask_invalid_features(pos_query, mask=mask)

        bbox_center_offset = self._pos_head(pos_query)

        reg_x = inputs
        for conv in self._reg_conv:
            reg_x = conv(reg_x)  # bs, channels, h, w
        reg_x = reg_x.flatten(2).transpose(1, 2).contiguous()  # reg_x should be 4
        reg_x = self.mask_invalid_features(reg_x, mask=mask)

        reg_features = reg_x
        bbox_coord = self._bbox_embed(reg_x)
        bbox_coord[..., :2] += bbox_center_offset
        det_output_coord = bbox_coord.sigmoid()

        det_output = {
            "pred_class": torch.clone(det_output_class),
            "pred_boxes": torch.clone(det_output_coord),
        }

        object_features = torch.concat(
            [cls_features, reg_features], dim=-1
        ).contiguous()

        det_output_coord = self.mask_invalid_features(
            det_output_coord, mask=mask
        ).reshape(batch_size, height * width, 4)
        det_output_class = self.mask_invalid_features(
            det_output_class.sigmoid(), mask=mask
        ).reshape(batch_size, height * width, -1)

        valid_cand_per_img = mask.flatten(1).eq(0).sum(dim=-1, keepdim=False)[0].min()
        avail_k = min(self._top_k, height * width, valid_cand_per_img)

        idx = self.get_topk_index(
            det_output_class,  # shape -> (bs, h*w, num_cls)
            k=avail_k,
            padding_mask=mask.flatten(1),
        )

        selected_objects = (
            object_features[idx].reshape(batch_size, avail_k, -1).contiguous().detach()
        )
        selected_objects_center = (
            det_output_coord[..., :2][idx]
            .reshape(batch_size, avail_k, -1)
            .contiguous()
            .detach()
        )

        return selected_objects, selected_objects_center, det_output
