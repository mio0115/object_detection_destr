import torch
from torch.nn import Module, Sequential, Conv2d, Linear, Sigmoid, ReLU

from ...utils.positional_embedding import gen_sineembed_for_position


class MiniDetector(Module):
    def __init__(
        self,
        input_shape: tuple[int, int],
        reg_ffn,
        cls_num: int,
        top_k: int,
        position_index_2d: torch.Tensor,
        hidden_dim: int = 256,
    ) -> None:
        super(MiniDetector, self).__init__()

        self._input_shape = input_shape
        self._top_k = top_k
        self._hidden_dim = hidden_dim
        self._pos_embed_2d = gen_sineembed_for_position(position_index_2d)

        self._cls_conv = Sequential(
            [
                Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=(3, 3),
                    stride=1,
                    padding="same",
                )
                for _ in range(4)
            ]
        )
        self._reg_conv = Sequential(
            [
                Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=(3, 3),
                    stride=1,
                    padding="same",
                )
                for _ in range(4)
            ]
        )
        self._pos_conv = Sequential(
            [
                Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=(3, 3),
                    stride=1,
                    padding="same",
                ), 
                ReLU() 
                for _ in range(3)
            ]
        )

        self._cls_head = Sequential(
            [Linear(in_features=hidden_dim, out_features=cls_num), Sigmoid()]
        )
        self._reg_head = reg_ffn
        self._pos_head = Sequential(
            [
                Linear(in_features=hidden_dim, out_features=hidden_dim),
                Linear(in_features=hidden_dim, out_features=2),
            ]
        )
    
    @property
    def input_shape(self):
        return self._input_shape
    
    @property
    def input_height(self):
        return self._input_shape[0]
    
    @property
    def input_width(self):
        return self._input_shape[1]
    
    @property
    def sequence_length(self):
        return self.input_height * self.input_width
    
    @property
    def embedding_dim(self):
        return self._hidden_dim

    def forward(self, inputs):
        batch_size = inputs.size(0)

        cls_x = inputs
        cls_x = self._cls_conv(cls_x)
        cls_x = cls_x.view(shape=(batch_size, self.sequence_length, self.embedding_dim))
        cls_features = cls_x
        cls_scores = self._cls_head(cls_x)

        pos_query = self._pos_embed_2d.view(shape=(batch_size, self.input_height, self.input_width, self.input_shape[-1]))
        pos_features = self._pos_conv(pos_query).view(shape=(batch_size, self.sequence_length, self.embedding_dim))

        pos_center_offset = self._pos_head(pos_features)
        pos_center_offset = torch.concat([[pos_center_offset, torch.zeros(size=(batch_size, self.sequence_length, 2), dtype=torch.float32)]])

        reg_x = inputs
        reg_x = self._reg_conv(reg_x)
        reg_x = reg_x.view(shape=(batch_size, self.sequence_length, self.embedding_dim))
        reg_features = reg_x
        bbox_coord = self._reg_head(reg_x) + pos_center_offset
        bbox_coord = Sigmoid()(bbox_coord)

        top_k = min(self._top_k, self.sequence_length)

        repr_cls_scores = torch.max(cls_scores, dim=-1)
        _, top_k_indices = torch.topk(repr_cls_scores, k=top_k)
        cls_output = torch.gather(cls_features, index=top_k_indices, dim=1)
        reg_output = torch.gather(reg_features, index=top_k_indices, dim=1)
        top_k_centers = torch.gather(bbox_coord, index=top_k_indices, dim=1)

        top_k_proposals = torch.concat([cls_output, reg_output], dim=-1)
        all_proposals = torch.concat([cls_scores, bbox_coord], dim=-1)

        return top_k_proposals, top_k_centers, all_proposals