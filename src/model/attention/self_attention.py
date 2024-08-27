import torch
from torch.nn import Module, Linear


class SelfAttention(Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_shape: tuple[int, int, int],
        heads_num: int = 8,
    ):
        super(SelfAttention, self).__init__()

        self._heads_num = heads_num
        # self._input_shape = input_shape
        # self._output_shape = output_shape

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        if self._heads_num > 1:
            batch_size = query.size(dim=0)
        else:
            batch_size = query.size(dim=1)

        attn_sc = torch.matmul(query, key.transpose(3, 2)) / torch.sqrt(query.size(-1))
        attn_sc = torch.softmax(input=attn_sc, dim=-1)
        output = torch.matmul(attn_sc, value)

        output = output.transpose(1, 2).view(
            batch_size, -1, self._heads_num * value.size(-1)
        )

        return output
