import math

import torch


def gen_sineembed_for_position(pos_tensor: torch.Tensor, d_model: int = 512):
    """
    Positional embedding of pos_tensor with depth d_model.
    Copied from destr (https://github.com/helq2612/destr).

    Args:
        pos_tensor: The tensor of coordinates to get positional embedding.
        d_model   : The depth of pos_tensor. (tf.shape(pos_tensor)[-1])

    Returns:
        torch.Tensor: A tensor which is offset of pos_tensor.
    """
    scale = 2 * math.pi
    hd_model = d_model // 2
    dim_t = torch.arange(
        start=0, end=hd_model, dtype=torch.float32, device=pos_tensor.device
    )
    dim_t = 10000 ** (2 * (dim_t // 2) / hd_model)

    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale

    pos_x = x_embed.unsqueeze(2) / dim_t
    pos_y = y_embed.unsqueeze(2) / dim_t

    pos_x = torch.stack(
        [pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1
    ).flatten(2)
    pos_y = torch.stack(
        [pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1
    ).flatten(2)
    pos = torch.concat([pos_y, pos_x], dim=2)

    return pos


def with_position_embedding(pos_tensor: torch.Tensor, d_model: int = 512):
    return pos_tensor + gen_sineembed_for_position(pos_tensor, d_model)


if __name__ == "__main__":
    t = torch.arange(start=0, end=2 * 7 * 2).view(2, 7, 2)
    print(gen_sineembed_for_position(pos_tensor=t))
