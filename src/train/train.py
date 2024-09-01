import os
import argparse

import torch

from ..model.model import build_model


def train():
    pass


def test(model):
    t = torch.rand((2, 3, 224, 224))

    output = model(t)

    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # train config
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-3,
        dest="lr",
        help="Learning rate of the model except backbone",
    )
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=1e-4,
        dest="lr_backbone",
        help="Learning rate of backbone. If you want to freeze the backbone, set it to 0",
    )

    # model config
    parser.add_argument(
        "-num_enc",
        "--number_encoder_blocks",
        dest="num_encoder_blocks",
        type=int,
        default=6,
        help="Number of encoder blocks in Transformer",
    )
    parser.add_argument(
        "-num_dec",
        "--number_decoder_blocks",
        dest="num_decoder_blocks",
        type=int,
        default=6,
        help="Number of decoder blocks in Transformer",
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=300,
        dest="top_k",
        help="The number objects chosen in mini-detector",
    )
    parser.add_argument(
        "-cls",
        "--class_number",
        type=int,
        default=2,
        dest="num_cls",
        help="The number of classes to classify in images",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        dest="hidden_dim",
        help="The hidden dimension in the model",
    )

    args = parser.parse_args()
    model = build_model(args=args)
