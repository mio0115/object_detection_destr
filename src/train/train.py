import os
import argparse

import torch
import torch.utils
import torch.utils.data

from ..model.model import build_model


def train_process(
    args,
    model: torch.nn.Module,
    loss_fn,
    optimizer: torch.optim.optimizer,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
):
    for idx in range(args.epochs):
        model.train(True)
        loss_model, loss_det = train_one_epoch(
            model, loss_fn, optimizer=optimizer, dataloader=train_loader
        )

        running_vloss = {"model": 0.0, "det": 0.0}
        model.eval()

        with torch.no_grad():
            for vdata in valid_loader:
                vinputs, vlabels = vdata

                voutputs_model, voutputs_det = model(vinputs)
                vloss_model = loss_fn(voutputs_model, vlabels)
                vloss_det = loss_fn(voutputs_det, vlabels)

                running_vloss["model"] += vloss_model
                running_vloss["det"] += vloss_det

            vloss_model = running_vloss["model"] / len(valid_loader)
            vloss_det = running_vloss["det"] / len(valid_loader)

        print(
            f"Epoch {idx+1:>2}: \n\t Train loss model: {loss_model:.4f} detetector: {loss_det:.4f}\n\t Valid loss model: {vloss_model:.4f} detector: {loss_det:.4f}"
        )


def train_one_epoch(
    model,
    loss_fn,
    optimizer: torch.optim.optimizer,
    dataloader: torch.utils.data.DataLoader,
):
    running_loss_det = 0.0
    running_loss_model = 0.0

    for data in dataloader:
        inputs, labels = data

        optimizer.zero_grad()
        model_outputs, det_outputs = model(inputs)

        loss_model = loss_fn(model_outputs, labels)
        loss_det = loss_fn(det_outputs, labels)

        loss_model.backward()
        loss_det.backward()

        optimizer.step()

        running_loss_model += loss_model.item()
        running_loss_det += loss_det.item()

    train_loss_model = running_loss_model / len(dataloader)
    train_loss_det = running_loss_det / len(dataloader)

    return train_loss_model, train_loss_det


def test(model):
    t = torch.rand((2, 3, 720, 1280))

    cls_output, bbox_output, det_output = model(t)

    breakpoint()


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
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        dest="epochs",
        help="Number of training epochs",
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

    test(model)
