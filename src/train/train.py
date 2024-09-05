import os
import argparse

import torch
import torchvision as tv

from ..model.model import build_model
from ..utils.transforms import TransformTypes, build_transform
from ..utils.matcher import build_matcher
from ..utils.criterion import SetCriterion
from ..utils.bbox_utils import complete_iou


def train(
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
    parser.add_argument(
        "--path_to_dataset",
        type=str,
        default="/workspace/dataset",
        dest="path_to_dataset",
        help="Path to dataset",
    )
    parser.add_argument(
        "--set_cost_class",
        default=0.2,
        type=float,
        dest="set_cost_class",
        help="Weight of class lost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=0.5,
        type=float,
        dest="set_cost_bbox",
        help="Weight of bbox lost",
    )
    parser.add_argument(
        "--set_cost_ciou",
        default=0.2,
        type=float,
        dest="set_cost_ciou",
        help="Weight of ciou lost",
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

    other_params = [
        param for name, param in model.named_parameters() if "backbone" not in name
    ]
    optim = torch.optim.Adam(
        [
            {"params": model._backbone.parameters(), "lr": args.lr_backbone},
            {"params": other_params},
        ],
        lr=args.lr,
    )

    matcher = build_matcher(args)
    criterion = SetCriterion(
        num_classes=args.num_cls,
        matcher=matcher,
        loss_weight={
            "class": args.set_cost_class,
            "bbox": args.set_cost_bbox,
            "ciou": args.set_cost_ciou,
        },
        loss_fn={
            "class": torch.nn.CrossEntropyLoss(),
            "bbox": torch.nn.L1Loss(),
            "ciou": complete_iou,
        },
    )

    cwd = os.getcwd()
    train_ds = tv.datasets.wrap_dataset_for_transforms_v2(
        tv.datasets.WIDERFace(
            root=os.path.join(cwd, "dataset"),
            split=TransformTypes.TRAIN.value,
            transform=build_transform(trans_type=TransformTypes.TRAIN),
        ),
        target_keys=["bbox"],
    )
    valid_ds = tv.datasets.wrap_dataset_for_transforms_v2(
        tv.datasets.WIDERFace(
            root=os.path.join(cwd, "dataset"),
            split=TransformTypes.VALID.value,
            transform=build_transform(trans_type=TransformTypes.VALID),
        ),
        target_keys=["bbox"],
    )

    train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=4, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(dataset=valid_ds, batch_size=4, shuffle=True)

    train(model=model, optimizer=optim, train_loader=train_dl, valid_loader=valid_dl)
