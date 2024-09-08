import os
import argparse
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from ..model.model import build_model
from ..dataset.transforms import TransformTypes, build_transform
from ..dataset.dataset import WiderFace, widerface_collate_fn
from ..utils.misc import to_device
from ..utils.matcher import build_matcher
from ..utils.criterion import SetCriterion, CompleteIOULoss


def train(
    args,
    model: torch.nn.Module,
    criterion,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
):
    writer = SummaryWriter()

    model = to_device(model, args.device)

    lowest_vloss, g_step, g_vstep, log_interval = 10000, 0, 0, 100
    for idx in range(args.epochs):
        model.train()
        loss_model, loss_det, duration, g_step = train_one_epoch(
            args,
            model,
            criterion,
            optimizer=optimizer,
            dataloader=train_loader,
            writer=writer,
            g_step=g_step,
        )

        running_vloss_model, running_vloss_det = 0.0, 0.0
        prefix_vloss_model, prefix_vloss_det = 0, 0
        model.eval()

        with torch.no_grad():
            for vdata in valid_loader:
                vinputs, vtargets = to_device(vdata, args.device)

                voutputs_model, voutputs_det = model(vinputs)
                vloss_model = criterion(voutputs_model, vtargets)
                vloss_det = criterion(voutputs_det, vtargets)

                running_vloss_model += vloss_model.item()
                running_vloss_det += vloss_det.item()

                g_vstep += 1
                if g_vstep % log_interval == 0:
                    avg_vloss_model = (
                        running_vloss_model - prefix_vloss_model
                    ) / log_interval
                    avg_vloss_det = (
                        running_vloss_det - prefix_vloss_det
                    ) / log_interval

                    writer.add_scalar("Loss/valid/model", avg_vloss_model, g_vstep)
                    writer.add_scalar("Loss/valid/det", avg_vloss_det, g_vstep)

            vloss_model = running_vloss_model / len(valid_loader.dataset)
            vloss_det = running_vloss_det / len(valid_loader.dataset)

        vloss = vloss_model * 0.7 + vloss_det * 0.3

        if vloss < lowest_vloss:
            torch.save(
                model.state_dict(),
                os.path.join("/", "workspace", "checkpoints", args.save_as),
            )

        print(
            f"""Epoch {idx+1:>2}: \n\t 
                Duration: {duration/60:.4f} minutes \n\t
                Train Loss \n\t\t 
                    model: {loss_model:.4f} detector: {loss_det:.4f}\n\t 
                Valid Loss \n\t\t 
                    model: {vloss_model:.4f} detector: {loss_det:.4f}"""
        )


def train_one_epoch(
    args,
    model,
    criterion,
    writer,
    g_step,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
):
    running_loss_det, running_loss_model = 0.0, 0.0
    log_interval = 100
    prefix_loss_model, prefix_loss_det = 0, 0
    start_time = time.time()

    for data in dataloader:
        inputs, targets = to_device(data, args.device)

        optimizer.zero_grad()
        model_outputs, det_outputs = model(inputs)

        losses_model = criterion(model_outputs, targets)
        losses_det = criterion(det_outputs, targets)

        total_loss = losses_model * 0.7 + losses_det * 0.3

        total_loss.backward()
        optimizer.step()

        running_loss_model += losses_model.item()
        running_loss_det += losses_det.item()

        g_step += 1
        if g_step % log_interval == 0:
            avg_loss_model = (running_loss_model - prefix_loss_model) / log_interval
            avg_loss_det = (running_loss_det - prefix_loss_det) / log_interval

            writer.add_scalar("Loss/train/model", avg_loss_model, g_step)
            writer.add_scalar("Loss/train/det", avg_loss_det, g_step)

    train_loss_model = running_loss_model / len(dataloader.dataset)
    train_loss_det = running_loss_det / len(dataloader.dataset)
    end_time = time.time()

    return train_loss_model, train_loss_det, end_time - start_time, g_step


def test_model(model):
    """Test if the model is available"""
    t = torch.rand((2, 3, 720, 1280))

    model(t)


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
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=4,
        type=int,
        dest="batch_size",
        help="Number of samples in batch",
    )
    parser.add_argument(
        "--augment_factor",
        default=5,
        type=int,
        dest="augment_factor",
        help="New size of dataset after data augmentation",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--model_weight_name",
        default="model_weights.pth",
        type=str,
        help="Name of model weights to restore",
    )
    parser.add_argument(
        "--device", default="cuda", dest="device", help="Device to use for training"
    )
    parser.add_argument(
        "--save_as",
        default="model_weights.pth",
        type=str,
        help="Name of model weights to save",
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
    optim = torch.optim.AdamW(
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
            "class": torch.nn.BCEWithLogitsLoss(),
            "bbox": torch.nn.L1Loss(),
            "ciou": CompleteIOULoss(),
        },
    ).to(args.device)

    path_to_dataset = os.path.join(os.getcwd(), "dataset")
    train_ds = WiderFace(
        root=path_to_dataset,
        split=TransformTypes.TRAIN,
        transform=build_transform(trans_type=TransformTypes.TRAIN),
        augment_factor=5,
    )
    valid_ds = WiderFace(
        root=path_to_dataset,
        split=TransformTypes.VALID,
        transform=build_transform(trans_type=TransformTypes.VALID),
        augment_factor=1,
    )

    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=widerface_collate_fn,
    )
    valid_dl = torch.utils.data.DataLoader(
        dataset=valid_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=widerface_collate_fn,
    )

    train(
        args,
        model=model,
        criterion=criterion,
        optimizer=optim,
        train_loader=train_dl,
        valid_loader=valid_dl,
    )
