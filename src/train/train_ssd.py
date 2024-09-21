import os
import argparse
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from .arg_parser import get_parser
from ..model.model_ssd import build_model
from ..dataset.transforms import TransformTypes, build_transform_ssd
from ..dataset.dataset import VOCDetection, voc_collate_fn
from ..utils.misc import to_device, resume
from ..utils.matcher import build_matcher, SimpleMatcher
from ..utils.criterion import (
    SSDCriterion,
    SSDClassCriterion,
    SSDLocalCriterion,
    MeanAveragePrecision,
)


def train(
    args,
    model: torch.nn.Module,
    criterion,
    # metric,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
):
    writer = SummaryWriter()

    model = to_device(model, args.device)

    lowest_vloss, g_step, g_vstep, log_interval = 10000, 0, 0, 100
    for idx in range(args.epochs):
        model.train()
        loss, duration, g_step = train_one_epoch(
            args,
            model,
            criterion,
            optimizer=optimizer,
            dataloader=train_loader,
            writer=writer,
            g_step=g_step,
        )

        running_vloss, prefix_vloss = 0.0, 0.0
        model.eval()

        with torch.no_grad():
            for vdata in valid_loader:
                vinputs, vtargets = to_device(vdata, args.device)

                voutputs = model(vinputs)
                # metric(voutputs_model, vtargets)
                vloss = criterion(voutputs, vtargets)

                running_vloss += vloss.item() * vinputs.size(0)

                g_vstep += 1
                if g_vstep % log_interval == 0:
                    avg_vloss = (running_vloss - prefix_vloss) / (
                        log_interval * args.batch_size
                    )

                    writer.add_scalar("Loss/valid", avg_vloss, g_vstep)

                    prefix_vloss = running_vloss

            # writer.add_scalar("Metric/mAP", metric.compute(), idx)
            vloss = running_vloss / len(valid_loader.dataset)

            # metric.reset()

        if vloss < lowest_vloss:
            torch.save(
                model.state_dict(),
                os.path.join("/", "workspace", "checkpoints", args.save_as),
            )
            lowest_vloss = vloss

        print(
            f"""Epoch {idx+1:>2}: \n\t 
                Duration: {duration/60:.4f} minutes \n\t
                Train Loss \n\t\t 
                    model: {loss:.4f}\n\t 
                Valid Loss \n\t\t 
                    model: {vloss:.4f}"""
        )
        writer.close()


def train_one_epoch(
    args,
    model,
    criterion,
    writer,
    g_step,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
):
    running_loss, prefix_loss = 0.0, 0.0
    log_interval = 100

    start_time = time.time()

    for data in dataloader:
        inputs, targets = to_device(data, args.device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)

        loss.backward()
        optimizer.step()

        g_step += 1
        if g_step % log_interval == 0:
            avg_loss = (running_loss - prefix_loss) / (log_interval * args.batch_size)

            writer.add_scalar("Loss/train/loss", avg_loss, g_step)
            # writer.add_scalar("Loss/train/class", avg_loss_cls, g_step)
            # writer.add_scalar("Loss/train/ciou", avg_loss_ciou, g_step)

            prefix_loss = running_loss

            # prefix_loss_class, prefix_loss_ciou = (
            #    running_loss_class,
            #   running_loss_ciou,
            # )

    train_loss = running_loss / len(dataloader.dataset)
    end_time = time.time()

    return train_loss, end_time - start_time, g_step


def test_model(model):
    """Test if the model is available"""
    t = torch.rand((2, 3, 720, 1280))

    model(t)


if __name__ == "__main__":
    parser = get_parser("ssd")
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
    if args.resume:
        model, optim = resume(model=model, optim=optim, args=args)

    matcher = build_matcher(SimpleMatcher, args)
    criterion = SSDCriterion(
        matcher=matcher,
        loss_fns={
            "class": SSDClassCriterion(),
            "local": SSDLocalCriterion(args),
        },
        loss_coef=args.coef_class_loss,
    ).to(args.device)
    # metric = MeanAveragePrecision().to("cpu")

    path_to_dataset = os.path.join(os.getcwd(), "dataset")
    train_ds = VOCDetection(
        root=path_to_dataset,
        image_set=TransformTypes.TRAIN,
        transform=build_transform_ssd(trans_type=TransformTypes.TRAIN),
        augment_factor=5,
    )
    valid_ds = VOCDetection(
        root=path_to_dataset,
        image_set=TransformTypes.VALID,
        transform=build_transform_ssd(trans_type=TransformTypes.VALID),
        augment_factor=1,
    )

    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=voc_collate_fn,
    )
    valid_dl = torch.utils.data.DataLoader(
        dataset=valid_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=voc_collate_fn,
    )

    train(
        args,
        model=model,
        criterion=criterion,
        # metric=metric,
        optimizer=optim,
        train_loader=train_dl,
        valid_loader=valid_dl,
    )
