import argparse


def get_parser(model_name: str):
    match model_name:
        case "ssd" | "SSD":
            return get_parser_ssd()
        case "destr" | "DESTR":
            return get_parser_destr()
        case _:
            raise KeyError(f"no parser for {model_name=}")


def get_parser_destr():
    parser = argparse.ArgumentParser()

    # train config
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-5,
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
        default=0.5,
        type=float,
        dest="set_cost_class",
        help="Weight of class lost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=0,
        type=float,
        dest="set_cost_bbox",
        help="Weight of bbox lost",
    )
    parser.add_argument(
        "--set_cost_ciou",
        default=0.5,
        type=float,
        dest="set_cost_ciou",
        help="Weight of ciou lost",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=12,
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
        "--resume_from",
        default="model_weights.pth",
        type=str,
        help="Name of model weights to resume",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        dest="device",
        help="Device to use for training",
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
    return parser


def get_parser_ssd():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-5,
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
        "--coef_class_loss",
        default=0.5,
        type=float,
        dest="coef_class_loss",
        help="Weight of class lost",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=12,
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
        "--resume_from",
        default="model_weights.pth",
        type=str,
        help="Name of model weights to resume",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        dest="device",
        help="Device to use for training",
    )
    parser.add_argument(
        "--save_as",
        default="model_weights.pth",
        type=str,
        help="Name of model weights to save",
    )

    # model config
    parser.add_argument(
        "-cls",
        "--class_number",
        type=int,
        default=20,
        dest="num_cls",
        help="The number of classes to classify in images",
    )
    parser.add_argument("--scale_min", type=float, default=0.2, dest="scale_min")
    parser.add_argument("--scale_max", type=float, default=0.9, dest="scale_max")

    return parser
