import gc
import os
import random
import argparse

import torch
import numpy as np

from experiments.config import EXPERIMENTS
from experiments.utils import (
    DATASET_PATH,
    check_and_retrieve_vocabulary,
    download_and_extract_dataset,
)
from experiments.types import train_model, test_model, finetune_model
from my_utils.encoding_convertions import ENCODING_OPTIONS


# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Supervised training arguments.")
    parser.add_argument(
        "--experiment_id",
        type=int,
        choices=list(EXPERIMENTS.keys()),
        help="Experiment ID (see experiments/config.py)",
        required=True,
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="kern",
        choices=ENCODING_OPTIONS,
        help="Encoding type to use",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Number of epochs with no improvement after which training will be stopped",
    )
    args = parser.parse_args()
    return args


def main():
    gc.collect()
    torch.cuda.empty_cache()

    # Run on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse arguments
    args = parse_arguments()
    print("#### ---- SUPERVISED TRAINING EXPERIMENT ---- ####")
    print("Arguments:")
    for k, v in args.__dict__.items():
        print(f"\t{k}: {v}")
    print(f"\tdevice: {device}")

    # Check if the dataset is downloaded
    if not os.path.exists(DATASET_PATH):
        download_and_extract_dataset()

    # Retrieve vocabulary
    w2i, i2w = check_and_retrieve_vocabulary(
        sax_type=EXPERIMENTS[args.experiment_id]["sax_type"], encoding=args.encoding
    )

    # Train from scratch
    if EXPERIMENTS[args.experiment_id]["from_experiment"] is None:
        print("Training from scratch")
        train_model(
            experiment_id=args.experiment_id,
            encoding=args.encoding,
            w2i=w2i,
            i2w=i2w,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )

    # Test an already trained model
    elif EXPERIMENTS[args.experiment_id]["finetune"] is None:
        print("Testing an already trained model")
        test_model(
            experiment_id=args.experiment_id,
            encoding=args.encoding,
            w2i=w2i,
            i2w=i2w,
            device=device,
        )

    # Finetune an already trained model
    else:
        print("Finetuning an already trained model")
        finetune_model(
            experiment_id=args.experiment_id,
            encoding=args.encoding,
            w2i=w2i,
            i2w=i2w,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )

    pass


if __name__ == "__main__":
    main()
