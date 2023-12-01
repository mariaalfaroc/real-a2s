import os
import gc
import random
import argparse

import torch
import numpy as np

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import config
from utils import str2bool
from model import CTCTrainedCRNN
from encoding_convertions import ENCODING_OPTIONS
from data import (
    train_data_generator,
    load_data_from_files,
    check_and_retrieveVocabulary_from_files,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Supervised training arguments.")
    parser.add_argument(
        "--use_multirest",
        type=str2bool,
        default="False",
        help="Whether to use samples that contain multirest",
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
    parser.add_argument("--train", type=str, required=True, help="Train data partition")
    parser.add_argument(
        "--val", type=str, required=True, help="Validation data partition"
    )
    parser.add_argument("--test", type=str, required=True, help="Test data partition")
    parser.add_argument(
        "--trainmodel",
        type=str2bool,
        default="True",
        help="Whether to initially train the model",
    )
    parser.add_argument(
        "--finetune",
        type=str2bool,
        default="False",
        help="Whether to finetune the model",
    )
    parser.add_argument(
        "--updateFT",
        type=str,
        required=False,
        help="Layers to update when freezing the model",
        default="ALL",
    )
    args = parser.parse_args()

    args.updateFT = [item for item in args.updateFT.split(",")]

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

    # Data globals
    config.set_source_data_dirs()
    print(f"Data used {config.source_dir.stem}")
    case = args.train.split("_")[0]
    output_dir = config.output_dir / f"{case}_{args.encoding}"
    os.makedirs(output_dir, exist_ok=True)
    nameOfVoc = "Vocab"
    nameOfVoc = nameOfVoc + "_woutmultirest" if not args.use_multirest else nameOfVoc
    nameOfVoc = nameOfVoc + "_" + args.train.split("_")[0]
    nameOfVoc = nameOfVoc + "_" + args.encoding

    # Set filepaths outputs
    multirest_appedix = "_withmultirest" if args.use_multirest else ""
    model_filepath = output_dir / f"model{multirest_appedix}.pt"
    logs_path = output_dir / f"results{multirest_appedix}.csv"

    # Data
    (
        XFTrain,
        YFTrain,
        XFVal,
        YFVal,
        XFTest,
        YFTest,
        XTrain_FT,
        YTrain_FT,
    ) = load_data_from_files(
        config.cases_dir / args.train,
        config.cases_dir / args.val,
        config.cases_dir / args.test,
        args.use_multirest,
    )
    w2i, i2w = check_and_retrieveVocabulary_from_files(
        nameOfVoc=nameOfVoc,
        use_multirest=args.use_multirest,
        encoding=args.encoding,
        YTrain=YFTrain,
        YVal=YFVal,
        YTest=YFTest,
        YTrain_FT=YTrain_FT,
    )

    # Model
    model = CTCTrainedCRNN(
        dictionaries=(w2i, i2w), encoding=args.encoding, device=device
    )

    # Pretrain:
    if args.trainmodel == True:
        # Train, validate, and test
        model.fit(
            train_data_generator(
                XFiles=XFTrain,
                YFiles=YFTrain,
                batch_size=args.batch_size,
                width_reduction=model.model.encoder.width_reduction,
                w2i=w2i,
                device=device,
                encoding=args.encoding,
            ),
            epochs=args.epochs,
            steps_per_epoch=len(XFTrain) // args.batch_size,
            val_data=(XFVal, YFVal),
            test_data=(XFTest, YFTest),
            batch_size=args.batch_size,
            patience=args.patience,
            weights_path=model_filepath,
            logs_path=logs_path,
        )

    # Fine tune the model:
    if args.finetune == True:
        # Checking that pretrained model exists:
        assert os.path.exists(model_filepath), "Model does not exist"
        model.load(model_filepath)
        print(f"Loaded pretrained model from {model_filepath}")

        # Filepaths globals
        case = args.train.split("_")[0]
        output_dir = config.output_dir / f"{case}_{args.encoding}"
        os.makedirs(output_dir, exist_ok=True)
        nameOfVoc = "Vocab"
        nameOfVoc = (
            nameOfVoc + "_woutmultirest" if not args.use_multirest else nameOfVoc
        )
        nameOfVoc = nameOfVoc + "_" + args.train.split("_")[0]
        nameOfVoc = nameOfVoc + "_" + args.encoding
        multirest_appedix = "_withmultirest" if args.use_multirest else ""
        updateFT_appendix = "_update" + "".join([u.capitalize() for u in args.updateFT])
        model_filepath = output_dir / f"model{multirest_appedix}{updateFT_appendix}.pt"
        logs_path = output_dir / f"results{multirest_appedix}{updateFT_appendix}.csv"

        # Loading data
        (
            XFTrain,
            YFTrain,
            XFVal,
            YFVal,
            XFTest,
            YFTest,
            XTrain_FT,
            YTrain_FT,
        ) = load_data_from_files(
            config.cases_dir / args.train,
            config.cases_dir / args.val,
            config.cases_dir / args.test,
            args.use_multirest,
        )

        assert len(XTrain_FT) > 0, "No data for fine tuning in the partition"
        print(f"Fine tuning with {len(XTrain_FT)} elements")

        # Dictionaries
        w2i_ft, _ = check_and_retrieveVocabulary_from_files(
            nameOfVoc=nameOfVoc,
            use_multirest=args.use_multirest,
            encoding=args.encoding,
            YTrain=YFTrain,
            YVal=YFVal,
            YTest=YFTest,
            YTrain_FT=YTrain_FT,
        )

        # Freezing the model except for the specified parts
        if len(args.updateFT) >= 1 and args.updateFT[0] != "ALL":
            model.updateModel(list_update_elements=args.updateFT)

        # Destination paths
        model_filepath = model_filepath.parent / model_filepath.name.replace(
            ".pt", f"_ft-{args.epochs}epochs.pt"
        )
        logs_path = logs_path.parent / logs_path.name.replace(
            ".csv", f"-{args.epochs}epochs.csv"
        )

        # Fine-tune, validate, and test the model
        model.fit(
            train_data_generator(
                XFiles=XTrain_FT,
                YFiles=YTrain_FT,
                batch_size=args.batch_size,
                width_reduction=model.model.encoder.width_reduction,
                w2i=w2i_ft,
                device=device,
                encoding=args.encoding,
            ),
            epochs=args.epochs,
            steps_per_epoch=len(XTrain_FT) // args.batch_size,
            val_data=(XFVal, YFVal),
            test_data=(XFTest, YFTest),
            batch_size=args.batch_size,
            patience=args.patience,
            weights_path=model_filepath,
            logs_path=logs_path,
        )

    pass


if __name__ == "__main__":
    main()
