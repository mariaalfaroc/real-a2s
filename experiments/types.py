import os
from typing import Dict

import torch

from network.model import CTCTrainedCRNN
from experiments.config import EXPERIMENTS
from my_utils.generators import (
    load_train_data_from_fold_file,
    load_test_data_from_fold_file,
    train_data_generator,
)
from experiments.utils import DATASET_PATH, PARTITIONS_PATH, RESULTS_PATH


#################################################################### EXPERIMENT TYPE 1 (train from scratch)


def train_model(
    *,
    experiment_id: int,
    encoding: str,
    w2i: Dict[str, int],
    i2w: Dict[int, str],
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
):
    # Retrieve experiment information
    experiment_dict = EXPERIMENTS[experiment_id]

    # Get the train, val and test finetune data folders
    x_train_folders = [
        os.path.join(DATASET_PATH, folder, experiment_dict["sax_type"])
        for folder in experiment_dict["train"]
    ]
    x_val_folder = os.path.join(
        DATASET_PATH,
        experiment_dict["val"],
        experiment_dict["sax_type"],
    )
    x_test_folder = os.path.join(
        DATASET_PATH,
        experiment_dict["test"],
        experiment_dict["sax_type"],
    )
    y_folder = os.path.join(DATASET_PATH, "krn", experiment_dict["sax_type"])

    # 5-fold cross-validation
    for fold in sorted(os.listdir(PARTITIONS_PATH)):
        if not fold.startswith("."):
            print(f"Fold: {fold}")

            # Get test filepaths
            XTrainFiles, YTrainFiles = load_train_data_from_fold_file(
                fold_path=os.path.join(PARTITIONS_PATH, fold, "train.txt"),
                x_folder=x_train_folders,
                y_folder=y_folder,
                percentage_size=experiment_dict["train_p"],
            )
            XValFiles, YValFiles = load_test_data_from_fold_file(
                fold_path=os.path.join(PARTITIONS_PATH, fold, "val.txt"),
                x_folder=x_val_folder,
                y_folder=y_folder,
            )
            XTestFiles, YTestFiles = load_test_data_from_fold_file(
                fold_path=os.path.join(PARTITIONS_PATH, fold, "test.txt"),
                x_folder=x_test_folder,
                y_folder=y_folder,
            )

            # Create model
            model = CTCTrainedCRNN(
                dictionaries=(w2i, i2w),
                encoding=encoding,
                device=device,
            )

            # Weights and logs path
            weights_path = os.path.join(
                RESULTS_PATH,
                f"experiment_{experiment_id}",
                "weights",
                f"model_{encoding}_{fold}.pt",
            )
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            logs_path = weights_path.replace("weights", "logs").replace(".pt", ".csv")
            os.makedirs(os.path.dirname(logs_path), exist_ok=True)

            # Evaluate the model
            model.fit(
                train_data_generator=train_data_generator(
                    XFiles=XTrainFiles,
                    YFiles=YTrainFiles,
                    batch_size=batch_size,
                    width_reduction=model.model.encoder.width_reduction,
                    w2i=w2i,
                    device=device,
                    krnParser=model.krnParser,
                ),
                epochs=epochs,
                steps_per_epoch=len(XTrainFiles) // batch_size,
                val_data=(XValFiles, YValFiles),
                patience=patience,
                test_data=(XTestFiles, YTestFiles),
                weights_path=weights_path,
                logs_path=logs_path,
            )


#################################################################### EXPERIMENT TYPE 2 (test an already trained model)


def test_model(
    *,
    experiment_id: int,
    encoding: str,
    w2i: Dict[str, int],
    i2w: Dict[int, str],
    device: torch.device,
):
    # Retrieve experiment information
    experiment_dict = EXPERIMENTS[experiment_id]

    # Get the test data folders
    x_test_folder = os.path.join(
        DATASET_PATH,
        experiment_dict["test"],
        experiment_dict["sax_type"],
    )
    y_test_folder = os.path.join(DATASET_PATH, "krn", experiment_dict["sax_type"])

    # 5-fold cross-validation
    for fold in sorted(os.listdir(PARTITIONS_PATH)):
        if not fold.startswith("."):
            print(f"Fold: {fold}")

            # Get test filepaths
            XTestFiles, YTestFiles = load_test_data_from_fold_file(
                fold_path=os.path.join(PARTITIONS_PATH, fold, "test.txt"),
                x_folder=x_test_folder,
                y_folder=y_test_folder,
            )

            # Retrieve the model
            weights_path = os.path.join(
                RESULTS_PATH,
                f"experiment_{experiment_dict['from_experiment']}",
                "weights",
                f"model_{encoding}_{fold}.pt",
            )
            if os.path.exists(weights_path):
                model = CTCTrainedCRNN(
                    dictionaries=(w2i, i2w),
                    encoding=encoding,
                    device=device,
                )
                model.load(path=weights_path)
            else:
                raise FileNotFoundError(f"Model {weights_path} not found!")

            # Logs path
            logs_path = (
                weights_path.replace("weights", "logs")
                .replace(".pt", ".csv")
                .replace(
                    f"experiment_{experiment_dict['from_experiment']}",
                    f"experiment_{experiment_id}",
                )
            )
            os.makedirs(os.path.dirname(logs_path), exist_ok=True)

            # Evaluate the model
            model.fit(
                # No training!
                train_data_generator=None,
                epochs=0,
                steps_per_epoch=0,
                val_data=None,
                patience=0,
                # Only testing!
                test_data=(XTestFiles, YTestFiles),
                weights_path=weights_path,
                logs_path=logs_path,
            )


#################################################################### EXPERIMENT TYPE 3 (finetune an already trained model)


def finetune_model(
    *,
    experiment_id: int,
    encoding: str,
    w2i: Dict[str, int],
    i2w: Dict[int, str],
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
):
    # Retrieve experiment information
    experiment_dict = EXPERIMENTS[experiment_id]

    # Get the train, val and test finetune data folders
    x_train_folder = os.path.join(
        DATASET_PATH,
        experiment_dict["finetune"],
        experiment_dict["sax_type"],
    )
    x_val_folder = os.path.join(
        DATASET_PATH,
        experiment_dict["val"],
        experiment_dict["sax_type"],
    )
    x_test_folder = os.path.join(
        DATASET_PATH,
        experiment_dict["test"],
        experiment_dict["sax_type"],
    )
    y_folder = os.path.join(DATASET_PATH, "krn", experiment_dict["sax_type"])

    # 5-fold cross-validation
    for fold in sorted(os.listdir(PARTITIONS_PATH)):
        if not fold.startswith("."):
            print(f"Fold: {fold}")

            # Get test filepaths
            XTrainFiles, YTrainFiles = load_test_data_from_fold_file(
                fold_path=os.path.join(PARTITIONS_PATH, fold, "train.txt"),
                x_folder=x_train_folder,
                y_folder=y_folder,
                percentage_size=experiment_dict["finetune_p"],
            )
            XValFiles, YValFiles = load_test_data_from_fold_file(
                fold_path=os.path.join(PARTITIONS_PATH, fold, "val.txt"),
                x_folder=x_val_folder,
                y_folder=y_folder,
            )
            XTestFiles, YTestFiles = load_test_data_from_fold_file(
                fold_path=os.path.join(PARTITIONS_PATH, fold, "test.txt"),
                x_folder=x_test_folder,
                y_folder=y_folder,
            )

            # Retrieve the model
            from_weights_path = os.path.join(
                RESULTS_PATH,
                f"experiment_{experiment_dict['from_experiment']}",
                "weights",
                f"model_{encoding}_{fold}.pt",
            )
            if os.path.exists(from_weights_path):
                model = CTCTrainedCRNN(
                    dictionaries=(w2i, i2w),
                    encoding=encoding,
                    device=device,
                )
                model.load(path=from_weights_path)
            else:
                raise FileNotFoundError(f"Model {from_weights_path} not found!")

            # New weights and logs path
            weights_path = from_weights_path.replace(
                f"experiment_{experiment_dict['from_experiment']}",
                f"experiment_{experiment_id}",
            )
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            logs_path = weights_path.replace("weights", "logs").replace(".pt", ".csv")
            os.makedirs(os.path.dirname(logs_path), exist_ok=True)

            # Evaluate the model
            model.fit(
                train_data_generator=train_data_generator(
                    XFiles=XTrainFiles,
                    YFiles=YTrainFiles,
                    batch_size=batch_size,
                    width_reduction=model.model.encoder.width_reduction,
                    w2i=w2i,
                    device=device,
                    krnParser=model.krnParser,
                ),
                epochs=epochs,
                steps_per_epoch=len(XTrainFiles) // batch_size,
                val_data=(XValFiles, YValFiles),
                patience=patience,
                test_data=(XTestFiles, YTestFiles),
                weights_path=weights_path,
                logs_path=logs_path,
            )
