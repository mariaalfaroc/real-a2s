import os
from typing import List, Dict, Tuple, Generator

import torch
from sklearn.utils import shuffle

from my_utils.encoding_convertions import krnConverter
from my_utils.preprocessing import preprocess_audio, preprocess_label, ctc_preprocess


def load_train_data_from_fold_file(
    fold_path: str,
    x_folder: List[str],
    y_folder: str,
    percentage_size: List[float] = [1.0, 1.0],
) -> Tuple[List[str], List[str]]:
    # For train
    assert len(x_folder) == len(
        percentage_size
    ), "x_folder and percentage_size must have the same length!"
    TotalXFiles = []
    TotalYFiles = []
    for xf, ps in zip(x_folder, percentage_size):
        XFiles, YFiles = load_test_data_from_fold_file(
            fold_path=fold_path,
            x_folder=xf,
            y_folder=y_folder,
            percentage_size=ps,
        )
        TotalXFiles.extend(XFiles)
        TotalYFiles.extend(YFiles)
    return TotalXFiles, TotalYFiles


def load_test_data_from_fold_file(
    fold_path: str,
    x_folder: str,
    y_folder: str,
    percentage_size: float = 1.0,
) -> Tuple[List[str], List[str]]:
    # For val, test and finetune
    XFiles = []
    YFiles = []
    with open(fold_path, "r") as file:
        for line in file:
            x, y = line.strip().split("\t")
            XFiles.append(os.path.join(x_folder, x))
            YFiles.append(os.path.join(y_folder, y))
    if percentage_size < 1.0 and percentage_size is not None:
        XFiles, YFiles = shuffle(XFiles, YFiles, random_state=42)
        XFiles = XFiles[: int(len(XFiles) * percentage_size)]
        YFiles = YFiles[: int(len(YFiles) * percentage_size)]
    return XFiles, YFiles


def train_data_generator(
    *,
    XFiles: List[str],
    YFiles: List[str],
    batch_size: int,
    width_reduction: int,
    w2i: Dict[str, int],
    device: torch.device,
    krnParser: krnConverter,
) -> Generator[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None
]:  # Generator[yield_type, send_type, return_type]
    index = 0
    while True:
        # Get batch files
        xf = XFiles[index : index + batch_size]
        yf = YFiles[index : index + batch_size]
        # Load batch files
        x, xl = map(
            list,
            zip(
                *[
                    preprocess_audio(f, training=True, width_reduction=width_reduction)
                    for f in xf
                ]
            ),
        )
        y, yl = map(
            list,
            zip(
                *[
                    preprocess_label(f, training=True, w2i=w2i, krnParser=krnParser)
                    for f in yf
                ]
            ),
        )
        # CTC preprocess
        x, xl, y, yl = ctc_preprocess(x, xl, y, yl, pad_index=w2i["<pad>"])
        yield x.to(device), xl.to(device), y.to(device), yl.to(device)

        index = (index + batch_size) % len(X)
        if index == 0:
            X, XL, Y, YL = shuffle(X, XL, Y, YL, random_state=42)
