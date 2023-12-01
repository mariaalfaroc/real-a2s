from typing import List, Dict, Tuple, Generator

import torch
from sklearn.utils import shuffle

from my_utils.encoding_convertions import krnConverter
from my_utils.preprocessing import preprocess_audio, preprocess_label, ctc_preprocess


def train_data_generator(
    *,
    XFiles: List[str],
    YFiles: List[str],
    batch_size: int,
    width_reduction: int,
    w2i: Dict[str, int],
    device: torch.device,
    encoding: str,
) -> Generator[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None
]:  # Generator[yield_type, send_type, return_type]
    # Kern converter
    krnParser = krnConverter(encoding=encoding)

    # Load all data in RAM
    X, XL = map(list, zip(*[preprocess_audio(xf, width_reduction) for xf in XFiles]))
    Y, YL = map(
        list,
        zip(
            *[
                preprocess_label(yf, training=True, w2i=w2i, krnParser=krnParser)
                for yf in YFiles
            ]
        ),
    )

    index = 0
    while True:
        x, xl, y, yl = (
            X[index : index + batch_size],
            XL[index : index + batch_size],
            Y[index : index + batch_size],
            YL[index : index + batch_size],
        )
        x, xl, y, yl = ctc_preprocess(x, xl, y, yl, pad_index=w2i["<pad>"])
        yield x.to(device), xl.to(device), y.to(device), yl.to(device)

        index = (index + batch_size) % len(X)
        if index == 0:
            X, XL, Y, YL = shuffle(X, XL, Y, YL, random_state=42)
