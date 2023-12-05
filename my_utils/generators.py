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
