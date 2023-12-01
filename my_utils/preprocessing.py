from typing import List, Dict, Tuple, Union

import torch
import numpy as np
from madmom.audio.spectrogram import (
    LogarithmicFilterbank,
    LogarithmicFilteredSpectrogram,
)

from my_utils.data import NUM_CHANNELS
from my_utils.encoding_convertions import krnConverter


def get_spectrogram_from_file(audiofilename: str) -> LogarithmicFilteredSpectrogram:
    audio_options = dict(
        num_channels=NUM_CHANNELS,
        sample_rate=44100,
        filterbank=LogarithmicFilterbank,
        frame_size=4096,
        fft_size=4096,
        hop_size=441 * 2,  # 25 fps -> 441 * 4 ; 50 fps -> 441 * 2
        num_bands=48,
        fmin=30,
        fmax=8000.0,
        fref=440.0,
        norm_filters=True,
        unique_filters=True,
        circular_shift=False,
        norm=True,
    )
    x = LogarithmicFilteredSpectrogram(audiofilename, **audio_options)
    #               width     height
    # x.shape = [num_frames, num_bins]
    # num_frames will vary from file to file because of the audio duration
    # num_bins will be the same for all files
    return x


def preprocess_audio(path: str, width_reduction: int) -> Tuple[np.ndarray, int]:
    x = get_spectrogram_from_file(path)
    # [num_frames, num_bins] == [width, height]
    x = np.transpose(x)
    # [height, width]
    x = np.flip(x, 0)  # Because of the ordering of the bins: from 0 Hz to max_freq Hz
    x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    x = np.expand_dims(x, 0)
    # [1, height, width]
    return x, x.shape[2] // width_reduction


def preprocess_label(
    path: str, training: bool, w2i: Dict[str, int], krnParser: krnConverter
) -> Union[Tuple[List[int], int], List[str]]:
    y = krnParser.convert(path)
    if training:
        y = [w2i[w] for w in y]
        return y, len(y)
    return y


############################################## TRAIN DATA GENERATOR BATCH PREPROCESS:


def ctc_preprocess(
    x: List[LogarithmicFilteredSpectrogram],
    xl: List[int],
    y: List[List[int]],
    yl: List[int],
    pad_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Zero-pad audios to maximum batch audio width
    max_width = max(x, key=np.shape).shape[2]
    x = np.array(
        [np.pad(i, pad_width=((0, 0), (0, 0), (0, max_width - i.shape[2]))) for i in x],
        dtype=np.float32,
    )
    # Zero-pad labels to maximum batch label length
    max_length = len(max(y, key=len))
    y = np.array([i + [pad_index] * (max_length - len(i)) for i in y], dtype=np.int32)
    return (
        torch.from_numpy(x),
        torch.tensor(xl, dtype=torch.int32),
        torch.from_numpy(y),
        torch.tensor(yl, dtype=torch.int32),
    )
