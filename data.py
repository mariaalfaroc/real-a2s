import os
import pathlib
from typing import Tuple, Generator, Union, List

import torch
import joblib
import numpy as np
from sklearn.utils import shuffle
from madmom.audio.spectrogram import LogarithmicFilterbank, LogarithmicFilteredSpectrogram

import config
from encoding_convertions import krnConverter

# joblib settings!
memory = joblib.memory.Memory("./joblib_cache", mmap_mode="r", verbose=1)

NUM_CHANNELS = 1
IMG_HEIGHT = NUM_BINS = 229 

# ---------- DATA LOADING ---------- #

def filter_multirest(YFiles: List[str]) -> List[str]:
    YFiles_filetered = YFiles.copy()

    for y in YFiles:
        data =  open(y, "r").read()
        if data.count("multirest") > 0:
            YFiles_filetered.remove(y)

    print(f"Removed {len(YFiles) - len(YFiles_filetered)} files")

    return YFiles_filetered

def filter_multirest_two_lists(XFiles: List[str], YFiles: List[str]) -> Tuple[List[str], List[str]]:
    YFiles_filetered = YFiles.copy()
    XFiles_filetered = XFiles.copy()

    for it in range(len(YFiles)):
        data =  open(YFiles[it], "r").read()
        if data.count("multirest") > 0:
            YFiles_filetered.remove(YFiles[it])
            XFiles_filetered.remove(XFiles[it])

    print(f"Removed {len(YFiles) - len(YFiles_filetered)} files")

    return XFiles_filetered, YFiles_filetered

def load_data_from_files(
    *args: Tuple[pathlib.PosixPath, pathlib.PosixPath, pathlib.PosixPath, bool]
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
    train_path, val_path, test_path, use_multirest = args

    # Loading train:
    with open(train_path) as f:
        in_file = f.read().splitlines()
    XTrain  = [u.split()[0] for u in in_file if not u.startswith("*")]
    YTrain  = [u.split()[1] for u in in_file if not u.startswith("*")]
    XTrain_FT  = [u.split()[0].split("*")[1] for u in in_file if u.startswith("*")]
    YTrain_FT  = [u.split()[1] for u in in_file if u.startswith("*")]
    if not use_multirest:
        XTrain, YTrain = filter_multirest_two_lists(XTrain, YTrain)
        XTrain_FT, YTrain_FT = filter_multirest_two_lists(XTrain_FT, YTrain_FT)

    # Loading validation:
    with open(val_path) as f:
        in_file = f.read().splitlines()
    XVal  = [u.split()[0] for u in in_file]
    YVal  = [u.split()[1] for u in in_file]
    if not use_multirest:
        XVal, YVal = filter_multirest_two_lists(XVal, YVal)

    # Loading test:
    with open(test_path) as f:
        in_file = f.read().splitlines()
    XTest  = [u.split()[0] for u in in_file]
    YTest  = [u.split()[1] for u in in_file]
    if not use_multirest:
        XTest, YTest = filter_multirest_two_lists(XTest, YTest)

    return XTrain, YTrain, XVal, YVal, XTest, YTest, XTrain_FT, YTrain_FT

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
		norm=True
	)
	x = LogarithmicFilteredSpectrogram(audiofilename, **audio_options)
    #               width     height
    # x.shape = [num_frames, num_bins]
    # num_frames will vary from file to file because of the audio duration
    # num_bins will be the same for all files
	return x

# ---------- TRANSCRIPTION UTILS ---------- #
def check_and_retrieveVocabulary_from_files(
    nameOfVoc: str,
    use_multirest: bool,
    encoding: str,
    YTrain: List[str],
    YVal: List[str],
    YTest: List[str],
    YTrain_FT: List[str]
) -> Tuple[dict, dict]:
    YFiles = YTrain.copy()
    YFiles.extend(YVal)
    YFiles.extend(YTest)
    YFiles.extend(YTrain_FT)

    w2ipath = config.vocab_dir / f"{nameOfVoc}w2i.npy"
    i2wpath = config.vocab_dir / f"{nameOfVoc}i2w.npy"

    w2i = []
    i2w = []

    if os.path.isfile(w2ipath):
        w2i = np.load(w2ipath, allow_pickle=True).item()
        i2w = np.load(i2wpath, allow_pickle=True).item()
    else:
        if not use_multirest:
            YFiles = filter_multirest(YFiles)
        w2i, i2w = make_vocabulary(nameOfVoc, YFiles, encoding)

    return w2i, i2w

def make_vocabulary(
    nameOfVoc: str,
    YFiles: List[str],
    encoding: str
) -> Tuple[dict, dict]:
    w2ipath = config.vocab_dir / f"{nameOfVoc}w2i.npy"
    i2wpath = config.vocab_dir / f"{nameOfVoc}i2w.npy"

    krnParser = krnConverter(encoding=encoding)

    # Create vocabulary
    vocabulary = []
    for yf in YFiles: 
        vocabulary.extend(krnParser.convert(yf))
    vocabulary = sorted(set(vocabulary))

    w2i = {}
    i2w = {}
    for i, w in enumerate(vocabulary):
        w2i[w] = i + 1
        i2w[i + 1] = w
    w2i["<pad>"] = 0
    i2w[0] = "<pad>"

    # Save vocabulary
    np.save(w2ipath, w2i)
    np.save(i2wpath, i2w)

    return w2i, i2w

def preprocess_audio(path: str, width_reduction: int) -> Tuple[np.ndarray, int]:
    x = get_spectrogram_from_file(path)
        # [num_frames, num_bins] == [width, height]
    x = np.transpose(x)
        # [height, width]
    x = np.flip(x, 0)   # Because of the ordering of the bins: from 0 Hz to max_freq Hz
    x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    x = np.expand_dims(x, 0)
        # [1, height, width]
    return x, x.shape[2] // width_reduction

def preprocess_label(
    path: str,
    training: bool,
    w2i: dict,
    encoding: str
) -> Union[Tuple[List[int], int], List[str]]:
    krnParser = krnConverter(encoding=encoding)
    y = krnParser.convert(path)
    if training:
        y = [w2i[w] for w in y]
        return y, len(y)
    return y

def ctc_preprocess(
    x: List[LogarithmicFilteredSpectrogram],
    xl: List[int],
    y: List[List[int]],
    yl: List[int],
    pad_index: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Zero-pad audios to maximum batch audio width
    max_width = max(x, key=np.shape).shape[2]
    x = np.array([np.pad(i, pad_width=((0, 0), (0, 0), (0, max_width - i.shape[2]))) for i in x], dtype=np.float32)
    # Zero-pad labels to maximum batch label length
    max_length = len(max(y, key=len))
    y = np.array([i + [pad_index] * (max_length - len(i)) for i in y], dtype=np.int32)
    return torch.from_numpy(x), torch.tensor(xl, dtype=torch.int32), torch.from_numpy(y), torch.tensor(yl, dtype=torch.int32)

def train_data_generator(
    *,
    XFiles: List[str],
    YFiles: List[str],
    batch_size: int,
    width_reduction: int,
    w2i: dict,
    device: torch.device,
    encoding: str
) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:  # Generator[yield_type, send_type, return_type]
    # Load all data in RAM
    X, XL = map(list, zip(*[preprocess_audio(xf, width_reduction) for xf in XFiles]))
    Y, YL = map(list, zip(*[preprocess_label(yf, training=True, w2i=w2i, encoding=encoding) for yf in YFiles]))

    index = 0
    while True:
        x, xl, y, yl = X[index:index + batch_size], XL[index:index + batch_size], Y[index:index + batch_size], YL[index:index + batch_size]
        x, xl, y, yl = ctc_preprocess(x, xl, y, yl, pad_index=w2i["<pad>"])
        yield x.to(device), xl.to(device), y.to(device), yl.to(device)
        index = (index + batch_size) % len(X)
        if index == 0:
            X, XL, Y, YL = shuffle(X, XL, Y, YL, random_state=42)

if __name__ == "__main__":
    from torchvision.utils import make_grid, save_image

    CHECK_DIR = "CHECK"

    os.makedirs(CHECK_DIR, exist_ok=True)
    
    use_multirest = False
    encoding = "kern"

    test = "Experiment7-F0_test.dat"
    train = "Experiment7-F0_train.dat"
    val = "Experiment7-F0_val.dat"

    nameOfVoc = "Vocab"
    nameOfVoc = nameOfVoc + "_woutmultirest" if not use_multirest else nameOfVoc

    config.set_source_data_dirs()
    XFTrain, YFTrain, XFVal, YFVal, XFTest, YFTest, XTrain_FT, YTrain_FT = load_data_from_files(config.cases_dir / train,\
        config.cases_dir / val, config.cases_dir / test, use_multirest)
    w2i, i2w = check_and_retrieveVocabulary_from_files(nameOfVoc=nameOfVoc, use_multirest=use_multirest, encoding=encoding,\
        YTrain=YFTrain, YVal=YFVal, YTest=YFTest, YTrain_FT=YTrain_FT)

    print(w2i)
    print("Vocabulary size:", len(w2i.keys()))

    gen = train_data_generator(
        XFiles=XFTrain, YFiles=YFTrain,
        batch_size=16,
        width_reduction=2,
        w2i=w2i,
        device=torch.device("cpu"),
        encoding=encoding
    )
    x, xl, y, yl = next(gen)
    print(x.shape, xl.shape, y.shape, yl.shape)
    print(f"Shape with padding: {y[0].shape}; Original shape: {yl[0].numpy()}")
    print([i2w[int(i)] for i in y[0]])

    save_image(make_grid(x, nrow=1), f"{CHECK_DIR}/x_batch.jpg")
    save_image(x[0], f"{CHECK_DIR}/x0.jpg")