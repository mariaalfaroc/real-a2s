import os, glob, random

import torch
import joblib
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from madmom.audio.spectrogram import LogarithmicFilterbank, LogarithmicFilteredSpectrogram

from encoding_convertions import krnConverter

import config

# joblib settings!
memory = joblib.memory.Memory("./joblib_cache", mmap_mode="r", verbose=1)

IMG_HEIGHT = NUM_BINS = 229 

# ---------- DATA LOADING ---------- #

def filter_multirest(YFiles: list):
    YFiles_filetered = YFiles.copy()

    for y in YFiles:
        data =  open(y, "r").read()
        if data.count("multirest") > 0:
            YFiles_filetered.remove(y)

    print(f"Removed {len(YFiles) - len(YFiles_filetered)} files")

    return YFiles_filetered

def load_data(num_samples: int, num_iter: int, multirest: bool):
    labels = sorted(glob.glob(str(config.labels_dir) + "/*" + config.label_extn))
    if not multirest:
        labels = filter_multirest(YFiles=labels)

    audios = sorted(glob.glob(str(config.audios_dir) + "/*" + config.audio_extn))


    # Matching elements:
    audios = [file.split("/")[-1].split(".")[0] for file in audios]
    labels = [file.split("/")[-1].split(".")[0] for file in labels]
    common_files = sorted(list(set(audios).intersection(labels)))

    audios = [os.path.join(config.audios_dir, file + config.audio_extn) for file in common_files]
    labels = [os.path.join(config.labels_dir, file + config.label_extn) for file in common_files]

    
    idx = np.arange(len(labels))
    if num_samples > 0: idx = np.random.choice(idx, num_samples, replace=False)

    X = np.array(audios, dtype="object")[idx]
    Y = np.array(labels, dtype="object")[idx]

    # 60% - 20% - 20%
    XTrain, XValTest, YTrain, YValTest = train_test_split(X, Y, test_size=0.4, random_state=num_iter)
    XVal, XTest, YVal, YTest = train_test_split(XValTest, YValTest, test_size=0.5, random_state=num_iter)
    print(f"Train size: {len(XTrain)}")
    print(f"Val size: {len(XVal)}")
    print(f"Test size: {len(XTest)}")

    return XTrain.tolist(), YTrain.tolist(), XVal.tolist(), YVal.tolist(), XTest.tolist(), YTest.tolist()

@memory.cache
def get_spectrogram_from_file(audiofilename):
	audio_options = dict(
		num_channels=1,
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

def check_and_retrieveVocabulary(nameOfVoc, multirest, encoding):
    w2ipath = config.vocab_dir / f"{nameOfVoc}w2i.npy"
    i2wpath = config.vocab_dir / f"{nameOfVoc}i2w.npy"

    w2i = []
    i2w = []

    if os.path.isfile(w2ipath):
        w2i = np.load(w2ipath, allow_pickle=True).item()
        i2w = np.load(i2wpath, allow_pickle=True).item()
    else:
        YFiles = sorted(glob.glob(str(config.labels_dir) + "/*" + config.label_extn))
        if not multirest:
            YFiles = filter_multirest(YFiles)
        w2i, i2w = make_vocabulary(nameOfVoc, YFiles, encoding)

    return w2i, i2w

def make_vocabulary(nameOfVoc, YFiles, encoding):
    w2ipath = config.vocab_dir / f"{nameOfVoc}w2i.npy"
    i2wpath = config.vocab_dir / f"{nameOfVoc}i2w.npy"


    krnParser = krnConverter()

    # Create vocabulary
    vocabulary = list()
    for yf in YFiles: 
        vocabulary.extend(krnParser.convert(yf, encoding))
        # vocabulary.extend(open(yf, "r").read().split())
    vocabulary = sorted(set(vocabulary))

    w2i = dict()
    i2w = dict()
    for i, w in enumerate(vocabulary):
        w2i[w] = i + 1
        i2w[i + 1] = w
    w2i["<pad>"] = 0
    i2w[0] = "<pad>"

    # Save vocabulary
    np.save(w2ipath, w2i)
    np.save(i2wpath, i2w)

    return w2i, i2w

def preprocess_audio(path, width_reduction):
    x = get_spectrogram_from_file(path)
        # [num_frames, num_bins] == [width, height]
    x = np.transpose(x)
        # [height, width]
    x = np.flip(x, 0)   # Because of the ordering of the bins: from 0 Hz to max_freq Hz
    x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    x = np.expand_dims(x, 0)
        # [1, height, width]
    return x, x.shape[2] // width_reduction

def preprocess_label(path, training, w2i, encoding):
    krnParser = krnConverter()
    y = krnParser.convert(path, encoding)
    if training:
        y = [w2i[w] for w in y]
        return y, len(y)
    return y

def ctc_preprocess(x, xl, y, yl, pad_index):
    # Zero-pad images to maximum batch image width
    max_width = max(x, key=np.shape).shape[2]
    x = np.array([np.pad(i, pad_width=((0, 0), (0, 0), (0, max_width - i.shape[2]))) for i in x], dtype=np.float32)
    # Zero-pad labels to maximum batch label length
    max_length = len(max(y, key=len))
    y = np.array([i + [pad_index] * (max_length - len(i)) for i in y], dtype=np.int32)
    return torch.from_numpy(x), torch.tensor(xl, dtype=torch.int32), torch.from_numpy(y), torch.tensor(yl, dtype=torch.int32)

def train_data_generator(*, XFiles, YFiles, batch_size, width_reduction, w2i, device, encoding):
    # Load all data in RAM
    X, XL = zip(*[preprocess_audio(xf, width_reduction) for xf in XFiles])
    Y, YL = zip(*[preprocess_label(yf, training=True, w2i=w2i, encoding=encoding) for yf in YFiles])

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

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs("CHECK", exist_ok=True)
    
    multirest = False
    nameOfVoc = "Vocab"
    nameOfVoc = nameOfVoc + "_woutmultirest" if not multirest else nameOfVoc

    config.set_source_data_dirs("Primus")
    XTrain, YTrain, XVal, YVal, XTest, YTest = load_data(num_samples=1000, num_iter=0, multirest=multirest)
    print(XTrain[:2], YTrain[:2])

    w2i, i2w = check_and_retrieveVocabulary(nameOfVoc=nameOfVoc, multirest=multirest, encoding = 'kern')
    print(w2i)
    print("Vocabulary size:", len(w2i.keys()))
    
    gen = train_data_generator(XFiles=XTrain, YFiles=YTrain, batch_size=16, width_reduction=2, w2i=w2i, device=torch.device("cpu"))
    x, xl, y, yl = next(gen)
    print(x.shape, xl.shape, y.shape, yl.shape)
    print(f"Shape with padding: {y[0].shape}; Original shape: {yl[0].numpy()}")
    print([i2w[int(i)] for i in y[0]])

    save_image(make_grid(x, nrow=1), f"CHECK/x_batch.jpg")
    save_image(x[0], f"CHECK/x0.jpg")