import os
import pathlib
from typing import List, Dict, Tuple

import numpy as np

import config
from my_utils.encoding_convertions import krnConverter


def filter_multirest(YFiles: List[str]) -> List[str]:
    YFiles_filetered = YFiles.copy()

    for y in YFiles:
        data = open(y, "r").read()
        if data.count("multirest") > 0:
            YFiles_filetered.remove(y)

    print(f"Removed {len(YFiles) - len(YFiles_filetered)} files")

    return YFiles_filetered


def filter_multirest_two_lists(
    XFiles: List[str], YFiles: List[str]
) -> Tuple[List[str], List[str]]:
    YFiles_filetered = YFiles.copy()
    XFiles_filetered = XFiles.copy()

    for it in range(len(YFiles)):
        data = open(YFiles[it], "r").read()
        if data.count("multirest") > 0:
            YFiles_filetered.remove(YFiles[it])
            XFiles_filetered.remove(XFiles[it])

    print(f"Removed {len(YFiles) - len(YFiles_filetered)} files")

    return XFiles_filetered, YFiles_filetered


def load_data_from_files(
    *args: Tuple[pathlib.PosixPath, pathlib.PosixPath, pathlib.PosixPath, bool]
) -> Tuple[
    List[str],  # XTrain
    List[str],  # YTrain
    List[str],  # XVal
    List[str],  # YVal
    List[str],  # XTest
    List[str],  # YTest
    List[str],  # XTrain_FineTune
    List[str],  # YTrain_FineTune
]:
    train_path, val_path, test_path, use_multirest = args

    # Loading train
    with open(train_path) as f:
        in_file = f.read().splitlines()
    XTrain = [u.split()[0] for u in in_file if not u.startswith("*")]
    YTrain = [u.split()[1] for u in in_file if not u.startswith("*")]
    XTrain_FT = [u.split()[0].split("*")[1] for u in in_file if u.startswith("*")]
    YTrain_FT = [u.split()[1] for u in in_file if u.startswith("*")]
    if not use_multirest:
        XTrain, YTrain = filter_multirest_two_lists(XTrain, YTrain)
        XTrain_FT, YTrain_FT = filter_multirest_two_lists(XTrain_FT, YTrain_FT)

    # Loading validation
    with open(val_path) as f:
        in_file = f.read().splitlines()
    XVal = [u.split()[0] for u in in_file]
    YVal = [u.split()[1] for u in in_file]
    if not use_multirest:
        XVal, YVal = filter_multirest_two_lists(XVal, YVal)

    # Loading test
    with open(test_path) as f:
        in_file = f.read().splitlines()
    XTest = [u.split()[0] for u in in_file]
    YTest = [u.split()[1] for u in in_file]
    if not use_multirest:
        XTest, YTest = filter_multirest_two_lists(XTest, YTest)

    return XTrain, YTrain, XVal, YVal, XTest, YTest, XTrain_FT, YTrain_FT


############################################## DICTIONARIES:


def check_and_retrieveVocabulary_from_files(
    nameOfVoc: str,
    use_multirest: bool,
    encoding: str,
    YTrain: List[str],
    YVal: List[str],
    YTest: List[str],
    YTrain_FT: List[str],
) -> Tuple[Dict[str, int], Dict[int, str]]:
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
    nameOfVoc: str, YFiles: List[str], encoding: str
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
