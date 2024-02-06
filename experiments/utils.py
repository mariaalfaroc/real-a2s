import os
import json
import requests
import tarfile
from typing import Dict, Tuple

from my_utils.encoding_convertions import krnConverter

PARTITIONS_PATH = "experiments/partitions"
RESULTS_PATH = "experiments/results"
os.makedirs(RESULTS_PATH, exist_ok=True)

############################################### Download dataset:

DATASET_PATH = "real_a2s_sax_dataset"


def download_and_extract_dataset():
    file_path = "real_a2s_sax_dataset.tgz"
    extract_path = "."

    # Download dataset
    response = requests.get(
        url="https://grfia.dlsi.ua.es/audio-to-score/real_a2s_sax_dataset.tgz"
    )
    with open(file_path, "wb") as file:
        file.write(response.content)
    # Extract dataset
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(extract_path)
    # Remove tar file
    os.remove(file_path)


############################################### Vocabulary:

VOCAB_PATH = "real_a2s_sax_dataset/vocab"
os.makedirs(VOCAB_PATH, exist_ok=True)


def check_and_retrieve_vocabulary(
    sax_type: str, encoding: str
) -> Tuple[Dict[str, int], Dict[int, str]]:
    def make_vocabulary(
        sax_type: str, encoding: str
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        krn_parser = krnConverter(encoding=encoding)

        vocab = []
        for f in os.listdir(os.path.join(DATASET_PATH, "krn", sax_type)):
            if (f.endswith(".krn") or f.endswith(".skm")) and not f.startswith("."):
                f = os.path.join(DATASET_PATH, "krn", sax_type, f)
                vocab.extend(krn_parser.convert(src_file=f))
        vocab = sorted(set(vocab))

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w

    w2i = {}
    i2w = {}

    w2i_path = os.path.join(VOCAB_PATH, sax_type, f"w2i_{encoding}.json")
    if os.path.isfile(w2i_path):
        with open(w2i_path, "r") as file:
            w2i = json.load(file)
        i2w = {v: k for k, v in w2i.items()}
    else:
        os.makedirs(os.path.join(VOCAB_PATH, sax_type), exist_ok=True)
        w2i, i2w = make_vocabulary(sax_type=sax_type, encoding=encoding)
        with open(w2i_path, "w") as file:
            json.dump(w2i, file)

    return w2i, i2w
