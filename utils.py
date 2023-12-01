import os
import shutil
from typing import List, Tuple

import torch
from pyMV2H.utils.mv2h import MV2H
from pyMV2H.metrics.mv2h import mv2h
from pyMV2H.utils.music import Music
from music21 import converter as converterm21
from pyMV2H.converter.midi_converter import MidiConverter as Converter

from encoding_convertions import decoupledDotKern2Kern, decoupledKern2Kern


def ctc_greedy_decoder(
    y_pred: torch.Tensor, xl: List[int], i2w: dict
) -> List[List[str]]:
    y_pred_decoded = []
    for s, l in zip(y_pred, xl):
        # Get real lenght
        s = s[:l, :]
        # Best path
        s = torch.argmax(s, dim=-1)
        # Merge repeated elements
        s = torch.unique_consecutive(s, dim=-1)
        # Convert to string; len(i2w) -> CTC-blank
        y_pred_decoded.append([i2w[int(i)] for i in s if int(i) != len(i2w)])
    return y_pred_decoded


# --------- METRICS --------- #


def compute_metrics(
    y_true: List[List[str]], y_pred: List[List[str]], encoding: str, aux_name: str
) -> dict:
    # ------------------------------- SER:
    ser = compute_ser(y_true, y_pred)
    # ------------------------------- MV2H:
    mv2h_dict = compute_mv2h(y_true, y_pred, encoding, aux_name)
    # ------------------------------- COMBINE:
    metrics = {"ser": ser, **mv2h_dict}
    return metrics


def levenshtein(a: list, b: list) -> int:
    n, m = len(a), len(b)

    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def compute_ser(y_true: List[List[str]], y_pred: List[List[str]]) -> float:
    ed_acc = 0
    length_acc = 0
    for t, h in zip(y_true, y_pred):
        ed_acc += levenshtein(t, h)
        length_acc += len(t)
    return 100.0 * ed_acc / length_acc


def compute_mv2h(
    y_true: List[List[str]], y_pred: List[List[str]], encoding: str, aux_name: str
) -> dict:
    def kern2txt(
        kern: List[str], encoding: str, aux_name: str
    ) -> Tuple[List[str], str]:
        # ------------------------------- MIDI:

        if encoding == "decoupled":
            kern = decoupledKern2Kern(kern)
        elif encoding == "decoupled_dot":
            kern = decoupledDotKern2Kern(kern)

        midi = kern.copy()
        midi.insert(0, "**kern")
        midi = "\n".join(midi)
        with open(aux_name + ".krn", "w") as fout:
            fout.write(midi)
        midi = converterm21.parse(aux_name + ".krn").write("midi")
        midi_file = midi.name

        shutil.copyfile(midi, midi_file)
        os.remove(aux_name + ".krn")

        # ------------------------------- TXT:
        txt_file = midi_file.replace("mid", "txt")
        converter = Converter(file=midi_file, output=txt_file)
        converter.convert_file()

        with open(txt_file, "r") as fin:
            f = fin.read().replace(".0", "")
        with open(txt_file, "w") as fout:
            fout.write(f)

        os.remove(midi_file)

        return kern, txt_file

    mv2h_total = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)

    krn_y_true = []
    krn_y_pred = []
    for t, h in zip(y_true, y_pred):
        t_krn, t_txt_file = kern2txt(
            kern=t, encoding=encoding, aux_name=aux_name + "_true"
        )
        h_krn, h_txt_file = kern2txt(
            kern=h, encoding=encoding, aux_name=aux_name + "_pred"
        )

        # Append to the list of kerns:
        krn_y_true.append(t_krn)
        krn_y_pred.append(h_krn)

        # MV2H:
        t_file = Music.from_file(t_txt_file)
        h_file = Music.from_file(h_txt_file)

        res_dict = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)
        try:
            res_dict = mv2h(t_file, h_file)
            mv2h_total.__multi_pitch__ += res_dict.multi_pitch
            mv2h_total.__voice__ += res_dict.voice
            mv2h_total.__meter__ += res_dict.meter
            mv2h_total.__harmony__ += res_dict.harmony
            mv2h_total.__note_value__ += res_dict.note_value
        except:
            pass
        os.remove(t_txt_file)
        os.remove(h_txt_file)

    mv2h_total.__multi_pitch__ /= len(y_true)
    mv2h_total.__voice__ /= len(y_true)
    mv2h_total.__meter__ /= len(y_true)
    mv2h_total.__harmony__ /= len(y_true)
    mv2h_total.__note_value__ /= len(y_true)

    mv2h_final_value = (
        mv2h_total.multi_pitch
        + mv2h_total.voice
        + mv2h_total.meter
        + mv2h_total.harmony
        + mv2h_total.note_value
    ) / 5
    mv2h_final_value *= 100

    # SER for the obtained kerns after the transducer:
    recon_ser = compute_ser(krn_y_true, krn_y_pred)

    return {"mv2h": mv2h_final_value, "recon_ser": recon_ser}


# --------- OTHERS --------- #


def str2bool(v: str) -> bool:
    if v == "True":
        return True
    return False


if __name__ == "__main__":
    examples = {
        "kern": {
            "y_true": [
                [
                    "*clefG2",
                    "*k[b-]",
                    "*M6/8",
                    "*MM90",
                    "=",
                    "4.dd",
                    "4.bb-",
                    "=",
                    "4.ff",
                    "4.cc",
                    "=",
                    "4f",
                    "8a",
                    "4cc",
                    "8ee",
                    "=",
                    "4.ff",
                    "4gg",
                    "8cc",
                    "=",
                    "8ff",
                    "8ee",
                    "8dd",
                    "8cc",
                    "8b-",
                    "8a",
                    "=",
                    "8dd",
                    "8cc",
                    "8b-",
                    "4.ee",
                    "=",
                    "8ff",
                    "8aa",
                    "8dd",
                    "8cc",
                    "8dd",
                    "8ee",
                    "=",
                    "4.ff",
                    "4.dd",
                    "=",
                    "2.gg",
                    "=",
                ]
            ],
            "y_pred": [
                [
                    "*clefG2",
                    "*k[b-e-a-]",
                    "*M6/8",
                    "*MM90",
                    "=",
                    "4.dd",
                    "4.bb-",
                    "4ff",
                    "4.cc",
                    "8f",
                    "8a",
                    "8cc",
                    "8ee",
                    "=",
                    "4.ff",
                    "8gg",
                    "=",
                    "8cc",
                    "8ff",
                    "8ee",
                    "8dd",
                    "8cc",
                    "8b-",
                    "8a",
                    "8dd",
                    "8cc",
                    "8b-",
                    "=",
                    "4.ee",
                    "=",
                    "8ff",
                    "8aa",
                    "8dd",
                    "8cc",
                    "8dd",
                    "8ee",
                    "=",
                    "4.ff",
                    "4dd",
                    "2gg",
                    "=",
                ]
            ],
        },
        "decoupled": {
            "y_true": [
                [
                    "*clefG2",
                    "*k[]",
                    "*M3/4",
                    "*MM80",
                    "=",
                    "2",
                    ".",
                    "ff",
                    "=",
                    "4",
                    "ff",
                    "4",
                    "ff",
                    "4",
                    "ff",
                    "=",
                    "8",
                    "ff",
                    "8",
                    "ff",
                    "8",
                    "ff",
                    "8",
                    "ff",
                    "8",
                    "ff",
                    "8",
                    "ff",
                    "=",
                    "2",
                    ".",
                    "ff",
                    "=",
                ]
            ],
            "y_pred": [
                [
                    "*clefG2",
                    "*k[]",
                    "*M3/4",
                    "*MM80",
                    "=",
                    "2",
                    "ff",
                    "4",
                    "ff",
                    "4",
                    "ff",
                    "4",
                    "ff",
                    "8",
                    "ff",
                    "8",
                    "ff",
                    "8",
                    "ff",
                    "8",
                    "ff",
                    "8",
                    "ff",
                    "8",
                    "ff",
                    "=",
                    "2",
                    "ff",
                    "=",
                ]
            ],
        },
        "decoupled_dot": {
            "y_true": [
                [
                    "*clefG2",
                    "*k[b-e-]",
                    "*M3/4",
                    "*MM85",
                    "=",
                    "4.",
                    "bb",
                    "-",
                    "8",
                    "ddd",
                    "16",
                    "eee",
                    "-",
                    "8",
                    "ccc",
                    "16",
                    "aa",
                    "=",
                    "4",
                    "gg",
                    "16",
                    "gg",
                    "16",
                    "aa",
                    "16",
                    "ff",
                    "16",
                    "cc",
                    "8",
                    "cc",
                    "8",
                    "dd",
                    "=",
                    "4",
                    "r",
                    "8",
                    "ee",
                    "-",
                    "8",
                    "gg",
                    "4",
                    "r",
                    "=",
                    "8",
                    "ff",
                    "8",
                    "aa",
                    "4",
                    "r",
                    "4",
                    "gg",
                    "=",
                    "8.",
                    "ccc",
                    "16",
                    "ddd",
                    "2",
                    "aa",
                    "=",
                ]
            ],
            "y_pred": [
                [
                    "*clefG2",
                    "*k[b-e-a-d-g-]",
                    "*M3/4",
                    "*MM120",
                    "=",
                    "4.",
                    "bb",
                    "-",
                    "16",
                    "ddd",
                    "16",
                    "eee",
                    "-",
                    "=",
                    "16",
                    "ccc",
                    "16",
                    "aa",
                    "4",
                    "gg",
                    "16",
                    "gg",
                    "16",
                    "aa",
                    "16",
                    "ff",
                    "16",
                    "cc",
                    "8",
                    "cc",
                    "8",
                    "dd",
                    "4",
                    "r",
                    "8",
                    "ee",
                    "-",
                    "8",
                    "gg",
                    "4",
                    "r",
                    "8",
                    "ff",
                    "8",
                    "aa",
                    "4",
                    "r",
                    "=",
                    "4",
                    "gg",
                    "8",
                    "ccc",
                    "16",
                    "ddd",
                    "=",
                    "2.",
                    "aa",
                    "=",
                ]
            ],
        },
    }

    for encoding in examples.keys():
        print("Encoding: ", encoding)
        y_true = examples[encoding]["y_true"]
        y_pred = examples[encoding]["y_pred"]

        metrics = compute_metrics(y_true, y_pred, encoding=encoding, aux_name="test")
        print("MV2H: ", metrics["mv2h"])
        print("SER: ", metrics["ser"])
        print("Recon-SER: ", metrics["recon_ser"])
