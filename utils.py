import os
import config
import numpy as np
import torch
from music21 import converter as converterm21
import shutil
from pyMV2H.converter.midi_converter import MidiConverter as Converter
from pyMV2H.utils.music import Music
from pyMV2H.utils.mv2h import MV2H
from pyMV2H.metrics.mv2h import mv2h

def ctc_greedy_decoder(y_pred: torch.Tensor, xl: tuple, i2w: dict) -> list:
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

def compute_ser(y_true: list, y_pred: list) -> float:
    ed_acc = 0
    length_acc = 0
    for t, h in zip(y_true, y_pred):
        ed_acc += levenshtein(t, h)
        length_acc += len(t)
    return 100. * ed_acc / length_acc

def write_plot_results(plotfile, results):
    with open(plotfile, "w") as datfile:
        datfile.write(f"{np.mean(results)}\t{np.std(results)}\n")


def retrieve_krn(in_seq: list) -> list:
    out_seq = list()
    it = 0
    while it < len(in_seq):
        if in_seq[it].startswith('*') or in_seq[it] == '=':
            out_seq.append(in_seq[it])
            it += 1
        else:
            new_token = ''
            # Duration:
            extract_duration = False
            while not extract_duration and it < len(in_seq):
                try:
                    int(in_seq[it].split(".")[0])
                    new_token = in_seq[it]
                    extract_duration = True
                except:
                    pass
                it += 1

            # Pitch:
            extract_pitch = False
            while not extract_pitch and it < len(in_seq):
                if list(set(in_seq[it].lower()))[0] in ['a','b','c','d','e','f','g', 'r']:
                    new_token += in_seq[it]
                    extract_pitch = True
                it += 1

            # Alteration:
            if it < len(in_seq):
                if in_seq[it] == '-' or in_seq[it] == '#':
                    new_token += in_seq[it]
                    it += 1

                out_seq.append(new_token)

    return out_seq




def compute_MV2H(y_true: list, y_pred: list, encoding: str):

    MV2H_global = MV2H(multi_pitch = 0, voice = 0, meter = 0, harmony = 0, note_value = 0)
    y_true_all = list()
    y_pred_all = list()
    for it in range(len(y_true)):
        # Converting to MIDI:
        ### True:
        if encoding == 'kern':
            y_true_krn = y_true[it]
        elif encoding == 'decoupled_dot':
            y_true_krn = retrieve_krn(y_true[it])
        y_true_krn.insert(0, '**kern')
        y_true_all.append(y_true_krn)
        with open('y_true.krn','w') as fout:
            for u in y_true_krn: fout.write(u + '\n')
        a = converterm21.parse('y_true.krn').write('midi')
        shutil.copyfile(a, 'y_true.mid')

        ### Pred:
        if encoding == 'kern':
            y_pred_krn = y_pred[it]
        elif encoding == 'decoupled_dot':
            y_pred_krn = retrieve_krn(y_pred[it])
        y_pred_krn.insert(0, '**kern')
        y_pred_all.append(y_pred_krn)
        with open('y_pred.krn','w') as fout:
            for u in y_pred_krn: fout.write(u + '\n')
        a = converterm21.parse('y_pred.krn').write('midi')
        shutil.copyfile(a, 'y_pred.mid')

        # Converting to TXT:
        ### True:
        reference_midi_file = 'y_true.mid'
        reference_file = 'y_true.txt'
        converter = Converter(file=reference_midi_file, output=reference_file)
        converter.convert_file()
        with open('y_true.txt','r') as fin:
            f = [u.replace(".0", "") for u in fin.readlines()]
        with open('y_true.txt','w') as fout:
            for u in f: fout.write(u)

        ### Pred:
        reference_midi_file = 'y_pred.mid'
        reference_file = 'y_pred.txt'
        converter = Converter(file=reference_midi_file, output=reference_file)
        converter.convert_file()
        with open('y_pred.txt','r') as fin:
            f = [u.replace(".0", "") for u in fin.readlines()]
        with open('y_pred.txt','w') as fout:
            for u in f: fout.write(u)

        # Figures of merit:
        reference_file = Music.from_file('y_true.txt')
        transcription_file = Music.from_file('y_pred.txt')

        res_dict = MV2H(multi_pitch = 0, voice = 0, meter = 0, harmony = 0, note_value = 0)
        try:
            res_dict = mv2h(reference_file, transcription_file)
            MV2H_global.__multi_pitch__  += res_dict.multi_pitch
            MV2H_global.__voice__ += res_dict.voice
            MV2H_global.__meter__ += res_dict.meter
            MV2H_global.__harmony__ += res_dict.harmony
            MV2H_global.__note_value__ += res_dict.note_value
        except:
            pass
        for u in [u for u in os.listdir() if u.endswith('.txt')]: os.remove(u)
        for u in [u for u in os.listdir() if u.endswith('.mid')]: os.remove(u)
        for u in [u for u in os.listdir() if u.endswith('.krn')]: os.remove(u)

    MV2H_global.__multi_pitch__  /= len(y_true)
    MV2H_global.__voice__ /= len(y_true)
    MV2H_global.__meter__ /= len(y_true)
    MV2H_global.__harmony__ /= len(y_true)
    MV2H_global.__note_value__ /= len(y_true)

    # Computer SER for the obtained kern after the transducer:
    ser = compute_ser(y_true_all, y_pred_all)

    return MV2H_global, ser



if __name__ == '__main__':
    Y = [['*clefG2','*k[b-]','*M6/8','*MM120','=','4.','f','4.','b','-','=','4.','g','4.','b','-','=','8','e','8','g','8','b','-','8','f','8','a','8','b','-','=','4.','cc','4','dd','8','ee','=','4','b','-','8','dd','4','cc','8','bb','-','=','8','ccc','8','bb','-','8','aa','4.','dd','=','8','r','8','dd','8','r','8','dd','8','r','8','dd','=','8','r','8','ee','8','r','8','ee','8','r','8','ee','=','2.','ccc','='],\
    ['*clefG2','*k[b-e-a-d-g-c-f-]','*M6/8','*MM90','=','4.','aa','-','4.','fff','-','=','4.','ccc','-','4.','gg','-','=','4','cc','-','8','ee','-','4','gg','-','8','bb','-','=','4.','ccc','-','4','ddd','-','8','gg','-','=','8','ccc','-','8','bb','-','8','aa','-','8','gg','-','8','ff','-','8','ee','-','=','8','aa','-','8','gg','-','8','ff','-','4.','bb','-','=','8','ccc','-','8','eee','-','8','aa','-','8','gg','-','8','aa','-','8','bb','-','=','4.','ccc','-','4.','aa','-','=','2.','ddd','-','=']]
    YPRED = [[], []]
    MV2H_res, ser = compute_MV2H(Y,YPRED, 'decoupled_dot')
    
    # in_seq = ['*clefG2', '*k[b-e-a-d-g-]', '*M3/4', '*MM120', '=', '4.', 'bb', '-', '16', 'ddd', '16', 'eee', '-', '=', '16', 'ccc', '16', 'aa', '4', 'gg', '16', 'gg', '16', 'aa', '16', 'ff', '16', 'cc', '8', 'cc', '8', 'dd', '4', 'r', '8', 'ee', '-', '8', 'gg', '4', 'r', '8', 'ff', '8', 'aa', '4', 'r', '=', '4', 'gg', '8', 'ccc', '16', 'ddd', '=', '2.', 'aa', '=']
    # out = retrieve_krn(in_seq)

    prueba = ['*clefG2', '*k[b-]', '*M4/4', '2', '=']
    out = retrieve_krn(prueba)

    print("hello")