import numpy as np
import torch

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