import time
import random
import pathlib
from typing import Tuple, Generator, List

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torchinfo import summary
from madmom.audio.spectrogram import LogarithmicFilteredSpectrogram

from network.modules import CRNN
from my_utils.data import preprocess_audio, preprocess_label, IMG_HEIGHT, NUM_CHANNELS
from my_utils.utils import ctc_greedy_decoder, compute_metrics


class CTCTrainedCRNN:
    def __init__(
        self, dictionaries: Tuple[dict, dict], encoding: str, device: torch.device
    ) -> None:
        super(CTCTrainedCRNN, self).__init__()
        self.w2i, self.i2w = dictionaries

        self.device = device
        self.encoding = encoding

        self.model = CRNN(output_size=len(self.w2i) + 1)  # +1 for the CTC blank!
        self.model.to(self.device)
        self.compile()
        self.summary()

    def summary(self) -> None:
        summary(self.model, input_size=[1, NUM_CHANNELS, IMG_HEIGHT, 256])

    def updateModel(self, list_update_elements: list) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and all(
                [u not in name for u in list_update_elements]
            ):
                param.requires_grad = False
            print(f"{name} -> Update: {param.requires_grad}")

    def compile(self) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.compute_ctc_loss = nn.CTCLoss(
            blank=len(self.w2i), zero_infinity=True
        )  # The target index cannot be blank!

    def save(self, path: pathlib.PosixPath) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: pathlib.PosixPath) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def on_train_begin(self, patience: int) -> None:
        self.logs = {"loss": [], "val_ser": [], "val_mv2h": [], "val_recon_ser": []}
        self.best_val_ser = float("Inf")
        self.best_epoch = 0
        self.patience = patience

    def train_step(
        self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        x, xl, y, yl = data
        self.optimizer.zero_grad()
        ypred = self.model(x)
        # ------ CTC Requirements ------
        # yhat: [batch, frames, features]
        ypred = ypred.log_softmax(dim=-1)
        ypred = ypred.permute(1, 0, 2)
        # ------------------------------
        loss = self.compute_ctc_loss(ypred, y, xl, yl)
        loss.backward()
        self.optimizer.step()
        return loss

    def fit(
        self,
        train_data_generator: Generator[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None
        ],  # Generator[yield_type, send_type, return_type],
        epochs: int,
        steps_per_epoch: int,
        val_data: Tuple[List[str], List[str]],
        test_data: Tuple[List[str], List[str]],
        batch_size: int,
        patience: int,
        weights_path: pathlib.PosixPath,
        logs_path: pathlib.PosixPath,
    ) -> None:
        # Preprocess val data and leave in RAM
        XFVal, YFVal = val_data
        XVal, XLVal = map(
            list,
            zip(
                *[
                    preprocess_audio(xf, self.model.encoder.width_reduction)
                    for xf in XFVal
                ]
            ),
        )
        YVal = [
            preprocess_label(yf, training=False, w2i=self.w2i, encoding=self.encoding)
            for yf in YFVal
        ]

        self.on_train_begin(patience=patience)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}", end="", flush=True)

            start = time.time()

            # Training
            self.model.train()
            for _ in range(steps_per_epoch):
                data = next(train_data_generator)
                loss = self.train_step(data)
            loss = loss.detach().cpu().item()
            self.logs["loss"].append(loss)
            print(f" - loss: {loss:.4f}", end="", flush=True)

            # Validating
            self.model.eval()
            metrics = self.evaluate(
                X=XVal,
                XL=XLVal,
                Y=YVal,
                batch_size=batch_size,
                encoding=self.encoding,
                aux_name="",
                print_metrics=False,
            )
            for k, v in metrics.items():
                self.logs[f"val_{k}"].append(v)
                print(f" - val_{k}: {v:.2f}", end="", flush=True)

            end = time.time()

            print(f" - {round(end - start)}s")

            # Early stopping
            if metrics["ser"] < self.best_val_ser:
                print(
                    f"SER performance improved from {self.best_val_ser:.2f} to {metrics['ser']:.2f}"
                )
                print(f"Saving model's weights to {weights_path}")
                self.best_val_ser = metrics["ser"]
                self.best_epoch = epoch
                self.patience = patience
                self.save(path=weights_path)
            else:
                self.patience -= 1
                if self.patience == 0:
                    print(f"Stopped by early stopping on epoch: {epoch + 1}")
                    break

        # Testing
        # Loading best weights
        self.load(path=weights_path)
        self.model.eval()
        print(
            f"Evaluating best validation model (at epoch {self.best_epoch}) over test data"
        )
        # Preprocess test data and leave in RAM
        XFTest, YFTest = test_data
        XTest, XLTest = map(
            list,
            zip(
                *[
                    preprocess_audio(xf, self.model.encoder.width_reduction)
                    for xf in XFTest
                ]
            ),
        )
        YTest = [
            preprocess_label(yf, training=False, w2i=self.w2i, encoding=self.encoding)
            for yf in YFTest
        ]
        # Evaluate
        metrics = self.evaluate(
            X=XTest,
            XL=XLTest,
            Y=YTest,
            batch_size=batch_size,
            encoding=self.encoding,
            aux_name=str(weights_path).split("/")[-2],
            print_random_samples=True,
        )
        for k, v in metrics.items():
            self.logs[f"test_{k}"] = v

        # Save logs
        print(f"Saving experiment's logs to {logs_path}")
        self.save_logs(logs_path)

    def evaluate(
        self,
        X: List[LogarithmicFilteredSpectrogram],
        XL: List[int],
        Y: List[List[str]],
        batch_size: int,
        encoding: str,
        aux_name: str,
        print_metrics: bool = True,
        print_random_samples: bool = False,
    ) -> dict:
        YPRED = []

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                x, xl = X[start : start + batch_size], XL[start : start + batch_size]
                # Zero-pad audios to maximum batch audio width
                max_width = max(x, key=np.shape).shape[2]
                x = np.array(
                    [
                        np.pad(
                            i, pad_width=((0, 0), (0, 0), (0, max_width - i.shape[2]))
                        )
                        for i in x
                    ],
                    dtype=np.float32,
                )
                x = torch.from_numpy(x).to(self.device)
                # Obtain predictions
                ypred = self.model(x).detach().cpu()
                # CTC-greedy decoder
                YPRED.extend(ctc_greedy_decoder(ypred, xl, self.i2w))

        metrics = compute_metrics(Y, YPRED, encoding=encoding, aux_name=aux_name)

        if print_metrics:
            for k, v in metrics.items():
                print(f"{k.upper()} (%): {v:.2f} - ", end="", flush=True)
            print(f"From {len(Y)} samples")

        if print_random_samples:
            index = random.randint(0, len(Y) - 1)
            print(f"Prediction - {YPRED[index]}")
            print(f"Ground truth - {Y[index]}")

        return metrics

    def save_logs(self, path: pathlib.PosixPath) -> None:
        # The last line on the CSV file is the one corresponding to the best validation model
        for k in self.logs.keys():
            if "test" not in k:
                self.logs[k].extend(["-", self.logs[k][self.best_epoch]])
            else:
                self.logs[k] = ["-"] * len(self.logs["loss"][:-1]) + [self.logs[k]]
        df_logs = pd.DataFrame.from_dict(self.logs)
        df_logs.to_csv(path, index=False)


if __name__ == "__main__":
    torch.cuda.empty_cache()

    model = CTCTrainedCRNN(
        dictionaries=({}, {}), encoding="kern", device=torch.device("cuda")
    )
