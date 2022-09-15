import time, random

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F

from data import preprocess_audio, preprocess_label, IMG_HEIGHT
from utils import ctc_greedy_decoder, compute_ser

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        layers = [
            # First block
            nn.Conv2d(1, 8, (10, 2), padding="same", bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # Second block
            nn.Conv2d(8, 8, (8, 5), padding="same", bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 1)),
        ]
        self.backbone = nn.Sequential(*layers)
        self.width_reduction = 2
        self.height_reduction = 2**2
        self.out_channels = 8

    def forward(self, x):
        return self.backbone(x)

class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()
        self.blstm = nn.LSTM(input_size, 256, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(256 * 2, output_size)
    
    def forward(self, x):
        b, c, h, w = x.size()
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(b, w, h * c)
        x, _ = self.blstm(x)
        x = self.dropout(x)
        return self.output(x)
    
class CRNN(nn.Module):
    def __init__(self, vocab_size):
        super(CRNN, self).__init__()
        self.encoder = CNN()
        self.decoder = RNN(
            input_size=(IMG_HEIGHT // self.encoder.height_reduction) * self.encoder.out_channels,
            output_size=vocab_size
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

# ----------------------------------------------------------------

class CTCTrainedCRNN():
    def __init__(self, dictionaries, device):
        super(CTCTrainedCRNN, self).__init__()
        self.w2i, self.i2w = dictionaries

        self.device = device

        self.model = CRNN(vocab_size=len(self.w2i) + 1) # +1 for the CTC blank!
        self.compile()
        self.model.to(self.device)

        # Print summary
        summary(self.model, input_size=[1, 1, IMG_HEIGHT, 256])

    def compile(self):
        self.optimizer = torch.optim.Adam(self.model.parameters())
        # The target index cannot be blank!
        self.compute_ctc_loss = nn.CTCLoss(blank=len(self.w2i))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path, map_location):
        self.model.load_state_dict(torch.load(path, map_location=map_location))

    def on_train_begin(self, patience):
        self.logs = {"loss": [], "val_ser": []}
        self.best_ser = np.Inf
        self.best_epoch = 0
        self.patience = patience
    
    def train_step(self, data):
        x, xl, y, yl = data
        self.optimizer.zero_grad()
        ypred = self.model(x)
        # ------ CTC Requirements ------
        ypred = F.log_softmax(ypred, dim=-1)
        ypred = ypred.permute(1, 0, 2)
        # ------------------------------
        loss = self.compute_ctc_loss(ypred, y, xl, yl)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()

    def fit(self, train_data_generator, epochs, steps_per_epoch, val_data, test_data, batch_size, patience, weights_path, logs_path):
        XFVal, YFVal = val_data
        # Preprocess val data and leave in RAM
        XVal, XLVal = zip(*[preprocess_audio(xf, self.model.encoder.width_reduction) for xf in XFVal])
        YVal = [preprocess_label(yf, training=False, w2i=self.w2i) for yf in YFVal]

        self.on_train_begin(patience)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            start = time.time()

            # Training
            self.model.train()
            for _ in range(steps_per_epoch):
                data = next(train_data_generator)
                loss = self.train_step(data)
            self.logs["loss"].append(loss)

            # Validating
            self.model.eval()
            val_ser = self.evaluate(X=XVal, XL=XLVal, Y=YVal, batch_size=batch_size, print_ser=False)
            self.logs["val_ser"].append(val_ser)

            end = time.time()   
            print(f"loss: {loss:.4f} - val_ser: {val_ser:.2f} - {round(end-start)}s")

            # Early stopping
            if val_ser < self.best_ser:
                print(f"SER improved from {self.best_ser:.2f} to {val_ser:.2f}. Saving model's weights to {weights_path}")
                self.best_ser = val_ser
                self.best_epoch = epoch
                self.patience = patience
                self.save(path=weights_path)
            else:
                self.patience -= 1
                if self.patience == 0:
                    print(f"Stopped by early stopping on epoch: {epoch + 1}")
                    break

        # Testing
        # Preprocess test data and leave in RAM
        XFTest, YFTest = test_data
        XTest, XLTest = zip(*[preprocess_audio(xf, self.model.encoder.width_reduction) for xf in XFTest])
        YTest = [preprocess_label(yf, training=False, w2i=self.w2i) for yf in YFTest]
        # Loading best weights
        self.load(path=weights_path, map_location=self.device)
        self.model.eval()
        print("Evaluating best validation model over test data")
        test_ser = self.evaluate(X=XTest, XL=XLTest, Y=YTest, batch_size=batch_size, print_random_samples=True)
        self.logs["test_ser"] = test_ser

        # Save logs
        print(f"Saving experiment's logs to {logs_path}")
        self.save_logs(logs_path)

        return test_ser

    def evaluate(self, X, XL, Y, batch_size, print_ser=True, print_random_samples=False):
        YPRED = []

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                x, xl = X[start:start + batch_size], XL[start:start + batch_size]
                # Zero-pad images to maximum batch image width
                max_width = max(x, key=np.shape).shape[2]
                x = np.array([np.pad(i, pad_width=((0, 0), (0, 0), (0, max_width - i.shape[2]))) for i in x], dtype=np.float32)
                x = torch.from_numpy(x).to(self.device)
                # Obtain predictions
                ypred = self.model(x).detach().cpu()
                # CTC-greedy decoder
                YPRED.extend(ctc_greedy_decoder(ypred, xl, self.i2w))

        ser = compute_ser(Y, YPRED)

        if print_ser:
            print(f"SER (%): {ser:.2f} - From {len(Y)} samples")

        if print_random_samples:
            index = random.randint(0, len(Y) - 1)
            print(f"Prediction - {YPRED[index]}")
            print(f"Ground truth - {Y[index]}")

        return ser

    def save_logs(self, path):
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

    model = CTCTrainedCRNN(dictionaries=(dict(), dict()), device=torch.device("cuda"))
    