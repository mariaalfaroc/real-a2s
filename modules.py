import torch
import torch.nn as nn

from data import IMG_HEIGHT, NUM_CHANNELS


class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        layers = [
            # First block
            nn.Conv2d(NUM_CHANNELS, 8, (10, 2), padding="same", bias=False),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class RNN(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(RNN, self).__init__()
        self.input_size = input_size
        self.blstm = nn.LSTM(
            self.input_size,
            256,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(256 * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.size()
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.reshape(b, -1, self.input_size)
        x, _ = self.blstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class CRNN(nn.Module):
    def __init__(self, output_size: int) -> None:
        super(CRNN, self).__init__()
        # CNN
        self.encoder = CNN()
        # RNN
        input_size = self.encoder.out_channels * (
            IMG_HEIGHT // self.encoder.height_reduction
        )
        self.decoder = RNN(input_size=input_size, output_size=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
