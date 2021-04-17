import torch
import torch.nn as nn
import models.Time_distributed


class CRNN(nn.Module):
    def __init__(self, dropout_rate):
        super(CRNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 4)),
            nn.Dropout(dropout_rate)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.rnn1 = nn.Sequential(
            nn.GRU(input_size=128, bidirectional=True, hidden_size=64,
                   batch_first=True)
        )

        self.rnn2 = nn.Sequential(
            nn.GRU(input_size=128, bidirectional=True, hidden_size=64,
                   batch_first=True)
        )

        self.doa = nn.Sequential(
            models.Time_distributed.TimeDistributed(nn.Linear(128, 128), batch_first=True),
            models.Time_distributed.TimeDistributed(nn.Linear(128, 42), batch_first=True),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.permute(0, 2, 1, 3)
        # print(out.shape)
        out = torch.reshape(out, (out.shape[0], 60, -1))

        out, h = self.rnn1(out)
        out, h = self.rnn2(out)

        out = self.doa(out)

        return out


