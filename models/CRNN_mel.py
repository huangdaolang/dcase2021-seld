import torch
import torch.nn as nn
import models.Time_distributed
from conformer import ConformerBlock


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

        self.conformer1 = nn.Sequential(
            ConformerBlock(dim=128, dim_head=64)
        )

        self.conformer2 = nn.Sequential(
            ConformerBlock(dim=128, dim_head=64)
        )

        self.conformer3 = nn.Sequential(
            ConformerBlock(dim=128, dim_head=64)
        )

        self.conformer4 = nn.Sequential(
            ConformerBlock(dim=128, dim_head=64)
        )

        self.conformer5 = nn.Sequential(
            ConformerBlock(dim=128, dim_head=64)
        )

        self.conformer6 = nn.Sequential(
            ConformerBlock(dim=128, dim_head=64)
        )

        self.conformer7 = nn.Sequential(
            ConformerBlock(dim=128, dim_head=64)
        )

        self.conformer8 = nn.Sequential(
            ConformerBlock(dim=128, dim_head=64)
        )
        self.doa = nn.Sequential(
            models.Time_distributed.TimeDistributed(nn.Linear(128, 128), batch_first=True),
            models.Time_distributed.TimeDistributed(nn.Linear(128, 36), batch_first=True),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.permute(0, 2, 1, 3)
        # print(out.shape)
        out = torch.reshape(out, (out.shape[0], 60, -1))

        out = self.conformer1(out)
        # out = self.conformer2(out)
        # out = self.conformer3(out)
        # out = self.conformer4(out)
        # out = self.conformer5(out)
        # out = self.conformer6(out)
        # out = self.conformer7(out)
        # out = self.conformer8(out)

        out = self.doa(out)

        return out


