import torch
import torch.nn as nn
import models.Time_distributed


class SampleCNN(nn.Module):
    def __init__(self, params):
        super(SampleCNN, self).__init__()

        self.params = params

        # 144000 x 4
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # 48000 x 128
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 16000 x 128
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 5333 x 256
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 1777 x 256
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 592 x 256
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Dropout(self.params.dropout_rate))
        # 197 x 256
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 60, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # output: 65 x 60

        self.rnn1 = nn.Sequential(
            nn.GRU(input_size=65, bidirectional=True, hidden_size=64,
                   batch_first=True)
        )

        self.rnn2 = nn.Sequential(
            nn.GRU(input_size=128, bidirectional=True, hidden_size=64,
                   batch_first=True)
        )

        self.doa = nn.Sequential(
            models.Time_distributed.TimeDistributed(nn.Linear(128, 128), batch_first=True),
            nn.Dropout(self.params.dropout_rate),
            models.Time_distributed.TimeDistributed(nn.Linear(128, 42), batch_first=True),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.shape[0], 4, -1)

        out = self.conv1(x)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = self.conv3(out)
        # print(out.shape)
        out = self.conv4(out)
        # print(out.shape)
        out = self.conv5(out)
        # print(out.shape)
        out = self.conv6(out)
        # print(out.shape)
        out = self.conv7(out)
        # print(out.shape)
        # out = out.permute(0, 2, 1)
        out, h = self.rnn1(out)
        # print(out.shape)
        out, h = self.rnn2(out)
        # print(out.shape)
        out = self.doa(out)

        return out
