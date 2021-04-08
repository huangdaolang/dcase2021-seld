import torch
import torch.nn as nn
import models.Time_distributed


class SampleCNN(nn.Module):
    def __init__(self, params):
        super(SampleCNN, self).__init__()

        self.params = params

        # # 59049 x 4
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(4, 128, kernel_size=3, stride=3, padding=0),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU())
        # # 19683 x 128
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3, stride=3))
        # # 6561 x 128
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3, stride=3))
        # # 2187 x 128
        # self.conv4 = nn.Sequential(
        #     nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3, stride=3))
        # # 729 x 256
        # self.conv5 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3, stride=3))
        # # 197 x 256
        # self.conv6 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3, stride=3),
        #     nn.Dropout(self.params.dropout_rate))
        # # 60 x 128
        # self.conv7 = nn.Sequential(
        #     nn.Conv1d(256, 128, kernel_size=20, stride=1, padding=1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3, stride=3))

        # 144000 x 128
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # 48000 x 128
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Dropout(self.params.dropout_rate),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))
        # 24000 x 128
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Dropout(self.params.dropout_rate),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))
        # 12000 x 128
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.Dropout(self.params.dropout_rate),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))
        # 6000 x 256
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.Dropout(self.params.dropout_rate),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))
        # 3000 x 256
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.Dropout(self.params.dropout_rate),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))
        # 1500 x 256
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Dropout(self.params.dropout_rate),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=5))
        # 300 x 128
        self.conv8 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=5))
        # 60 x 128

        self.rnn1 = nn.Sequential(
            nn.GRU(input_size=128, bidirectional=True, hidden_size=128,
                   batch_first=True)
        )

        self.rnn2 = nn.Sequential(
            nn.GRU(input_size=256, bidirectional=True, hidden_size=128,
                   batch_first=True)
        )

        self.sed = nn.Sequential(
            models.Time_distributed.TimeDistributed(nn.Linear(256, 128), batch_first=True),
            nn.Dropout(self.params.dropout_rate),
            models.Time_distributed.TimeDistributed(nn.Linear(128, 14), batch_first=True),
            nn.Sigmoid()
        )

        self.doa = nn.Sequential(
            models.Time_distributed.TimeDistributed(nn.Linear(256, 128), batch_first=True),
            nn.Dropout(self.params.dropout_rate),
            models.Time_distributed.TimeDistributed(nn.Linear(128, 42), batch_first=True),
            nn.Tanh()
        )

    def forward(self, x):
        # input x : 32 x 144000 x 1
        # expected conv1d input : minibatch_size x num_channel x width

        x = x.view(x.shape[0], 4, -1)
        # x : 23 x 1 x 144000
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
        out = self.conv8(out)
        out = out.permute(0, 2, 1)
        # print(out.shape)
        out, h = self.rnn1(out)
        # print(out.shape)
        out, h = self.rnn2(out)
        # print(out.shape)
        doa_out = self.doa(out)

        sed_out = self.sed(out)

        return sed_out, doa_out
