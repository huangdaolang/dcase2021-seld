import torch
import torch.nn as nn
import models.Time_distributed


class SampleCNN(nn.Module):
    def __init__(self, params):
        super(SampleCNN, self).__init__()

        self.params = params

        # 59049 x 4
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # 19683 x 128
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 6561 x 128
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 2187 x 128
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 729 x 256
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 243 x 256
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Dropout(self.params.dropout_rate))
        # 81 x 256
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 60, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))

        # self.rnn1 = nn.Sequential(
        #     nn.GRU(input_size=65, bidirectional=True, dropout=self.params.dropout_rate, hidden_size=64,
        #            batch_first=True)
        # )
        #
        # self.rnn2 = nn.Sequential(
        #     nn.GRU(input_size=self.params.rnn_size[1], bidirectional=True, dropout=self.params.dropout_rate, hidden_size=64,
        #            batch_first=True)
        # )

        self.doa = nn.Sequential(
            models.Time_distributed.TimeDistributed(nn.Linear(65, self.params.fnn_size[0]), batch_first=True),
            nn.Dropout(self.params.dropout_rate),
            models.Time_distributed.TimeDistributed(nn.Linear(self.params.fnn_size[0], self.params.data_out[1][-1]), batch_first=True),
            nn.Tanh()
        )

        self.sed = nn.Sequential(
            models.Time_distributed.TimeDistributed(nn.Linear(65, self.params.fnn_size[0]), batch_first=True),
            nn.Dropout(self.params.dropout_rate),
            models.Time_distributed.TimeDistributed(nn.Linear(self.params.fnn_size[0], self.params.data_out[0][-1]), batch_first=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width

        x = x.view(x.shape[0], 4, -1)
        # x : 23 x 1 x 59049
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        # out, h = self.rnn1(out)
        # out, h = self.rnn2(out)
        doa_out = self.doa(out)

        sed_out = self.sed(out)

        return sed_out, doa_out
