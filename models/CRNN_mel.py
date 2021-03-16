import torch
import torch.nn as nn
import models.Time_distributed


class CRNN(nn.Module):
    def __init__(self, data_in, data_out, dropout_rate, nb_cnn2d_filt, f_pool_size, t_pool_size,
              rnn_size, fnn_size, doa_objective):
        super(CRNN, self).__init__()
        self.data_in = data_in
        self.data_out = data_out

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.data_in[-3], out_channels=nb_cnn2d_filt, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(nb_cnn2d_filt),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(t_pool_size[0], f_pool_size[0])),
            nn.Dropout(dropout_rate)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=nb_cnn2d_filt, out_channels=nb_cnn2d_filt, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(nb_cnn2d_filt),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(t_pool_size[1], f_pool_size[1])),
            nn.Dropout(dropout_rate)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=nb_cnn2d_filt, out_channels=nb_cnn2d_filt, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(nb_cnn2d_filt),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(t_pool_size[2], f_pool_size[2])),
            nn.Dropout(dropout_rate)
        )

        self.rnn1 = nn.Sequential(
            nn.GRU(input_size=rnn_size[0], bidirectional=True, dropout=dropout_rate, hidden_size=int(rnn_size[0]/2),
                   batch_first=True)
        )

        self.rnn2 = nn.Sequential(
            nn.GRU(input_size=rnn_size[1], bidirectional=True, dropout=dropout_rate, hidden_size=int(rnn_size[1]/2),
                   batch_first=True)
        )

        self.doa = nn.Sequential(
            models.Time_distributed.TimeDistributed(nn.Linear(128, fnn_size[0]), batch_first=True),
            nn.Dropout(dropout_rate),
            models.Time_distributed.TimeDistributed(nn.Linear(fnn_size[0], self.data_out[1][-1]), batch_first=True),
            nn.Tanh()
        )

        self.sed = nn.Sequential(
            models.Time_distributed.TimeDistributed(nn.Linear(128, fnn_size[0]), batch_first=True),
            nn.Dropout(dropout_rate),
            models.Time_distributed.TimeDistributed(nn.Linear(fnn_size[0], self.data_out[0][-1]), batch_first=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv1(x)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = self.conv3(out)
        # print(out.shape)
        out = out.permute(0, 2, 1, 3)
        # print(out.shape)
        out = torch.reshape(out, (out.shape[0], self.data_out[0][-2], -1))
        # print(out.shape)
        out, h = self.rnn1(out)
        out, h = self.rnn2(out)

        doa_out = self.doa(out)

        sed_out = self.sed(out)

        return sed_out, doa_out


