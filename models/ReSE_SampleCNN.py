import torch
import torch.nn as nn
import models.Time_distributed
from conformer import ConformerBlock
from torchsummary import summary
import parameter


class Basic_Block(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes):
        super(Basic_Block, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv_align = nn.Sequential(
            nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(planes)
        )

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.max_pooling = nn.MaxPool1d(3, stride=3)

        # SE block
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_down = nn.Conv1d(
            planes, planes // 16, kernel_size=1, bias=False)
        self.conv_up = nn.Conv1d(
            planes // 16, planes, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        if self.inplanes != self.planes:
            shortcut = self.conv_align(x)
        else:
            shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)

        res = out1 * out + shortcut
        res = self.relu(res)
        res = self.max_pooling(res)
        return res


class ReSE_SampleCNN(nn.Module):
    def __init__(self, params, block):
        super(ReSE_SampleCNN, self).__init__()

        self.params = params

        # 144000 x 4
        self.conv1 = nn.Sequential(
            nn.Conv1d(8, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())

        self.conv2 = block(128, 128)
        self.conv3 = block(128, 128)
        self.conv4 = block(128, 256)
        self.conv5 = block(256, 512)
        self.conv6 = block(512, 256)
        self.conv7 = block(256, 128)

        self.avgpool = nn.AdaptiveAvgPool1d(60)

        self.conformer1 = nn.Sequential(
            ConformerBlock(dim=128, dim_head=64)
        )
        self.conformer2 = nn.Sequential(
            ConformerBlock(dim=128, dim_head=64)
        )

        self.doa = nn.Sequential(
            models.Time_distributed.TimeDistributed(nn.Linear(128, 128), batch_first=True),
            nn.Dropout(self.params.dropout_rate),
            models.Time_distributed.TimeDistributed(nn.Linear(128, 42), batch_first=True),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.shape[0], 8, -1)

        out = self.conv1(x)

        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.avgpool(out)

        out = out.permute(0, 2, 1)
        out = self.conformer1(out)
        out = self.conformer2(out)
        out = self.doa(out)

        return out


if __name__ == "__main__":
    params = parameter.get_params()
    model = ReSE_SampleCNN(params, Basic_Block)
    summary(model, input_size=(8, 144000))