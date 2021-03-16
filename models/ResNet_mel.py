import torch
import torch.nn as nn
import models.Time_distributed


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, data_in, data_out, block, num_block, num_classes=100):
        super().__init__()
        self.data_in = data_in
        self.data_out = data_out
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.data_in[-3], 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different input size than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((60, 2))

        self.conv6 = nn.Conv2d(512, 64, kernel_size=3, padding=1, bias=False)

        self.rnn1 = nn.Sequential(
            nn.GRU(input_size=304, bidirectional=True, dropout=0.5, hidden_size=152,
                   batch_first=True)
        )

        self.rnn2 = nn.Sequential(
            nn.GRU(input_size=304, bidirectional=True, dropout=0.5, hidden_size=152,
                   batch_first=True)
        )

        self.doa = nn.Sequential(
            models.Time_distributed.TimeDistributed(nn.Linear(128, 128), batch_first=True),
            models.Time_distributed.TimeDistributed(nn.Linear(128, self.data_out[1][-1]), batch_first=True),
            nn.Tanh()
        )

        self.sed = nn.Sequential(
            models.Time_distributed.TimeDistributed(nn.Linear(128, 128), batch_first=True),
            models.Time_distributed.TimeDistributed(nn.Linear(128, self.data_out[0][-1]), batch_first=True),
            nn.Sigmoid()
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = self.conv6(output)
        output = output.permute(0, 2, 1, 3)

        output = torch.reshape(output, (output.shape[0], self.data_out[0][-2], -1))

        output, h = self.rnn1(output)
        output, h = self.rnn2(output)

        doa_out = self.doa(output)

        sed_out = self.sed(output)

        return sed_out, doa_out


def get_resnet(data_in, data_out):
    return ResNet(data_in, data_out, BasicBlock, [2, 2, 2, 2])
