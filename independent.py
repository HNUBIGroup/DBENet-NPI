
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,),
                 dilation=(1,), if_bias=False, relu=True, same_padding=True, bn=True):
        super(Conv1d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p0,
                              dilation=dilation, bias=True if if_bias else False)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        x = F.dropout(x, 0.2)
        return x


class HMCN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HMCN, self).__init__()

        self.conv0 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=True),
        )

        self.conv1 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(3,), same_padding=True),
        )

        self.conv2 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(5,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(5,), same_padding=True),
        )

        self.conv3 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
        )
        self.linear = nn.Linear(in_channel, out_channel * 4)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = torch.cat([x0, x1, x2, x3], dim=1)

        x = x.squeeze(2)
        x = self.linear(x)
        x = x.unsqueeze(2)
        x = x + x4

        return x
