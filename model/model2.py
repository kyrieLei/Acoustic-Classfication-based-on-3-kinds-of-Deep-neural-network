import math
import torch
from torch import nn


# DSSDB
class Spatblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Spatblock, self).__init__()
        self.dilaconv1 = Sparable(in_channels, in_channels)
        self.dilaconv2 = Sparable(in_channels, in_channels)
        self.dilaconv3 = Sparable(in_channels, in_channels)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(3 * (in_channels), out_channels, kernel_size=1, stride=1)

        self.att = eca_layer(3 * in_channels , k_size=3)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x1 = self.dilaconv1(x)
        x2 = self.dilaconv2(x1)
        x3 = self.dilaconv3(x2)
        o1 = self.conv1(x)
        o2 = self.conv2(x1)
        o3 = torch.cat((o1, o2, x3), dim=1)
        o3 = self.att(o3)
        o3 = self.conv3(o3)
        o3 = self.norm(o3)
        o3 = self.relu(o3)
        o3 = self.pool(o3)

        return o3


class Sparable(nn.Module):
    def __init__(self, cin, cout, p=0.25, min_mid_channels=4):
        super(Sparable, self).__init__()

        assert 0 <= p <= 1
        self.ch_conv1 = nn.Conv2d(
            cin,
            cout,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=False,
        )

        # 1x3的网络
        self.sp_conv1 = nn.Conv2d(
            cout,
            cout,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 2),
            dilation=(1, 2),
            bias=False,
        )
        # 3X1的网络
        self.sp_conv2 = nn.Conv2d(
            cout,
            cout,
            kernel_size=(3, 1),
            stride=1,
            padding=(2, 0),
            dilation=(2, 1),
            bias=False,
        )

        self.norm = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.quantized.FloatFunctional()

    def forward(self, x):
        x = self.ch_conv1(x)
        x = self.shortcut.add(self.sp_conv1(x), self.sp_conv2(x))
        x = self.norm(x)
        return self.relu(x)


class eca_layer(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.shortcut = nn.quantized.FloatFunctional()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return self.shortcut.mul(x, y.expand_as(x))


class Cnn(nn.Module):
    def __init__(
            self,
            channels=[24, 32, 64, 64, 64, 64]
    ):
        super(Cnn, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.shortcut = nn.quantized.FloatFunctional()
        self.dequant = torch.quantization.DeQuantStub()

        self.conv = nn.Conv2d(1, 16, kernel_size=1)

        self.conv1 = Spatblock(16, channels[0])
        self.conv2 = Spatblock(channels[0], channels[1])
        self.conv3 = Spatblock(channels[1], channels[2])
        self.conv4 = Spatblock(channels[2], channels[3])
        self.conv5 = Spatblock(channels[3], channels[4])
        self.conv6 = Spatblock(channels[4], channels[5])

        self.pool=nn.AdaptiveAvgPool2d(1)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.3)
        self.flatten=nn.Flatten()
        self.l1=nn.Linear(224,100)
        self.l2=nn.Linear(100,3)

    def forward(self,x):
        """
                :param x:
                :return:
                """
        """Input size - (batch_size, 1, time_steps, mel_bins)  """

        x = self.quant(x)
        x = x.transpose(2, 3)
        x = self.conv(x)

        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)
        feat5 = self.conv5(feat4)
        # feat6 = self.conv6(feat5)

        m1 = self.pool(feat2)
        o1 = self.pool(feat3)
        o2 = self.pool(feat4)
        o3 = self.pool(feat5)

        x_1 = torch.cat((m1, o1, o2, o3), dim=1)

        x_1 = self.flatten(x_1)
        x_1 = self.relu(self.l1(x_1))
        x_1 = self.dropout(x_1)
        x_1 = self.l2(x_1)
        x_1 = self.dequant(x_1)

        return x_1
