import torch
import torch.nn as nn
from numpy import concatenate

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        #
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=3):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),  #反卷积
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Block2(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, (2, 1), bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, (2, 1), bias=False),  #反卷积
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Block3(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, (1, 0), bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, (1, 0), bias=False),  # 反卷积
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class YNet(nn.Module):
    def __init__(self, in_channels=1, features=16):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block2(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block3(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block2(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        # YNet
        self.initial_down2 = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down11 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down22 = Block2(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down33 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down44 = Block3(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down55 = Block2(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down66 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)

        self.up2 = Block2(
            features * 8 * 3, features * 8, down=False, act="relu", use_dropout=True
        )

        self.up3 = Block3(
            features * 8 * 3, features * 8, down=False, act="relu", use_dropout=True
        )

        self.up4 = Block(
            features * 8 * 3, features * 4, down=False, act="relu", use_dropout=False
        )

        self.up5 = Block2(
            features * 4 * 3, features * 2, down=False, act="relu", use_dropout=False
        )

        self.up6 = Block(
            features * 2 * 3, features, down=False, act="relu", use_dropout=False
        )

        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 3, 1, kernel_size=4, stride=2, padding=1),

        )

    def forward(self, x1, x2):

        d1 = self.initial_down(x1)

        d2 = self.down1(d1)

        d3 = self.down2(d2)

        d4 = self.down3(d3)

        d5 = self.down4(d4)

        d6 = self.down5(d5)

        bottleneck1 = self.bottleneck1(d6)

        d11 = self.initial_down2(x2)

        d22 = self.down11(d11)

        d33 = self.down22(d22)

        d44 = self.down33(d33)

        d55 = self.down44(d44)

        d66 = self.down55(d55)

        bottleneck2 = self.bottleneck2(d66)

        bottleneck = torch.cat([bottleneck1, bottleneck2], 1)

        up1 = self.up1(bottleneck)

        up2 = self.up2(torch.cat([up1, d6, d66], 1))

        up3 = self.up3(torch.cat([up2, d5, d55], 1))

        up4 = self.up4(torch.cat([up3, d4, d44], 1))

        up5 = self.up5(torch.cat([up4, d3, d33], 1))

        up6 = self.up6(torch.cat([up5, d2, d22], 1))

        return self.final_up(torch.cat([up6, d1, d11], 1))


def test():
    x1 = torch.randn((1, 1, 1080, 1440))
    x2 = torch.randn((1, 1, 1080, 1440))
    model = YNet(in_channels=1, features=64)
    preds = model(x1,x2)
    print(preds.shape)


if __name__ == "__main__":
    test()
