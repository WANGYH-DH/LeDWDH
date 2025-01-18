import torch
import torch.nn as nn


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
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.initial_down1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down11 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down22 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down33 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down44 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down55 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down66 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 3, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 3, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 3, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
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
        bottleneck1 = self.bottleneck(d6)

        d11 = self.initial_down1(x2)
        d22 = self.down11(d11)
        d33 = self.down22(d22)
        d44 = self.down33(d33)
        d55 = self.down44(d44)
        d66 = self.down55(d55)
        bottleneck2 = self.bottleneck(d66)

        bottleneck = torch.cat([bottleneck1, bottleneck2], 1)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d6, d66], 1))
        up3 = self.up3(torch.cat([up2, d5, d55], 1))
        up4 = self.up4(torch.cat([up3, d4, d44], 1))
        up5 = self.up5(torch.cat([up4, d3, d33], 1))
        up6 = self.up6(torch.cat([up5, d2, d22], 1))
        return self.final_up(torch.cat([up6, d1, d11], 1))


def test():
    x = torch.randn((1, 1, 512, 512))
    y = torch.randn((1, 1, 512, 512))
    model = YNet(in_channels=1, features=64)
    preds = model(x, y)
    print(preds.shape)


if __name__ == "__main__":
    test()
