import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(self.norm(out))
        return out


class Discriminator(nn.Module):
    """Discriminator for SRResNet
    https://arxiv.org/pdf/1609.04802v5.pdf
    """

    def __init__(self, image_size=512):
        super().__init__()
        blocks = [
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        blocks.extend(
            [
                ConvBlock(*arg)
                for arg in [
                    (64, 64, 3, 2, 1),
                    (64, 128, 3, 1, 1),
                    (128, 128, 3, 2, 1),
                    (128, 256, 3, 1, 1),
                    (256, 256, 3, 2, 1),
                    (256, 512, 3, 1, 1),
                    (512, 512, 3, 2, 1),
                ]
            ]
        )

        self.features = nn.Sequential(*blocks)

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(image_size * image_size * 2, image_size * 2)
        self.fc2 = nn.Linear(image_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        out = self.features(input)
        out = out.view(out.size(0), -1)

        out = self.LeakyReLU(self.fc1(out))
        out = self.sigmoid(self.fc2(out))

        return out.view(-1, 1).squeeze(1)
