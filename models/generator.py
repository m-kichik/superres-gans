import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, activation):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(num_features=channels),
            activation(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(num_features=channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


class UpsampleBlock(nn.Module):
    def __init__(self, n_upsamples: int, channels: int, kernel_size: int, activation):
        super().__init__()

        layers = []
        for _ in range(n_upsamples):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels * 2**2,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    nn.PixelShuffle(2),
                    activation(),
                ]
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SRResNet(nn.Module):
    """Super-Resolution Residual Neural Network
    https://arxiv.org/pdf/1609.04802v5.pdf
    """

    def __init__(self, n_upsamples: int, channels: int = 3, num_blocks: int = 16):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=9,
                stride=1,
                padding=4,
            ),
            nn.PReLU(),
        )

        self.res_blocks = [
            ResidualBlock(channels=64, kernel_size=3, activation=nn.PReLU)
            for _ in range(num_blocks)
        ]
        self.res_blocks.append(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.res_blocks.append(nn.BatchNorm2d(num_features=64))
        self.res_blocks = nn.Sequential(*self.res_blocks)

        self.upsample = UpsampleBlock(
            n_upsamples=n_upsamples,
            channels=64,
            kernel_size=3,
            activation=nn.PReLU,
        )

        self.tail = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=channels,
                kernel_size=9,
                stride=1,
                padding=4,
            ),
            # nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = x + self.res_blocks(x)
        x = self.upsample(x)
        x = self.tail(x)
        return x
