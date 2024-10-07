import torch

from model import SRResNet, Discriminator
from dataset import DIV2K


def main():
    # G = SRResNet(scale_factor=2)
    # D = Discriminator()

    # generated = G(torch.zeros((2, 3, 256, 256)))
    # print(D(generated))

    train_ds = DIV2K("data/DIV2K_train_HR")
    for item in train_ds:
        pass


if __name__ == "__main__":
    main()
