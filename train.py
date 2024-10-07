from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import SRResNet, Discriminator
from dataset import DIV2K
from tools import vanilla_gen_step, vanilla_discr_step, ns_gen_step, ns_discr_step

EXP_NAME = "vanilla_SRResNet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_vanilla(
    train_loader: DataLoader,
    val_loader: DataLoader,
    G: nn.Module,
    D: nn.Module,
    G_optim: torch.optim.Optimizer,
    D_optim: torch.optim.Optimizer,
    discriminator_steps: int,
    n_epochs: int,
    eval_steps: int = 10,
) -> None:

    G.train()
    D.train()
    step_i = 0
    for epoch_i in tqdm(range(n_epochs)):
        for batch_i, data in enumerate(train_loader):
            low_res_batch, high_res_batch = data

            low_res_batch = low_res_batch.to(DEVICE)
            high_res_batch = high_res_batch.to(DEVICE)

            d_loss = vanilla_discr_step(low_res_batch, high_res_batch, G, D, D_optim)

            if step_i % discriminator_steps == 0:
                g_loss = vanilla_gen_step(low_res_batch, G, D, G_optim)

            print(f"discr loss: {d_loss:.5f} | gen loss: {g_loss:.5f}")

            step_i += 1

        if eval_steps and epoch_i % eval_steps == 0:
            # TODO: implement evaluation
            G.eval()
            pass

    torch.save(G.state_dict(), f"weights/G_{EXP_NAME}.pth")
    torch.save(D.state_dict(), f"weights/D_{EXP_NAME}.pth")


def train_ns(
    train_loader: DataLoader,
    val_loader: DataLoader,
    G: nn.Module,
    D: nn.Module,
    G_optim: torch.optim.Optimizer,
    D_optim: torch.optim.Optimizer,
    discriminator_steps: int,
    n_epochs: int,
    r1_regularizer: float = 1.0,
    eval_steps: int = 10,
) -> None:

    G.train()
    D.train()
    step_i = 0
    for epoch_i in tqdm(range(n_epochs)):
        for batch_i, data in enumerate(train_loader):
            low_res_batch, high_res_batch = data

            low_res_batch = low_res_batch.to(DEVICE)
            high_res_batch = high_res_batch.to(DEVICE)

            d_loss = ns_discr_step(
                low_res_batch, high_res_batch, G, D, D_optim, r1_regularizer
            )

            if step_i % discriminator_steps == 0:
                g_loss = ns_gen_step(low_res_batch, G, D, G_optim)

            step_i += 1

            print(f"discr loss: {d_loss:.5f} | gen loss: {g_loss:.5f}")

        if eval_steps and epoch_i % eval_steps == 0:
            # TODO: implement evaluation
            G.eval()
            pass

    torch.save(G.state_dict(), f"weights/G_{EXP_NAME}.pth")
    torch.save(D.state_dict(), f"weights/D_{EXP_NAME}.pth")


def main():
    low_res_size = 64
    high_res_size = 512
    scale_factor = int(math.log2(high_res_size // low_res_size))

    div2k_train = DIV2K(
        "data/DIV2K_train_HR", low_res_size=low_res_size, high_res_size=high_res_size
    )
    div2k_val = DIV2K(
        "data/DIV2K_valid_HR", low_res_size=low_res_size, high_res_size=high_res_size
    )

    train_loader = DataLoader(div2k_train, batch_size=4, shuffle=True)
    val_loader = DataLoader(div2k_val, batch_size=4, shuffle=True)

    G = SRResNet(scale_factor=scale_factor).to(DEVICE)
    D = Discriminator(image_size=high_res_size).to(DEVICE)

    G_optim = torch.optim.RMSprop(G.parameters(), lr=2e-4)
    D_optim = torch.optim.RMSprop(D.parameters(), lr=2e-4)

    # train_vanilla(
    #     train_loader,
    #     val_loader,
    #     G,
    #     D,
    #     G_optim,
    #     D_optim,
    #     discriminator_steps=1,
    #     n_epochs=10
    # )

    train_ns(
        train_loader,
        val_loader,
        G,
        D,
        G_optim,
        D_optim,
        discriminator_steps=4,
        r1_regularizer=0.1,
        n_epochs=10,
    )


if __name__ == "__main__":
    main()
