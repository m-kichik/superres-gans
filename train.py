from tqdm import tqdm
import math
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips

from models import SRResNet, Discriminator
from dataset import DIV2K
from tools import (
    vanilla_gen_step,
    vanilla_discr_step,
    ns_gen_step,
    ns_discr_step,
    ns_mse_gen_step,
    ns_mse_vgg_gen_step,
    ns_mse_vgg_discr_step,
)

EXP_NAME = "ns_mse_vgg_discr4_without_sigmoid_ep100"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
gan_loss,
pixels_mse_loss,
vgg_mse_loss
"""
LOSS_WEIGHTS = [
        10e-3,
        0.5,
        0.5
    ]

# Initialize LPIPS model
lpips_loss = lpips.LPIPS(net="alex").to(DEVICE)


def evaluate(G, val_loader):
    G.eval()
    psnr_total = 0
    ssim_total = 0
    lpips_total = 0
    num_images = len(val_loader)

    with torch.no_grad():
        for low_res_batch, high_res_batch in val_loader:

            low_res_batch = low_res_batch.to(DEVICE)
            high_res_batch = high_res_batch.to(DEVICE)

            # Generate high-resolution images
            generated_batch = G(low_res_batch)

            # Calculate PSNR, SSIM, and LPIPS for each image in the batch
            for generated, target in zip(generated_batch, high_res_batch):

                generated_np = generated.cpu().numpy().transpose(1, 2, 0)  # CHW to HWC
                target_np = target.cpu().numpy().transpose(1, 2, 0)

                # Normalize to [0, 255] for PSNR and SSIM
                generated_np = np.clip(generated_np * 255.0, 0, 255).astype(np.uint8)
                target_np = np.clip(target_np * 255.0, 0, 255).astype(np.uint8)

                # Calculate PSNR
                psnr_val = cv2.PSNR(generated_np, target_np)
                psnr_total += psnr_val

                # Calculate SSIM
                ssim_val = ssim(
                    target_np, generated_np, multichannel=True, channel_axis=2
                )
                ssim_total += ssim_val

                # Calculate LPIPS
                lpips_val = lpips_loss(
                    generated.unsqueeze(0), target.unsqueeze(0)
                ).item()
                lpips_total += lpips_val

    avg_psnr = psnr_total / num_images
    avg_ssim = ssim_total / num_images
    avg_lpips = lpips_total / num_images

    return avg_psnr, avg_ssim, avg_lpips

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

    best_avg_metric = float("-inf")

    avg_psnr, avg_ssim, avg_lpips = evaluate(G, val_loader)
    wandb.log(
        {
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "avg_lpips": avg_lpips,
            "epoch": -1,
        }
    )

    for epoch_i in tqdm(range(n_epochs)):
        for batch_i, data in enumerate(train_loader):
            low_res_batch, high_res_batch = data

            low_res_batch = low_res_batch.to(DEVICE)
            high_res_batch = high_res_batch.to(DEVICE)

            d_loss = vanilla_discr_step(low_res_batch, high_res_batch, G, D, D_optim)

            if step_i % discriminator_steps == 0:
                g_loss = vanilla_gen_step(low_res_batch, G, D, G_optim)

            print(f"discr loss: {d_loss:.5f} | gen loss: {g_loss:.5f}")
            wandb.log(
                {
                    "discriminator_loss": d_loss,
                    "generator_loss": g_loss,
                    "epoch": epoch_i,
                    "step": step_i,
                }
            )

            step_i += 1

        if eval_steps and epoch_i % eval_steps == 0:
            avg_psnr, avg_ssim, avg_lpips = evaluate(G, val_loader)
            wandb.log(
                {
                    "avg_psnr": avg_psnr,
                    "avg_ssim": avg_ssim,
                    "avg_lpips": avg_lpips,
                    "epoch": epoch_i,
                }
            )

            # Calculate average metric
            avg_metric = (
                avg_psnr + avg_ssim - avg_lpips
            ) / 3  # Example: using -LPIPS since lower is better

            # Save model weights if average metric is improved
            if avg_metric > best_avg_metric:
                best_avg_metric = avg_metric
                torch.save(G.state_dict(), f"weights/G_best_avg_metric_{EXP_NAME}.pth")
                print(f"Best average metric updated: {best_avg_metric:.4f}")

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

    best_avg_metric = float("-inf")

    avg_psnr, avg_ssim, avg_lpips = evaluate(G, val_loader)
    wandb.log(
        {
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "avg_lpips": avg_lpips,
            "epoch": -1,
        }
    )

    for epoch_i in tqdm(range(n_epochs)):
        for batch_i, data in enumerate(train_loader):
            low_res_batch, high_res_batch = data

            low_res_batch = low_res_batch.to(DEVICE)
            high_res_batch = high_res_batch.to(DEVICE)

            d_loss = ns_discr_step(
                low_res_batch, high_res_batch, G, D, D_optim, r1_regularizer
            )

            if step_i % discriminator_steps == 0:
                g_loss = ns_mse_vgg_gen_step(
                    low_res_batch, high_res_batch, G, D, G_optim, LOSS_WEIGHTS, DEVICE
                )

            step_i += 1

            print(f"discr loss: {d_loss:.5f} | gen loss: {g_loss:.5f}")

        if eval_steps and epoch_i % eval_steps == 0:
            avg_psnr, avg_ssim, avg_lpips = evaluate(G, val_loader)
            wandb.log(
                {
                    "avg_psnr": avg_psnr,
                    "avg_ssim": avg_ssim,
                    "avg_lpips": avg_lpips,
                    "epoch": epoch_i,
                }
            )

            # Calculate average metric
            avg_metric = (
                avg_psnr + avg_ssim - avg_lpips
            ) / 3  # Example: using -LPIPS since lower is better

            # Save model weights if average metric is improved
            if avg_metric > best_avg_metric:
                best_avg_metric = avg_metric
                torch.save(G.state_dict(), f"weights/G_best_avg_metric_{EXP_NAME}.pth")
                print(f"Best average metric updated: {best_avg_metric:.4f}")

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

    wandb.init(project="superres-gans", name=EXP_NAME)  # Adjust project name as needed
    div2k_train = DIV2K("data/DIV2K_train_HR")
    div2k_val = DIV2K("data/DIV2K_valid_HR")

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
        n_epochs=100,
    )

    wandb.finish()

if __name__ == "__main__":
    main()
