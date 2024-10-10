from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .d_vgg_mse import DiscriminatorVggMse


def ns_mse_vgg_gen_step(
    X: torch.Tensor,
    Y: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    G_optim: torch.optim.Optimizer,
    weights: List[float],
    device: str,
) -> torch.Tensor:
    G.train()
    D.eval()

    X_gen = G(X)

    scores_gen = D(X_gen)
    mse = nn.MSELoss()
    d_vgg_mse = DiscriminatorVggMse(weight=0.0001, inpad_size=16).to(device)
    loss = [
        w * l
        for w, l in zip(
            weights,
            [F.softplus(-scores_gen).mean(), mse(X_gen, Y), d_vgg_mse.loss(X_gen, Y)],
        )
    ]
    loss = sum(loss)
    G_optim.zero_grad()
    loss.backward()
    G_optim.step()

    return loss.item()


def ns_mse_vgg_discr_step(
    X: torch.Tensor,
    Y: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    D_optim: torch.optim.Optimizer,
    weights: List[float],
    device: str,
    r1_regularizer: float = 1.0,
) -> torch.Tensor:
    G.eval()
    D.train()
    D_optim.zero_grad()

    with torch.no_grad():
        X_gen = G(X)
    Y.requires_grad_()

    scores_gen = D(X_gen)
    scores_real = D(Y)

    loss_gen = F.softplus(scores_gen).mean()

    mse = nn.MSELoss()
    d_vgg_mse = DiscriminatorVggMse(weight=0.0001, inpad_size=16).to(device)
    loss = [
        w * l
        for w, l in zip(
            weights,
            [F.softplus(scores_gen).mean(), mse(X_gen, Y), d_vgg_mse.loss(X_gen, Y)],
        )
    ]
    loss_gen = sum(loss)

    loss_real = F.softplus(-scores_real).mean()
    scores_real.sum().backward(retain_graph=True, create_graph=True)

    gradients = Y.grad
    grad_penalty = (gradients.view(gradients.size(0), -1).norm(2, dim=1) ** 2).mean()

    D_optim.zero_grad()
    loss = loss_gen + loss_real + r1_regularizer * grad_penalty
    loss.backward()
    D_optim.step()
    gradients.detach_()  # to avoid memory leak!

    return loss.item()
