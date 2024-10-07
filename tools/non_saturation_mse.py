import torch
import torch.nn as nn
import torch.nn.functional as F


def ns_mse_gen_step(
    X: torch.Tensor,
    Y: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    G_optim: torch.optim.Optimizer,
    gamma: float,
) -> torch.Tensor:
    G.train()
    D.eval()

    X_gen = G(X)

    scores_gen = D(X_gen)
    mse = nn.MSELoss()
    loss = gamma * F.softplus(-scores_gen).mean() + (1 - gamma) * mse(X_gen, Y)
    G_optim.zero_grad()
    loss.backward()
    G_optim.step()

    return loss.item()
