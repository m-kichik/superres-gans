import torch
import torch.nn as nn
import torch.nn.functional as F


def vanilla_gen_step(
    X: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    G_optim: torch.optim.Optimizer,
) -> torch.Tensor:

    G.train()
    D.eval()

    X_gen = G(X)

    scores_gen = D(X_gen)
    loss = -F.binary_cross_entropy(scores_gen, torch.zeros_like(scores_gen))
    G_optim.zero_grad()
    loss.backward()
    G_optim.step()

    return loss.item()


def vanilla_discr_step(
    X: torch.Tensor,
    Y: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    D_optim: torch.optim.Optimizer,
) -> torch.Tensor:

    G.eval()
    D.train()
    with torch.no_grad():
        X_gen = G(X)
    scores_gen = D(X_gen)
    scores_real = D(Y)
    loss_gen = F.binary_cross_entropy(scores_gen, torch.zeros_like(scores_gen))
    loss_real = F.binary_cross_entropy(scores_real, torch.ones_like(scores_real))
    loss = loss_gen + loss_real

    D_optim.zero_grad()
    loss.backward()
    D_optim.step()

    return loss.item()
