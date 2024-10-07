import torch
import torch.nn as nn
import torch.nn.functional as F

def ns_gen_step(
    X: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    G_optim: torch.optim.Optimizer,
) -> torch.Tensor:
    G.train()
    D.eval()

    X_gen = G(X)
    
    scores_gen = D(X_gen)
    loss = F.softplus(-scores_gen).mean()
    G_optim.zero_grad()
    loss.backward()
    G_optim.step()

    return loss.item()


def ns_discr_step(
    X: torch.Tensor,
    Y: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    D_optim: torch.optim.Optimizer,
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