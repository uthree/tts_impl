import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPM(nn.Module):
    """
    Implementation of Denoising Diffusion Probabilistic Models (WIP)
    https://arxiv.org/abs/2006.11239
    """

    def __init__(
        self,
        model: nn.Module,
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
        num_timesteps: int = 1000,
        loss_function=nn.HuberLoss(),
    ):
        super().__init__()
        self.model = model
        self.beta = torch.linspace(beta_min, beta_max, num_timesteps)
        self.alpha = 1 - self.beta
        self.num_timesteps = num_timesteps
        self.loss_function = loss_function

        # calculate alpha_bar
        self.alpha_bar = []
        for t in range(1, num_timesteps + 1):
            self.alpha_bar.append(torch.prod(self.alpha[:t]))
        self.alpha_bar = torch.Tensor(self.alpha_bar)

        # calculate beta_tilde ( for calculate sigma_t )
        self.beta_tilde = [1]
        for t in range(1, num_timesteps):
            self.beta_tilde.append(
                (1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t]) * self.beta[t]
            )
        self.beta_tilde = torch.Tensor(self.beta_tilde)

    def forward(self, x, condition=None):
        t = torch.randint(low=1, high=self.num_timesteps, size=(x.shape[0],))
        alpha_bar_t = torch.index_select(self.alpha_bar, 0, t).to(x.device)
        while alpha_bar_t.ndim < x.ndim:
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        e = torch.randn(*x.shape, device=x.device)
        t = t.to(x.device)
        e_theta = self.model(
            x=torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * e,
            t=t,
            condition=condition,
        )
        loss = self.loss_function(e_theta, e)
        return loss

    # sample as DDIM (http://arxiv.org/abs/2010.02502)
    @torch.no_grad()
    def sample(self, x_shape=(1,), condition=None, num_steps=20, eta=0):
        # device
        device = self.model.parameters().__next__().device

        # Initialize
        x = torch.randn(*x_shape, device=device)
        steps = list(torch.linspace(0, self.num_timesteps - 1, num_steps).int().numpy())
        steps_next = [0] + steps[:-1]
        alpha = torch.cumprod((1 - self.beta), dim=0)
        for t, t_next in zip(reversed(steps), reversed(steps_next)):
            t_tensor = torch.full((x_shape[0],), t, device=device)
            e_theta = self.model(x=x, time=t_tensor, condition=condition)

            e = torch.randn(*x_shape, device=device)
            sigma = (
                eta
                * torch.sqrt((1 - alpha[t_next]) / (1 - alpha[t]))
                * torch.sqrt(1 - alpha[t] / alpha[t_next])
            )
            x_t0 = (x - torch.sqrt(1 - alpha[t]) * e_theta) / torch.sqrt(alpha[t])
            term_1 = torch.sqrt(alpha[t_next]) * x_t0
            term_2 = torch.sqrt(1 - alpha[t_next] - sigma**2) * e_theta
            term_3 = sigma * e

            if t == 0:
                x = x_t0
            else:
                x = term_1 + term_2 + term_3
        return x
