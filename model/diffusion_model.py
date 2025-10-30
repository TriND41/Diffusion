import torch
import torch.nn as nn
from model.modules.u_net import UNet
from model.utils.scheduling import linear_beta_schedule

class DiffusionModel(nn.Module):
    def __init__(
        self,
        channels: int,
        n_conditions: int,
        bilinear: bool = False,
        timesteps: int = 1000
    ) -> None:
        super().__init__()
        betas = linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.timesteps = timesteps

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)

        self.model = UNet(channels, channels, n_conditions, bilinear)
    
    def sample_noise(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_minus_one_alpha_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha_t * x_0 + sqrt_minus_one_alpha_t * noise

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        noise = torch.randn(x.size(), dtype=x.dtype, device=x.device)
        t = torch.randint(0, self.timesteps, [batch_size], device=x.device)
        
        x = self.sample_noise(x, t, noise)
        x = self.model(x, cond)

        return x