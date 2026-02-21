import torch
import torch.nn.functional as F
import math

class DiffusionModel:
    def __init__(self, timesteps=1000, device="cuda", schedule_type = "cosine"):
        self.timesteps = timesteps
        self.device = device

        if schedule_type == "cosine":
            self.beta = self.cosine_beta_schedule(timesteps).to(device)
        else:
            self.beta = torch.linspace(1e-4, 0.02, timesteps).to(device)

        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_prev = F.pad(self.alpha_hat[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat)
        self.sqrt_recip_alpha_hat = torch.sqrt(1.0 / self.alpha_hat)
        self.sqrt_recipm1_alpha_hat = torch.sqrt(1.0 / self.alpha_hat - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_mean_coef1 = self.beta * torch.sqrt(self.alpha_hat_prev) / (1.0 - self.alpha_hat)
        self.posterior_mean_coef2 = (1.0 - self.alpha_hat_prev) * torch.sqrt(self.alpha) / (1.0 - self.alpha_hat)
        self.posterior_variance = self.beta * (1.0 - self.alpha_hat_prev) / (1.0 - self.alpha_hat)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    # Calculate q(x_t | x_0)
    def add_noise(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha_hat = self.sqrt_alpha_hat[t][:, None, None, None]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t][:, None, None, None]
        return sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise, noise

    # Estimate p(x{t-1} | x_t) (Reverse process)
    def sample_step(self, model, x_t, t, condition, cfg_scale=3.0):
        t_batch = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
        noise_cond = model(x_t, t_batch, condition)
        noise_uncond = model(x_t, t_batch, torch.zeros_like(condition))
        eps = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        
        x_0_pred = (self.sqrt_recip_alpha_hat[t] * x_t - self.sqrt_recipm1_alpha_hat[t] * eps).clamp(-1, 1)
        mean = self.posterior_mean_coef1[t] * x_0_pred + self.posterior_mean_coef2[t] * x_t
        if t == 0: return mean
        return mean + torch.sqrt(self.posterior_variance[t]) * torch.randn_like(x_t)