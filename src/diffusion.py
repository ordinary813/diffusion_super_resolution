import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.timesteps = timesteps
        self.device = device
        
        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_prev = F.pad(self.alpha_hat[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat)
        self.sqrt_recip_alpha_hat = torch.sqrt(1.0 / self.alpha_hat)
        self.sqrt_recipm1_alpha_hat = torch.sqrt(1.0 / self.alpha_hat - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.beta * (1.0 - self.alpha_hat_prev) / (1.0 - self.alpha_hat)
        # Log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance = torch.log(self.posterior_variance.clamp(min=1e-20))
        
        self.posterior_mean_coef1 = (
            self.beta * torch.sqrt(self.alpha_hat_prev) / (1.0 - self.alpha_hat)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_hat_prev) * torch.sqrt(self.alpha) / (1.0 - self.alpha_hat)
        )

    # Retrievve values at timestep t
    def _get_tensor_values(self, tensor, t):
        values = tensor[t]
        return values[:, None, None, None]

    # q(x_t | x_0)
    def add_noise(self, x_0, t):
        epsilon = torch.randn_like(x_0)
        x_t = (
            self._get_tensor_values(self.sqrt_alpha_hat, t) * x_0 +
            self._get_tensor_values(self.sqrt_one_minus_alpha_hat, t) * epsilon
        )
        return x_t, epsilon

    # Predict x_0 from x_t
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._get_tensor_values(self.sqrt_recip_alpha_hat, t) * x_t -
            self._get_tensor_values(self.sqrt_recipm1_alpha_hat, t) * noise
        )
    
    # Compute q(x_t-1 | x_t, x_0)
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self._get_tensor_values(self.posterior_mean_coef1, t) * x_start +
            self._get_tensor_values(self.posterior_mean_coef2, t) * x_t
        )
        posterior_log_variance = self._get_tensor_values(self.posterior_log_variance, t)
        return posterior_mean, posterior_log_variance

    # Reversing the noise p(x_{t-1} | x_t)
    def sample_step(self, model, x_t, t, condition, cfg_scale=3.0):
        with torch.no_grad():
            t_batch = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
            
            if cfg_scale > 1.0:
                # Conditional prediction (using the LR image)
                noise_cond = model(x_t, t_batch, condition)
                
                # Unconditional prediction (using a black image)
                noise_uncond = model(x_t, t_batch, torch.zeros_like(condition))
                
                # Combine them: push away from unconditional, towards conditional
                predicted_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                # Standard sampling if scale is 1.0
                predicted_noise = model(x_t, t_batch, condition)

            # Predict x_0 and clamp it to [-1, 1] to stop color explosion
            x_start = self.predict_start_from_noise(x_t, t_batch, predicted_noise)
            x_start = torch.clamp(x_start, -1.0, 1.0)
            
            # Posterior Mean
            mean, log_variance = self.q_posterior(x_start, x_t, t_batch)
            
            if t == 0:
                return mean
            else:
                noise = torch.randn_like(x_t)
                variance = torch.exp(0.5 * log_variance)
                return mean + (variance * noise)
