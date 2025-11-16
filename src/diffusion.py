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
        
        # For training process (forward)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat)
        
        # For sampling process
        self.alpha_hat_prev = F.pad(self.alpha_hat[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)
        self.beta_over_sqrt_one_minus_alpha_hat = (1.0 - self.alpha) / self.sqrt_one_minus_alpha_hat
        self.posterior_variance = self.beta * (1.0 - self.alpha_hat_prev) / (1.0 - self.alpha_hat)
        self.posterior_log_variance = torch.log(self.posterior_variance.clamp(min=1e-20))

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

    # Reversing the noise p(x_{t-1} | x_t)
    def sample_step(self, model, x_t, t, condition):
        with torch.no_grad():
            t_batch = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise using the U-Net
            predicted_noise = model(x_t, t_batch, condition)
            #predicted_noise = torch.clamp(predicted_noise, -1.0, 1.0)
            
            # Get pre-calculated coefficients
            beta_over_sqrt_t = self._get_tensor_values(self.beta_over_sqrt_one_minus_alpha_hat, t_batch)
            sqrt_recip_alpha_t = self._get_tensor_values(self.sqrt_recip_alpha, t_batch)
            
            # Calculate mean
            mean = sqrt_recip_alpha_t * (x_t - beta_over_sqrt_t * predicted_noise)
            
            if t == 0:
                return mean
            else:
                # Get variance
                log_variance = self._get_tensor_values(self.posterior_log_variance, t_batch)
                variance = torch.exp(0.5 * log_variance)
                
                # Sample
                noise = torch.randn_like(x_t)
                return mean + (variance * noise)
