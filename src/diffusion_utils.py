import torch

def get_noise_schedule(T_max, beta_start=0.0001, beta_end=0.02, device='cpu'):

    betas = torch.linspace(beta_start, beta_end, T_max, device=device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    return alphas_cumprod

def forward_diffusion(E_0, t, alphas_cumprod):
    
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t]).view(-1, 1, 1)
    noise = torch.randn_like(E_0)
    E_t = sqrt_alphas_cumprod_t * E_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return E_t