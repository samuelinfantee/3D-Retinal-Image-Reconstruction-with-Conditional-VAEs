# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


_WINDOW_CACHE: Dict[Tuple[int, float, str, torch.dtype], torch.Tensor] = {}


def _gaussian_1d(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
	g = torch.exp(-(coords**2) / (2 * (sigma**2)))
	g = g / g.sum()
	return g


def _gaussian_window_3d(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	key = (int(window_size), float(sigma), str(device), dtype)
	w = _WINDOW_CACHE.get(key)
	if w is not None:
		return w

	g1 = _gaussian_1d(window_size, sigma, device=device, dtype=dtype)
	# separable outer products -> [W,W,W]
	g3 = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
	w = g3.unsqueeze(0).unsqueeze(0)  # [1,1,W,W,W]
	_WINDOW_CACHE[key] = w
	return w


@torch.no_grad()
def psnr_per_sample(x_hat: torch.Tensor, x: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
	"""Compute PSNR per sample for tensors shaped [B,1,D,H,W] (or [B,C,...])."""
	x_hat = x_hat.float()
	x = x.float()
	dims = tuple(range(1, x.dim()))
	mse = (x_hat - x).pow(2).mean(dim=dims).clamp_min(eps)
	psnr = 10.0 * torch.log10((float(data_range) ** 2) / mse)
	return psnr


@torch.no_grad()
def ssim3d_per_sample(
	x_hat: torch.Tensor,
	x: torch.Tensor,
	window_size: int = 7,
	sigma: float = 1.5,
	data_range: float = 1.0,
	K1: float = 0.01,
	K2: float = 0.03,
	eps: float = 1e-8,
) -> torch.Tensor:
	"""Compute 3D SSIM per sample for tensors shaped [B,1,D,H,W].

	This is the standard local SSIM using a Gaussian window implemented via conv3d.
	Assumes values are scaled to [0, data_range].
	"""
	if x_hat.dim() != 5 or x.dim() != 5:
		raise ValueError("ssim3d_per_sample expects 5D tensors [B,C,D,H,W]")
	if x_hat.size(0) != x.size(0) or x_hat.shape[2:] != x.shape[2:]:
		raise ValueError("x_hat and x must have same batch and spatial shape")
	if x_hat.size(1) != x.size(1):
		raise ValueError("x_hat and x must have same number of channels")

	# SSIM constants
	L = float(data_range)
	C1 = (K1 * L) ** 2
	C2 = (K2 * L) ** 2

	x_hat = x_hat.float()
	x = x.float()
	B, C, _, _, _ = x.shape
	window = _gaussian_window_3d(window_size, sigma, device=x.device, dtype=x.dtype)
	pad = window_size // 2

	# Expand window for grouped conv if C>1
	window = window.expand(C, 1, window_size, window_size, window_size)

	mu_x = F.conv3d(x, window, padding=pad, groups=C)
	mu_y = F.conv3d(x_hat, window, padding=pad, groups=C)

	sigma_x = F.conv3d(x * x, window, padding=pad, groups=C) - mu_x * mu_x
	sigma_y = F.conv3d(x_hat * x_hat, window, padding=pad, groups=C) - mu_y * mu_y
	sigma_xy = F.conv3d(x * x_hat, window, padding=pad, groups=C) - mu_x * mu_y

	# Numerical stability
	sigma_x = sigma_x.clamp_min(0.0)
	sigma_y = sigma_y.clamp_min(0.0)

	num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
	den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
	ssim_map = num / (den + eps)

	# Average over channels and spatial dims -> [B]
	ssim = ssim_map.mean(dim=(1, 2, 3, 4))
	return ssim
