# pyright: reportMissingImports=false

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple


# -----------------------------
# 3D conv helpers
# -----------------------------
class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dropout_p: float = 0.1,
        norm: str = "group",
    ):
        super().__init__()
        self.dropout_p = float(dropout_p)
        self.norm = str(norm).lower()

        if self.norm == "batch":
            norm1 = nn.BatchNorm3d(out_ch)
            norm2 = nn.BatchNorm3d(out_ch)
        elif self.norm == "group":
            norm1 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
            norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        else:
            raise ValueError(f"Unsupported norm='{norm}'. Use 'group' or 'batch'.")

        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            norm1,
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            norm2,
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        # Use funcional dropout to avoid changing module indexing in nn.Sequential,
        if self.dropout_p > 0:
            y = F.dropout3d(y, p=self.dropout_p, training=self.training)
        return y


class EncoderBackbone3D(nn.Module):
    """
    Fully convolutional 3D encoder backbone -> fixed-size vector via AdaptiveAvgPool3d(1).
    Works for arbitrary D, H, W.
    """
    def __init__(
        self,
        in_ch: int,
        base_ch: int = 16,
        levels: int = 4,
        dropout_p: float = 0.1,
        norm: str = "group",
    ):
        super().__init__()
        chs = [base_ch * (2 ** i) for i in range(levels)]
        blocks = []
        prev = in_ch
        for ch in chs:
            blocks.append(ConvBlock3D(prev, ch, stride=2, dropout_p=dropout_p, norm=norm))  # downsample
            prev = ch

        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.out_dim = chs[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.pool(h).flatten(1)  # [B, out_dim]
        return h


class GaussianHead(nn.Module):
    def __init__(self, in_dim: int, z_dim: int):
        super().__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.logvar = nn.Linear(in_dim, z_dim)

    def forward(self, h: torch.Tensor):
        return self.mu(h), self.logvar(h)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
    """
    KL(q || p) for diagonal Gaussians. Returns scalar mean over batch.
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        (logvar_p - logvar_q) +
        (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8) -
        1.0
    )
    return kl.sum(dim=1).mean()


class ConditionalDecoder3D(nn.Module):
    """
    3D decoder conditioned on c via a learned conditioning vector.
    Upsamples with trilinear interpolation to match target D,H,W.
    """
    def __init__(
        self,
        c_ch: int,
        z_dim: int,
        base_ch: int = 128,
        out_ch: int = 1,
        dropout_p: float = 0.1,
        norm: str = "group",
    ):
        super().__init__()
        self.c_ch = int(c_ch)
        self.dropout_p = float(dropout_p)
        # Keep a global conditioning path (helps), but also inject c spatially during upsampling.
        self.c_encoder = EncoderBackbone3D(in_ch=c_ch, base_ch=8, levels=3, dropout_p=dropout_p, norm=norm)
        cond_dim = self.c_encoder.out_dim

        self.fc = nn.Sequential(
            nn.Linear(z_dim + cond_dim, base_ch),
            nn.SiLU(inplace=True),
            nn.Linear(base_ch, base_ch * 2 * 2 * 2),  # start from 4x4x4
            nn.SiLU(inplace=True),
        )

        # keep channels constant during upsampling; inject spatial condition by concatenation
        self.refine = ConvBlock3D(base_ch + self.c_ch, base_ch, stride=1, dropout_p=dropout_p, norm=norm)

        # reduce channels near the end
        self.tail = nn.Sequential(
            ConvBlock3D(base_ch, base_ch // 2, stride=1, dropout_p=dropout_p, norm=norm),
            ConvBlock3D(base_ch // 2, base_ch // 4, stride=1, dropout_p=dropout_p, norm=norm),
            nn.Conv3d(base_ch // 4, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor, target_dhw: Tuple[int, int, int]) -> torch.Tensor:
        B = z.shape[0]
        cond_vec = self.c_encoder(c)
        h = torch.cat([z, cond_vec], dim=1)

        h = self.fc(h).view(B, -1, 2, 2, 2)  # [B, base_ch, 4, 4, 4]
        if self.dropout_p > 0:
            h = F.dropout(h, p=self.dropout_p, training=self.training)

        Dt, Ht, Wt = target_dhw
        while h.shape[-3] < Dt or h.shape[-2] < Ht or h.shape[-1] < Wt:
            new_d = min(h.shape[-3] * 2, Dt)
            new_h = min(h.shape[-2] * 2, Ht)
            new_w = min(h.shape[-1] * 2, Wt)
            h = F.interpolate(h, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)

            # Spatial conditioning: resize c to current resolution and concat.
            # Use trilinear for intensity channel(s) and nearest for mask-like channels.
            if self.c_ch >= 2:
                c_img = F.interpolate(c[:, 0:1], size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)
                c_msk = F.interpolate(c[:, 1:2], size=(new_d, new_h, new_w), mode="nearest")
                if self.c_ch > 2:
                    c_rest = F.interpolate(c[:, 2:self.c_ch], size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)
                    c_resized = torch.cat([c_img, c_msk, c_rest], dim=1)
                else:
                    c_resized = torch.cat([c_img, c_msk], dim=1)
            else:
                c_resized = F.interpolate(c[:, 0:1], size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)

            h = torch.cat([h, c_resized], dim=1)
            h = self.refine(h)

        out = self.tail(h)

        if out.shape[-3:] != (Dt, Ht, Wt):
            out = F.interpolate(out, size=(Dt, Ht, Wt), mode="trilinear", align_corners=False)

        return out


@dataclass
class CVAEOutput:
    x_hat: torch.Tensor
    mu_q: torch.Tensor
    logvar_q: torch.Tensor
    mu_p: torch.Tensor
    logvar_p: torch.Tensor


class ConditionalVAE(nn.Module):
    """
    3D CVAE:
      q(z|x,c) posterior
      p(z|c)   conditional prior
      p(x|z,c) decoder
    """
    def __init__(
        self,
        c_ch: int,
        x_ch: int = 1,
        z_dim: int = 64,
        enc_base_ch: int = 16,
        dec_base_ch: int = 128,
        out_activation: str = "sigmoid",
        dropout_p: float = 0.1,
        norm: str = "group",
    ):
        super().__init__()
        self.out_activation = out_activation

        dropout_p = float(dropout_p)
        norm = str(norm).lower()

        self.q_backbone = EncoderBackbone3D(
            in_ch=x_ch + c_ch,
            base_ch=enc_base_ch,
            levels=3,
            dropout_p=dropout_p,
            norm=norm,
        )
        self.q_head = GaussianHead(self.q_backbone.out_dim, z_dim)

        self.p_backbone = EncoderBackbone3D(
            in_ch=c_ch,
            base_ch=enc_base_ch,
            levels=3,
            dropout_p=dropout_p,
            norm=norm,
        )
        self.p_head = GaussianHead(self.p_backbone.out_dim, z_dim)

        self.decoder = ConditionalDecoder3D(
            c_ch=c_ch,
            z_dim=z_dim,
            base_ch=dec_base_ch,
            out_ch=x_ch,
            dropout_p=dropout_p,
            norm=norm,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> CVAEOutput:
        qc_in = torch.cat([x, c], dim=1)  # [B, x_ch+c_ch, D, H, W]
        h_q = self.q_backbone(qc_in)
        mu_q, logvar_q = self.q_head(h_q)

        h_p = self.p_backbone(c)
        mu_p, logvar_p = self.p_head(h_p)

        z = reparameterize(mu_q, logvar_q)

        x_hat = self.decoder(z, c, target_dhw=(x.shape[-3], x.shape[-2], x.shape[-1]))
        if self.out_activation == "sigmoid":
            x_hat = torch.sigmoid(x_hat)

        # If c includes a sampling mask channel, enforce known voxels exactly.
        # Expected convention: c[:,0]=masked_image, c[:,1]=mask in {0,1}.
        if c.dim() == 5 and c.size(1) >= 2:
            c_img = c[:, 0:1]
            c_msk = c[:, 1:2]
            x_hat = x_hat * (1.0 - c_msk) + c_img

        return CVAEOutput(x_hat=x_hat, mu_q=mu_q, logvar_q=logvar_q, mu_p=mu_p, logvar_p=logvar_p)

    @torch.no_grad()
    def sample(self, c: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        h_p = self.p_backbone(c)
        mu_p, logvar_p = self.p_head(h_p)
        z = mu_p if deterministic else reparameterize(mu_p, logvar_p)

        x_hat = self.decoder(z, c, target_dhw=(c.shape[-3], c.shape[-2], c.shape[-1]))
        if self.out_activation == "sigmoid":
            x_hat = torch.sigmoid(x_hat)

        if c.dim() == 5 and c.size(1) >= 2:
            c_img = c[:, 0:1]
            c_msk = c[:, 1:2]
            x_hat = x_hat * (1.0 - c_msk) + c_img
        return x_hat


def cvae_loss(out: CVAEOutput, x: torch.Tensor, beta: float = 1.0, recon: str = "mse") -> Dict[str, torch.Tensor]:
    if recon == "l1":
        recon_loss = F.l1_loss(out.x_hat, x, reduction="mean")
    else:
        recon_loss = F.mse_loss(out.x_hat, x, reduction="mean")

    kl = kl_divergence_gaussians(out.mu_q, out.logvar_q, out.mu_p, out.logvar_p)
    total = recon_loss + beta * kl
    return {"loss": total, "recon": recon_loss, "kl": kl}
