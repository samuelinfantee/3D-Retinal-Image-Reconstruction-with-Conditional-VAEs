# pyright: reportMissingImports=false

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from OCTADataset import get_train_val_test, OCTA500SplitDataset
from CVAE import ConditionalVAE, cvae_loss   # wherever you saved the model code

import time
import os
import argparse
from typing import Any, Dict, Optional

from metrics import psnr_per_sample, ssim3d_per_sample


def minmax01(x):
    # unused now (kept only because you had it)
    x = x.astype("float32")
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def _run_id() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if job_id is None:
        return "local"
    return f"{job_id}" if task_id is None else f"{job_id}_{task_id}"


def save_checkpoint(
    save_path: str,
    model: torch.nn.Module,
    model_kwargs: Dict[str, Any],
    train_config: Dict[str, Any],
    epoch: int,
    val_loss: float,
    train_metrics: Optional[Dict[str, float]] = None,
    val_metrics: Optional[Dict[str, float]] = None,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "val_loss": float(val_loss),
        "model_kwargs": dict(model_kwargs),
        "train_config": dict(train_config),
        "train_metrics": dict(train_metrics or {}),
        "val_metrics": dict(val_metrics or {}),
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, save_path)


def main():
    parser = argparse.ArgumentParser(description="Train 3D Conditional VAE")
    parser.add_argument("--dataset-path", type=str, default="/general/sinfante/octa600/processed_small_filtered")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--sampling-rate", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--beta-warmup-epochs",
        type=int,
        default=5,
        help="Linearly ramp beta from 0 to --beta over this many epochs (0 disables).",
    )
    parser.add_argument("--recon", type=str, default="mse", choices=["mse", "l1"])
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    DATASET_PATH = args.dataset_path

    # speed-up for fixed input sizes (64^3)
    torch.backends.cudnn.benchmark = True

    # IMPORTANT: returns (train, val, test)
    train_files, val_files, test_files = get_train_val_test(
        DATASET_PATH, (0.9, 0.05, 0.05), seed=args.seed
    )

    # include_sampl_mat=True -> condition has 2 channels: [masked_image, mask]
    # transform=None because we will normalize x on GPU in the training loop
    train_ds = OCTA500SplitDataset(train_files, sampling_rate=args.sampling_rate, include_sampl_mat=True, transform=None)
    val_ds   = OCTA500SplitDataset(val_files,   sampling_rate=args.sampling_rate, include_sampl_mat=True, transform=None)

    TRAIN_BATCH_SIZE = args.train_batch_size   # (Conv3D is heavy)
    VAL_BATCH_SIZE = args.val_batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=8,              # match --cpus-per-task=8
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    c, x = next(iter(train_loader))
    print("c:", c.shape, "x:", x.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Using device:", device)

    # c_ch = 2 because include_sampl_mat=True
    model_kwargs = {
        "c_ch": 2,
        "x_ch": 1,
        "z_dim": 16,
        "enc_base_ch": 8,
        "dec_base_ch": 32,
        "out_activation": "sigmoid",
        "dropout_p": 0.1,
        "norm": "group", #use "group" or "batch"
    }

    model = ConditionalVAE(**model_kwargs).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    beta = float(args.beta)
    beta_warmup_epochs = max(0, int(args.beta_warmup_epochs))
    epochs = int(args.epochs)
    recon = str(args.recon)

    train_config = {
        "dataset_path": DATASET_PATH,
        "seed": int(args.seed),
        "sampling_rate": float(args.sampling_rate),
        "beta": float(beta),
        "beta_warmup_epochs": int(beta_warmup_epochs),
        "recon": str(recon),
        "split_ratios": (0.9, 0.05, 0.05),
    }

    run_id = _run_id()
    save_dir = os.path.join(args.save_dir, f"run_{run_id}")
    best_ckpt_path = os.path.join(save_dir, "best.pt")
    last_ckpt_path = os.path.join(save_dir, "last.pt")
    best_val = float("inf")

    best_train_metrics: Dict[str, float] = {}
    best_val_metrics: Dict[str, float] = {}

    for epoch in range(epochs):
        # Linear beta warmup (helps avoid posterior collapse early on)
        if beta_warmup_epochs <= 0:
            beta_eff = beta
        else:
            beta_eff = beta * min(1.0, float(epoch + 1) / float(beta_warmup_epochs))

        model.train()
        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "psnr": 0.0, "ssim": 0.0}
        n = 0

        t_epoch0 = time.time()

        for step, (c, x) in enumerate(train_loader, start=1):
            c = c.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)

            # --- GPU min-max normalization of x (THIS is where it belongs) ---
            mn = x.amin(dim=(2, 3, 4), keepdim=True)
            mx = x.amax(dim=(2, 3, 4), keepdim=True)
            x = (x - mn) / (mx - mn + 1e-8)
            # Normalize condition to the same scale as x.
            # c = [masked_image, mask]. Keep masked voxels at 0 by re-applying mask.
            c_img = c[:, 0:1]
            c_msk = c[:, 1:2]
            c_img = ((c_img - mn) / (mx - mn + 1e-8)) * c_msk
            c = torch.cat([c_img, c_msk], dim=1)
            # ---------------------------------------------------------------

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x=x, c=c)
                losses = cvae_loss(out, x, beta=beta_eff, recon=recon)

            # Metrics in fp32
            x_hat_f = out.x_hat.detach().float()
            x_f = x.detach().float()
            psnr_vals = psnr_per_sample(x_hat_f, x_f)
            ssim_vals = ssim3d_per_sample(x_hat_f, x_f)

            scaler.scale(losses["loss"]).backward()
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            n += bs
            for k in running:
                if k in losses:
                    running[k] += losses[k].item() * bs
            running["psnr"] += float(psnr_vals.sum().item())
            running["ssim"] += float(ssim_vals.sum().item())

        for k in running:
            running[k] /= max(n, 1)

        # quick validation
        model.eval()
        val_loss = 0.0
        vn = 0
        v_running = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "psnr": 0.0, "ssim": 0.0}
        with torch.no_grad():
            for c, x in val_loader:
                c = c.to(device, non_blocking=True)
                x = x.to(device, non_blocking=True)

                # --- GPU min-max normalization of x (validation too) ---
                mn = x.amin(dim=(2, 3, 4), keepdim=True)
                mx = x.amax(dim=(2, 3, 4), keepdim=True)
                x = (x - mn) / (mx - mn + 1e-8)
                # Normalize condition to match x (see training loop).
                c_img = c[:, 0:1]
                c_msk = c[:, 1:2]
                c_img = ((c_img - mn) / (mx - mn + 1e-8)) * c_msk
                c = torch.cat([c_img, c_msk], dim=1)
                # ------------------------------------------------------

                out = model(x=x, c=c)
                losses = cvae_loss(out, x, beta=beta_eff, recon=recon)

                x_hat_f = out.x_hat.detach().float()
                x_f = x.detach().float()
                v_psnr_vals = psnr_per_sample(x_hat_f, x_f)
                v_ssim_vals = ssim3d_per_sample(x_hat_f, x_f)
                bs = x.size(0)
                vn += bs
                val_loss += losses["loss"].item() * bs
                v_running["loss"] += float(losses["loss"].item()) * bs
                v_running["recon"] += float(losses["recon"].item()) * bs
                v_running["kl"] += float(losses["kl"].item()) * bs

                v_running["psnr"] += float(v_psnr_vals.sum().item())
                v_running["ssim"] += float(v_ssim_vals.sum().item())
        val_loss /= max(vn, 1)
        for k in v_running:
            v_running[k] /= max(vn, 1)

        epoch_time_s = time.time() - t_epoch0

        # checkpoints
        save_checkpoint(
            last_ckpt_path,
            model,
            model_kwargs=model_kwargs,
            train_config=train_config,
            epoch=epoch + 1,
            val_loss=val_loss,
            train_metrics={"psnr": float(running["psnr"]), "ssim": float(running["ssim"])},
            val_metrics={"psnr": float(v_running["psnr"]), "ssim": float(v_running["ssim"])},
        )
        if val_loss < best_val:
            best_val = val_loss
            best_train_metrics = {"psnr": float(running["psnr"]), "ssim": float(running["ssim"])}
            best_val_metrics = {"psnr": float(v_running["psnr"]), "ssim": float(v_running["ssim"])}
            save_checkpoint(
                best_ckpt_path,
                model,
                model_kwargs=model_kwargs,
                train_config=train_config,
                epoch=epoch + 1,
                val_loss=val_loss,
                train_metrics=best_train_metrics,
                val_metrics=best_val_metrics,
            )

        # Print on multiple lines so logs don't hide validation metrics due to wrapping.
        print(f"Epoch {epoch+1:03d} | time {epoch_time_s:.1f}s | beta {beta_eff:.3f}")
        print(
            f"TRAIN | loss {running['loss']:.8f} | recon {running['recon']:.8f} | kl {running['kl']:.8f} | "
            f"psnr {running['psnr']:.3f} | ssim {running['ssim']:.4f}"
        )
        print(
            f"VAL   | loss {v_running['loss']:.8f} | recon {v_running['recon']:.8f} | kl {v_running['kl']:.8f} | "
            f"psnr {v_running['psnr']:.3f} | ssim {v_running['ssim']:.4f}"
        )
        print(
            f"BEST  | val psnr {best_val_metrics.get('psnr', float('nan')):.3f} | "
            f"val ssim {best_val_metrics.get('ssim', float('nan')):.4f} | "
            f"train psnr {best_train_metrics.get('psnr', float('nan')):.3f} | "
            f"train ssim {best_train_metrics.get('ssim', float('nan')):.4f}"
        )

    # Inference example: reconstruct from condition only
    model.eval()
    c, x = next(iter(val_loader))
    c = c.to(device)
    x = x.to(device)

    # Normalize c using x's min/max (same convention as training/eval).
    mn = x.amin(dim=(2, 3, 4), keepdim=True)
    mx = x.amax(dim=(2, 3, 4), keepdim=True)
    c_img = c[:, 0:1]
    c_msk = c[:, 1:2]
    c_img = ((c_img - mn) / (mx - mn + 1e-8)) * c_msk
    c = torch.cat([c_img, c_msk], dim=1)

    # NOTE: sample() uses c only and returns sigmoid output; no x normalization needed here
    with torch.no_grad():
        x_hat = model.sample(c, deterministic=True)

    print("Done. x_hat shape:", x_hat.shape)


if __name__ == "__main__":
    main()