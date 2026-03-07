#!/usr/bin/env python
# pyright: reportMissingImports=false

import argparse
import os
import random
import sys

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from metrics import psnr_per_sample, ssim3d_per_sample


def _add_cvae_to_path() -> str:
	repo_root = os.path.dirname(os.path.abspath(__file__))
	# Add the directory containing this script (where CVAE.py, OCTADataset.py, metrics.py live)
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)
	return repo_root


def _load_checkpoint(path: str) -> dict:
	ckpt = torch.load(path, map_location="cpu")
	if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
		raise ValueError(f"Checkpoint at {path} does not look like a CVAE checkpoint")
	return ckpt


def _try_import_matplotlib():
	try:
		import matplotlib  # type: ignore
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt  # type: ignore
		return plt
	except Exception as e:
		print(f"WARNING: matplotlib not available; skipping image comparisons ({e})")
		return None


def _save_side_by_side_png(plt, save_path: str, gen_2d, gt_2d) -> None:
	fig = plt.figure(figsize=(6, 3), dpi=150)
	ax1 = fig.add_subplot(1, 2, 1)
	ax1.imshow(gen_2d, cmap="gray", vmin=0.0, vmax=1.0)
	ax1.set_title("Reconstruction")
	ax1.axis("off")

	ax2 = fig.add_subplot(1, 2, 2)
	ax2.imshow(gt_2d, cmap="gray", vmin=0.0, vmax=1.0)
	ax2.set_title("Ground truth")
	ax2.axis("off")

	fig.tight_layout(pad=0.2)
	fig.savefig(save_path)
	plt.close(fig)


def main() -> None:
	repo_root = _add_cvae_to_path()
	from OCTADataset import get_train_val_test, OCTA500SplitDataset  # noqa: E402
	from CVAE import ConditionalVAE, cvae_loss  # noqa: E402

	parser = argparse.ArgumentParser(description="Evaluate a trained ConditionalVAE on the test split")
	default_ckpt = os.path.join(repo_root, "models", "best_model.pt")
	parser.add_argument(
		"--checkpoint",
		type=str,
		default=default_ckpt,
		required=False,
		help=f"Path to .pt checkpoint (default: {default_ckpt})",
	)
	parser.add_argument(
		"--dataset-path",
		type=str,
		required=True,
		help="Root dataset directory (this script will NOT read dataset_path from the checkpoint)",
	)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--seed", type=int, default=None, help="Overrides seed stored in checkpoint")
	parser.add_argument("--sampling-rate", type=float, default=None, help="Overrides sampling_rate stored in checkpoint")
	parser.add_argument("--beta", type=float, default=None, help="Overrides beta stored in checkpoint")
	parser.add_argument("--recon", type=str, default=None, choices=["mse", "l1"], help="Overrides recon loss")
	parser.add_argument(
		"--save-comparisons-dir",
		type=str,
		default=None,
		help="If set, saves side-by-side PNGs (generated vs GT) for random test samples",
	)
	parser.add_argument("--num-comparisons", type=int, default=20)
	parser.add_argument(
		"--comparison-seed",
		type=int,
		default=None,
		help="Seed for choosing random test samples (defaults to the split seed)",
	)
	args = parser.parse_args()

	if not os.path.isfile(args.checkpoint):
		raise FileNotFoundError(
			"Checkpoint not found: "
			+ str(args.checkpoint)
			+ "\nTip: pass --checkpoint models\\best_model.pt (or an absolute path)."
		)

	ckpt = _load_checkpoint(args.checkpoint)
	model_kwargs = ckpt.get("model_kwargs", {})
	train_config = ckpt.get("train_config", {})

	dataset_path = args.dataset_path

	seed = int(args.seed if args.seed is not None else train_config.get("seed", 42))
	sampling_rate = float(
		args.sampling_rate if args.sampling_rate is not None else train_config.get("sampling_rate", 0.6)
	)
	beta = float(args.beta if args.beta is not None else train_config.get("beta", 1.0))
	recon = str(args.recon if args.recon is not None else train_config.get("recon", "mse"))
	split_ratios = tuple(train_config.get("split_ratios", (0.9, 0.05, 0.05)))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Using device:", device)
	if device.type == "cuda":
		print("GPU:", torch.cuda.get_device_name(0))

	model = ConditionalVAE(**model_kwargs).to(device)
	model.load_state_dict(ckpt["model_state_dict"], strict=True)
	model.eval()

	_, _, test_files = get_train_val_test(dataset_path, split_ratios, seed=seed)
	test_ds = OCTA500SplitDataset(
		test_files,
		sampling_rate=sampling_rate,
		include_sampl_mat=True,
		transform=None,
	)
	loader_kwargs = {
		"batch_size": args.batch_size,
		"shuffle": False,
		"num_workers": args.num_workers,
		"pin_memory": (device.type == "cuda"),
		"persistent_workers": (args.num_workers > 0),
	}
	if args.num_workers > 0:
		loader_kwargs["prefetch_factor"] = 4

	test_loader = DataLoader(test_ds, **loader_kwargs)

	epoch = ckpt.get("epoch")
	val_loss = ckpt.get("val_loss")
	print(f"Loaded checkpoint: {args.checkpoint}")
	print(f"Checkpoint epoch: {epoch} | val_loss: {val_loss}")
	print(f"Test config: seed={seed} sampling_rate={sampling_rate} beta={beta} recon={recon}")

	totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
	metrics = {"psnr": 0.0, "ssim": 0.0}
	n = 0
	with torch.no_grad():
		for c, x in test_loader:
			c = c.to(device, non_blocking=True)
			x = x.to(device, non_blocking=True)

			# GPU min-max normalization (match training)
			mn = x.amin(dim=(2, 3, 4), keepdim=True)
			mx = x.amax(dim=(2, 3, 4), keepdim=True)
			x = (x - mn) / (mx - mn + 1e-8)
			# Normalize condition (c=[masked_image, mask]) to match x scale.
			c_img = c[:, 0:1]
			c_msk = c[:, 1:2]
			c_img = ((c_img - mn) / (mx - mn + 1e-8)) * c_msk
			c = torch.cat([c_img, c_msk], dim=1)

			with autocast("cuda", enabled=(device.type == "cuda")):
				out = model(x=x, c=c)
				losses = cvae_loss(out, x, beta=beta, recon=recon)

			x_hat_f = out.x_hat.detach().float()
			x_f = x.detach().float()
			psnr_vals = psnr_per_sample(x_hat_f, x_f)
			ssim_vals = ssim3d_per_sample(x_hat_f, x_f)

			bs = x.size(0)
			n += bs
			for k in totals:
				totals[k] += float(losses[k].item()) * bs
			metrics["psnr"] += float(psnr_vals.sum().item())
			metrics["ssim"] += float(ssim_vals.sum().item())

	for k in totals:
		totals[k] /= max(n, 1)
	for k in metrics:
		metrics[k] /= max(n, 1)

	print(
		"TEST | "
		f"loss {totals['loss']:.6f} | recon {totals['recon']:.6f} | kl {totals['kl']:.6f} | "
		f"psnr {metrics['psnr']:.3f} | ssim {metrics['ssim']:.4f} | "
		f"n={n}"
	)

	# Optional: save side-by-side generated vs GT images for random test samples.
	if args.save_comparisons_dir:
		plt = _try_import_matplotlib()
		if plt is None:
			return

		out_dir = args.save_comparisons_dir
		os.makedirs(out_dir, exist_ok=True)
		k = int(args.num_comparisons)
		rng = random.Random(int(args.comparison_seed if args.comparison_seed is not None else seed))
		k = max(0, min(k, len(test_ds)))
		indices = rng.sample(range(len(test_ds)), k=k) if k > 0 else []
		print(f"Saving {len(indices)} comparisons to: {out_dir}")

		# Use a simple single-worker loop to keep index -> file mapping stable.
		# Each dataset item returns: c=[2,D,H,W], x=[1,D,H,W].
		for j, idx in enumerate(indices):
			c_cpu, x_cpu = test_ds[idx]
			c = c_cpu.unsqueeze(0).to(device)
			x = x_cpu.unsqueeze(0).to(device)

			# Normalize GT (match training) so visualization is comparable across samples.
			mn = x.amin(dim=(2, 3, 4), keepdim=True)
			mx = x.amax(dim=(2, 3, 4), keepdim=True)
			x01 = (x - mn) / (mx - mn + 1e-8)
			# Normalize condition to match x scale (same as training).
			c_img = c[:, 0:1]
			c_msk = c[:, 1:2]
			c_img = ((c_img - mn) / (mx - mn + 1e-8)) * c_msk
			c = torch.cat([c_img, c_msk], dim=1)

			with torch.no_grad():
				with autocast("cuda", enabled=(device.type == "cuda")):
					out = model(x=x01, c=c)
					x_gen = out.x_hat

			# Take a mid-slice along depth for a 2D PNG.
			D = int(x01.shape[-3])
			slice_idx = D // 2
			gt_2d = x01[0, 0, slice_idx].detach().float().cpu().numpy()
			gen_2d = x_gen[0, 0, slice_idx].detach().float().cpu().numpy()

			save_path = os.path.join(out_dir, f"cmp_{j:03d}_idx{idx}.png")
			_save_side_by_side_png(plt, save_path, gen_2d, gt_2d)

		print("Done saving comparisons.")


if __name__ == "__main__":
	main()

