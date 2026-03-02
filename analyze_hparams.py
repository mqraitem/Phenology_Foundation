"""Hyperparameter analysis for Prithvi crop models.

Usage:
  # Phase 1: Evaluate all checkpoints on validation (requires GPU, slow)
  python analyze_hparams.py --evaluate --selected_months 3 6 9 12

  # Phase 2: Print analysis tables from cached results (CPU-only, fast)
  python analyze_hparams.py --selected_months 3 6 9 12

  # Both at once:
  python analyze_hparams.py --evaluate --selected_months 3 6 9 12
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml

import path_config
from lib.utils import (
	get_data_paths, get_masks_paper, eval_data_loader_crops,
	build_model, parse_param, str2bool, months_to_str, get_months_subdir,
)
from lib.dataloaders.dataloaders import CycleDataset
from torch.utils.data import DataLoader


def parse_hparams_from_filename(filename):
	"""Extract all hyperparameters encoded in a checkpoint filename."""
	hparams = {}
	hparams["lr"] = parse_param(filename, "lr", default=None, cast=float)
	hparams["batch_size"] = parse_param(filename, "batch_size", default=None, cast=int)
	hparams["gradaccum"] = parse_param(filename, "gradaccum", default=1, cast=int)
	hparams["n_layers"] = parse_param(filename, "n_layers", default=2, cast=int)
	hparams["crop"] = parse_param(filename, "crop", default=48, cast=int)
	hparams["epochlen"] = parse_param(filename, "epochlen", default=5000, cast=int)
	hparams["backbone_lr_scale"] = parse_param(filename, "backbone_lr_scale", default=None, cast=float)
	hparams["feat"] = parse_param(filename, "feat", default=None, cast=str)
	hparams["hlsch"] = parse_param(filename, "hlsch", default=None, cast=int)
	hparams["hlsly"] = parse_param(filename, "hlsly", default=None, cast=int)
	hparams["progfusion"] = parse_param(filename, "progfusion", default=False, cast=str2bool)
	hparams["confignorm"] = parse_param(filename, "confignorm", default=False, cast=str2bool)
	hparams["feed_timeloc"] = parse_param(filename, "feed_timeloc", default=False, cast=str2bool)
	hparams["loss"] = parse_param(filename, "loss", default="mse", cast=str)

	# Effective batch size
	if hparams["batch_size"] is not None:
		hparams["eff_batch"] = hparams["batch_size"] * hparams["gradaccum"]

	return hparams


def evaluate_all_checkpoints(selected_months, cache_path):
	"""Evaluate every checkpoint on validation and cache results."""
	months_sub = get_months_subdir(selected_months)
	n_timesteps = len(selected_months)
	m_str = months_to_str(selected_months)
	file_suffix = f"_m{m_str}"

	device = "cuda"
	groups_dir = os.path.join(path_config.get_checkpoint_root(), months_sub)

	# Supported model families
	supported = [
		"prithvi_pretrained_crops_conv3d",
		"prithvi_pretrained_multiscale_crops_conv3d",
		"prithvi_pretrained_crops_cathls_conv3d",
	]

	all_groups = [g for g in os.listdir(groups_dir)
	              if any(s in g for s in supported)]

	# Load val data once
	path_val = get_data_paths("validation", 1.0, selected_months)
	cycle_dataset_val = CycleDataset(
		path_val, split="validation", data_percentage=1.0,
		n_timesteps=n_timesteps, file_suffix=file_suffix)
	orig_means = cycle_dataset_val.means.copy()
	orig_stds = cycle_dataset_val.stds.copy()
	val_dataloader = DataLoader(cycle_dataset_val, batch_size=1, shuffle=False, num_workers=2)

	tiles_paper_masks = get_masks_paper("train")

	# Load existing cache if present
	if os.path.exists(cache_path):
		results_df = pd.read_csv(cache_path)
		done_keys = set(results_df["checkpoint"].values)
		print(f"Loaded {len(done_keys)} cached results from {cache_path}")
	else:
		results_df = pd.DataFrame()
		done_keys = set()

	rows = []

	for group in sorted(all_groups):
		pth_dir = os.path.join(groups_dir, group)
		pth_files = sorted([p for p in os.listdir(pth_dir) if p.endswith(".pth")])

		for params_file in pth_files:
			if params_file in done_keys:
				print(f"  [cached] {params_file}")
				continue

			checkpoint_path = os.path.join(pth_dir, params_file)
			print(f"\nEvaluating: {group}/{params_file}")

			try:
				model, feed_timeloc, crop_size = build_model(group, params_file, n_timesteps)
			except ValueError as e:
				print(f"  Skipping: {e}")
				continue

			# Handle config normalization
			use_config_norm = "_confignorm-True" in params_file
			cycle_dataset_val.set_feed_timeloc(feed_timeloc)
			if use_config_norm:
				with open('configs/prithvi_300m.yaml', 'r') as f:
					norm_cfg = yaml.safe_load(f)
				cycle_dataset_val.means = np.array(norm_cfg["pretrained_cfg"]["mean"])
				cycle_dataset_val.stds = np.array(norm_cfg["pretrained_cfg"]["std"])

			model = model.to(device)
			ckpt = torch.load(checkpoint_path, weights_only=False)
			model.load_state_dict(ckpt["model_state_dict"])

			if crop_size is None:
				crop_size = 48

			acc, _, val_loss = eval_data_loader_crops(
				val_dataloader, model, device, tiles_paper_masks,
				crop_size=crop_size, stride=path_config.get_eval_stride())

			# Reset dataset
			cycle_dataset_val.set_feed_timeloc(False)
			cycle_dataset_val.means = orig_means
			cycle_dataset_val.stds = orig_stds

			hparams = parse_hparams_from_filename(params_file)
			row = {
				"group": group,
				"checkpoint": params_file,
				"val_loss": val_loss,
				"G": acc[0], "M": acc[1], "S": acc[2], "D": acc[3],
				"Mean": np.mean(list(acc.values())),
				"train_epoch": ckpt.get("epoch", None),
			}
			row.update(hparams)
			rows.append(row)

			# Free GPU memory
			del model
			torch.cuda.empty_cache()

			# Incremental save
			new_df = pd.DataFrame(rows)
			combined = pd.concat([results_df, new_df], ignore_index=True)
			combined.to_csv(cache_path, index=False)
			print(f"  G={acc[0]:.1f}  M={acc[1]:.1f}  S={acc[2]:.1f}  D={acc[3]:.1f}  Mean={np.mean(list(acc.values())):.1f}")

	# Final save
	if rows:
		new_df = pd.DataFrame(rows)
		results_df = pd.concat([results_df, new_df], ignore_index=True)
		results_df.to_csv(cache_path, index=False)

	return results_df


def print_separator(title):
	print(f"\n{'='*80}")
	print(f"  {title}")
	print(f"{'='*80}\n")


def analyze(cache_path):
	"""Print analysis tables from cached evaluation results."""
	df = pd.read_csv(cache_path)
	phases = ["G", "M", "S", "D", "Mean"]

	# Identify model families
	df["family"] = "unknown"
	df.loc[df["group"].str.contains("cathls"), "family"] = "cathls"
	df.loc[df["group"].str.contains("multiscale") & ~df["group"].str.contains("cathls"), "family"] = "multiscale"
	df.loc[~df["group"].str.contains("multiscale") & ~df["group"].str.contains("cathls"), "family"] = "baseline"

	# =====================================================================
	# Table 1: All runs ranked by Mean MAE
	# =====================================================================
	print_separator("TABLE 1: All runs ranked by validation Mean MAE")

	display_cols = ["family", "lr", "batch_size", "gradaccum", "eff_batch",
	                "backbone_lr_scale"] + phases
	# Add family-specific columns if present
	if df["feat"].notna().any():
		display_cols.insert(5, "feat")
	if df["progfusion"].notna().any():
		display_cols.insert(5, "progfusion")

	ranked = df.sort_values("Mean")[display_cols].reset_index(drop=True)
	ranked.index = ranked.index + 1  # 1-indexed rank
	ranked.index.name = "rank"
	print(ranked.round(1).to_string())

	# =====================================================================
	# Table 2: Per-family: best run
	# =====================================================================
	print_separator("TABLE 2: Best run per model family")
	for family in ["baseline", "multiscale", "cathls"]:
		fam_df = df[df["family"] == family]
		if fam_df.empty:
			continue
		best = fam_df.loc[fam_df["Mean"].idxmin()]
		print(f"\n--- {family.upper()} (n={len(fam_df)} runs) ---")
		print(f"  Checkpoint: {best['checkpoint']}")
		for p in phases:
			print(f"  {p}: {best[p]:.1f}", end="")
		print()

	# =====================================================================
	# Table 3: Effect of Learning Rate (within each family)
	# =====================================================================
	print_separator("TABLE 3: Effect of learning rate (mean across batch sizes)")
	for family in ["baseline", "multiscale", "cathls"]:
		fam_df = df[df["family"] == family]
		if fam_df.empty:
			continue
		pivot = fam_df.groupby("lr")[phases].mean().sort_index()
		print(f"\n--- {family.upper()} ---")
		print(pivot.round(1).to_string())

	# =====================================================================
	# Table 4: Effect of Batch Size / Effective Batch Size
	# =====================================================================
	print_separator("TABLE 4: Effect of effective batch size (mean across LRs)")
	for family in ["baseline", "multiscale", "cathls"]:
		fam_df = df[df["family"] == family]
		if fam_df.empty:
			continue
		pivot = fam_df.groupby("eff_batch")[phases].mean().sort_index()
		print(f"\n--- {family.upper()} ---")
		print(pivot.round(1).to_string())

	# =====================================================================
	# Table 5: Effect of Gradient Accumulation (baseline only)
	# =====================================================================
	baseline_df = df[df["family"] == "baseline"]
	if not baseline_df.empty and baseline_df["gradaccum"].nunique() > 1:
		print_separator("TABLE 5: Effect of gradient accumulation (baseline)")
		pivot = baseline_df.groupby(["lr", "gradaccum"])[phases].mean()
		print(pivot.round(1).to_string())

	# =====================================================================
	# Table 6: Effect of Backbone LR Scale
	# =====================================================================
	has_scale = df[df["backbone_lr_scale"].notna()]
	if not has_scale.empty:
		print_separator("TABLE 6: Effect of backbone_lr_scale")
		for family in has_scale["family"].unique():
			fam_df = has_scale[has_scale["family"] == family]
			if fam_df.empty:
				continue
			pivot = fam_df.groupby("backbone_lr_scale")[phases].mean().sort_index()
			print(f"\n--- {family.upper()} ---")
			print(pivot.round(1).to_string())

	# =====================================================================
	# Table 7: LR x Batch Size interaction (baseline)
	# =====================================================================
	if not baseline_df.empty:
		print_separator("TABLE 7: LR x Batch Size interaction — Mean MAE (baseline)")
		pivot = baseline_df.pivot_table(
			values="Mean", index="lr", columns="batch_size", aggfunc="mean")
		print(pivot.round(1).to_string())

	# =====================================================================
	# Table 8: Multiscale — Backbone LR Scale x LR
	# =====================================================================
	ms_df = df[df["family"] == "multiscale"]
	if not ms_df.empty and ms_df["backbone_lr_scale"].notna().any():
		print_separator("TABLE 8: Multiscale — backbone_lr_scale x LR — Mean MAE")
		pivot = ms_df.pivot_table(
			values="Mean", index="backbone_lr_scale", columns="lr", aggfunc="mean")
		print(pivot.round(1).to_string())

	# =====================================================================
	# Table 9: CatHLS — Fusion Mode comparison
	# =====================================================================
	cathls_df = df[df["family"] == "cathls"]
	if not cathls_df.empty and cathls_df["progfusion"].notna().any():
		print_separator("TABLE 9: CatHLS — progressive fusion comparison")
		pivot = cathls_df.groupby("progfusion")[phases].mean()
		print(pivot.round(1).to_string())

		# Also show fusion x LR
		print("\nFusion x LR — Mean MAE:")
		pivot2 = cathls_df.pivot_table(
			values="Mean", index="progfusion", columns="lr", aggfunc="mean")
		print(pivot2.round(1).to_string())

	# =====================================================================
	# Table 10: LR x Effective Batch — heatmap-style (all families)
	# =====================================================================
	print_separator("TABLE 10: LR x Effective Batch — Mean MAE (all Prithvi models)")
	pivot = df.pivot_table(
		values="Mean", index="lr", columns="eff_batch", aggfunc="mean")
	print(pivot.round(1).to_string())

	# =====================================================================
	# Summary: sensitivity ranking
	# =====================================================================
	print_separator("SUMMARY: Hyperparameter sensitivity (std of Mean MAE)")
	sensitivities = {}
	for col in ["lr", "batch_size", "eff_batch", "gradaccum", "backbone_lr_scale", "progfusion", "family"]:
		valid = df[df[col].notna()]
		if valid.empty or valid[col].nunique() < 2:
			continue
		group_means = valid.groupby(col)["Mean"].mean()
		sensitivities[col] = group_means.std()

	sens_df = pd.Series(sensitivities).sort_values(ascending=False)
	print("Hyperparameter         Std of group-mean MAE")
	print("-" * 45)
	for param, std in sens_df.items():
		print(f"  {param:25s} {std:.2f}")


def main():
	parser = argparse.ArgumentParser(description="Hyperparameter analysis for Prithvi crop models")
	parser.add_argument("--selected_months", type=int, nargs="+", default=[3, 6, 9, 12])
	parser.add_argument("--evaluate", action="store_true",
	                   help="Run evaluation on all checkpoints (requires GPU)")
	parser.add_argument("--cache_dir", type=str, default="results",
	                   help="Directory to store/load cached evaluation results")
	args = parser.parse_args()

	months_sub = get_months_subdir(args.selected_months)
	cache_dir = os.path.join(args.cache_dir, months_sub)
	os.makedirs(cache_dir, exist_ok=True)
	cache_path = os.path.join(cache_dir, "hparam_analysis_cache.csv")

	if args.evaluate:
		print(f"Evaluating all checkpoints, caching to {cache_path} ...")
		evaluate_all_checkpoints(args.selected_months, cache_path)

	if not os.path.exists(cache_path):
		print(f"No cached results found at {cache_path}")
		print("Run with --evaluate first (requires GPU)")
		return

	analyze(cache_path)


if __name__ == "__main__":
	main()
