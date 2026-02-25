import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
import argparse
from tqdm import tqdm
import pandas as pd
import path_config
import sys
sys.path.append("../")

from lib.utils import get_data_paths, eval_data_loader, eval_data_loader_crops, get_masks_paper, str2bool, months_to_str, get_months_subdir, get_results_dir, build_model
from lib.dataloaders.dataloaders import CycleDataset

#######################################################################################


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--selected_months", type=int, nargs="+", default=[3, 6, 9, 12],
						help="Which months to use (e.g., 3 6 9 12)")
	args = parser.parse_args()

	selected_months = args.selected_months
	months_sub = get_months_subdir(selected_months)
	n_timesteps = len(selected_months)
	m_str = months_to_str(selected_months)
	file_suffix = f"_m{m_str}"

	device = "cuda"
	groups_dir = os.path.join(path_config.get_checkpoint_root(), months_sub)

	if not os.path.exists(groups_dir):
		print(f"No checkpoint directory found at {groups_dir}")
		return

	all_groups = os.listdir(groups_dir)

	# Filter to supported model groups only
	supported = ["shallow_transformer_patch", "shallow_transformer_pixels",
	             "prithvi_pretrained_conv3d", "prithvi_random_conv3d",
	             "prithvi_pretrained_crops_conv3d",
	             "prithvi_pretrained_multiscale_crops_conv3d",
	             "prithvi_pretrained_crops_cathls_conv3d"]
	all_groups = [g for g in all_groups if any(s in g for s in supported)]

	results_dir = get_results_dir(selected_months)
	best_params_path = os.path.join(results_dir, "best_params.csv")

	if os.path.exists(best_params_path):
		param_df = pd.read_csv(best_params_path)
		best_param_df_cached = param_df.to_dict(orient="list")

		already_done_groups = best_param_df_cached["Model Name"]
		all_groups = [group for group in all_groups if group not in already_done_groups]

		best_param_df = {}
		best_param_df["Model Name"] = best_param_df_cached["Model Name"]
		best_param_df["Best Param"] = best_param_df_cached["Best Param"]
	else:
		best_param_df = {}
		best_param_df["Model Name"] = []
		best_param_df["Best Param"] = []


	for group in tqdm(all_groups):
		data_percentage = group.split("_")[-1]

		batch_size = 4 if "shallow_transformer" in group else 2

		# Load the first checkpoint to determine selected_months
		pth_files = [p for p in os.listdir(os.path.join(groups_dir, group)) if p.endswith(".pth")]
		if not pth_files:
			continue

		path_val = get_data_paths("validation", data_percentage, selected_months)

		cycle_dataset_val = CycleDataset(path_val, split="validation", data_percentage=data_percentage, n_timesteps=n_timesteps, file_suffix=file_suffix)
		orig_means = cycle_dataset_val.means.copy()
		orig_stds  = cycle_dataset_val.stds.copy()
		val_dataloader = DataLoader(cycle_dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)

		best_param = None
		best_acc = 1000

		for params in pth_files:

			checkpoint = os.path.join(groups_dir, group, params)
			print(f"Loading checkpoint: {checkpoint}")

			try:
				model, feed_timeloc, crop_size = build_model(group, params, n_timesteps)
			except ValueError as e:
				print(f"Skipping: {e}")
				continue

			# Apply dataset settings matching this checkpoint's training config
			use_config_norm = "_confignorm-True" in params
			cycle_dataset_val.set_feed_timeloc(feed_timeloc)
			if use_config_norm:
				with open('configs/prithvi_300m.yaml', 'r') as f:
					norm_cfg = yaml.safe_load(f)
				cycle_dataset_val.means = np.array(norm_cfg["pretrained_cfg"]["mean"])
				cycle_dataset_val.stds  = np.array(norm_cfg["pretrained_cfg"]["std"])

			model = model.to(device)
			model.load_state_dict(torch.load(checkpoint)["model_state_dict"])

			if crop_size is not None:
				acc_dataset_val, _, _ = eval_data_loader_crops(val_dataloader, model, device, get_masks_paper("train"), crop_size=crop_size)
			else:
				acc_dataset_val, _, _ = eval_data_loader(val_dataloader, model, device, get_masks_paper("train"))

			# Reset dataset to defaults for next checkpoint
			cycle_dataset_val.set_feed_timeloc(False)
			cycle_dataset_val.means = orig_means
			cycle_dataset_val.stds  = orig_stds

			print(f"Parameters: {params}")
			print(f"Val avg acc: {np.mean(list(acc_dataset_val.values()))}")

			if np.mean(list(acc_dataset_val.values())) < best_acc:
				best_acc = np.mean(list(acc_dataset_val.values()))
				best_param = params

		print(f"Best parameters: {best_param}")
		best_param_df["Model Name"].append(group)
		best_param_df["Best Param"].append(best_param)

		best_param_df_to_save = pd.DataFrame(best_param_df)
		best_param_df_to_save.to_csv(best_params_path, index=False)

if __name__ == "__main__":
	main()
