import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
import argparse
import pandas as pd
import path_config

from lib.utils import get_masks_paper, eval_data_loader_df, eval_data_loader_crops_df, get_data_paths, str2bool, months_to_str, get_months_subdir, get_results_dir, build_model
from lib.dataloaders.dataloaders import CycleDataset

#######################################################################################


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
						help="Dataset split to evaluate on (train/val/test)")
	parser.add_argument("--selected_months", type=int, nargs="+", default=[3, 6, 9, 12],
						help="Which months to use (e.g., 3 6 9 12)")
	args = parser.parse_args()

	selected_months = args.selected_months
	months_sub = get_months_subdir(selected_months)

	device = "cuda"
	output_dir = get_results_dir(selected_months)
	best_param_df = pd.read_csv(os.path.join(output_dir, "best_params.csv"))
	model_names = best_param_df["Model Name"].values

	for model_name in model_names:

		data_percentage = float(model_name.split("_")[-1])

		# Map split name for data loading
		if args.split == "test":
			data_split = "testing"
		elif args.split == "val":
			data_split = "validation"
		else:  # train
			data_split = "training"

		data_loader_name = args.split

		if os.path.exists(f"{output_dir}/{model_name}_{data_loader_name}.csv"):
			print(f"Results for {model_name} on {data_loader_name} already exist, skipping...")
			continue

		config_dir = os.path.join(path_config.get_checkpoint_root(), months_sub, model_name)
		best_param = best_param_df[best_param_df["Model Name"] == model_name]["Best Param"].values[0]

		# Load checkpoint
		ckpt = torch.load(os.path.join(config_dir, best_param))
		n_timesteps = len(selected_months)
		months_str = months_to_str(selected_months)
		file_suffix = f"_m{months_str}"

		print(f"Best parameters: {best_param}")
		print(f"Selected months: {selected_months} (n_timesteps={n_timesteps})")

		data_path = get_data_paths(data_split, data_percentage, selected_months)
		cycle_dataset = CycleDataset(data_path, split=data_split, data_percentage=data_percentage, n_timesteps=n_timesteps, file_suffix=file_suffix)
		orig_means = cycle_dataset.means.copy()
		orig_stds  = cycle_dataset.stds.copy()

		data_loader = DataLoader(cycle_dataset, batch_size=2, shuffle=False, num_workers=2)

		try:
			model, feed_timeloc, crop_size = build_model(model_name, best_param, n_timesteps)
		except ValueError as e:
			print(f"Skipping: {e}")
			continue

		# Apply dataset settings matching this checkpoint's training config
		use_config_norm = "_confignorm-True" in best_param
		cycle_dataset.set_feed_timeloc(feed_timeloc)
		if use_config_norm:
			with open('configs/prithvi_300m.yaml', 'r') as f:
				norm_cfg = yaml.safe_load(f)
			cycle_dataset.means = np.array(norm_cfg["pretrained_cfg"]["mean"])
			cycle_dataset.stds  = np.array(norm_cfg["pretrained_cfg"]["std"])

		model = model.to(device)
		model.load_state_dict(ckpt["model_state_dict"])

		if crop_size is not None:
			out_df = eval_data_loader_crops_df(data_loader, model, device, get_masks_paper(data_loader_name), crop_size=crop_size, stride=path_config.get_eval_stride())
		else:
			out_df = eval_data_loader_df(data_loader, model, device, get_masks_paper(data_loader_name))

		# Reset dataset to defaults
		cycle_dataset.set_feed_timeloc(False)
		cycle_dataset.means = orig_means
		cycle_dataset.stds  = orig_stds

		out_df.to_csv(f"{output_dir}/{model_name}_{data_loader_name}.csv", index=False)


if __name__ == "__main__":
	main()
