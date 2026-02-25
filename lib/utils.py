import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
import path_config


def months_to_str(selected_months):
	return "-".join(str(m) for m in sorted(selected_months))

def get_months_subdir(selected_months):
	return f"m{months_to_str(selected_months)}"

def get_checkpoint_dir(group_name, data_percentage, selected_months):
	months_sub = get_months_subdir(selected_months)
	d = os.path.join(path_config.get_checkpoint_root(), months_sub, f"{group_name}_{data_percentage}")
	os.makedirs(d, exist_ok=True)
	return d

def get_results_dir(selected_months):
	months_sub = get_months_subdir(selected_months)
	d = os.path.join("results", months_sub)
	os.makedirs(d, exist_ok=True)
	return d

def normalize_doy(day_of_year):
	return day_of_year / 547

def compute_or_load_means_stds(
	data_dir,
	split,
	data_percentage,
	num_bands,
	load_raster_fn,
	file_suffix="",
):
	if file_suffix.startswith("_m"):
		months_sub = file_suffix[1:]  # "_m3-6-9-12" -> "m3-6-9-12"
		mean_stds_dir = os.path.join(path_config.get_mean_stds_dir(), months_sub)
	else:
		mean_stds_dir = path_config.get_mean_stds_dir()
	os.makedirs(mean_stds_dir, exist_ok=True)
	means_stds_path = f"{mean_stds_dir}/means_stds_{data_percentage}{file_suffix}.pkl"

	# Load precomputed means and stds if available
	if os.path.exists(means_stds_path):
		with open(means_stds_path, 'rb') as f:
			means, stds = pickle.load(f)
		return means, stds

	# Don't compute stats for test split
	if split in ["test", "testing"]:
		raise ValueError("Cannot compute mean and std for test split")

	# Initialize accumulators for Chan et al. algorithm
	global_mean = np.zeros(num_bands, dtype=np.float64)
	global_var = np.zeros(num_bands, dtype=np.float64)
	global_count = np.zeros(num_bands, dtype=np.float64)

	for i in tqdm(range(len(data_dir)), desc=f"Computing stats for {num_bands} band(s)"):
		image_path = data_dir[i][0]
		gt_path = data_dir[i][1]

		# Load all time steps for this sample
		images = []
		for path in image_path:
			images.append(load_raster_fn(path)[:, np.newaxis])

		img = np.concatenate(images, axis=1)  # shape: (num_bands, time_steps, H, W)

		# Create mask for dead pixels (zeros in both time and bands dimensions)
		# A pixel is dead if it's zero across all bands and all time steps
		dead_pixel_mask = np.all(img == 0, axis=(0, 1))  # shape: (H, W)

		# Expand mask to match image shape: (num_bands, time_steps, H, W)
		time_steps = img.shape[1]
		expanded_mask = np.repeat(dead_pixel_mask[np.newaxis, :, :], time_steps, axis=0)  # (time_steps, H, W)
		expanded_mask = np.repeat(expanded_mask[np.newaxis, :, :, :], num_bands, axis=0)  # (num_bands, time_steps, H, W)

		img_flat = img.reshape(num_bands, -1)
		mask_flat = ~expanded_mask.reshape(num_bands, -1)

		# Compute statistics per band using Chan et al. algorithm
		for b in range(num_bands):
			valid_values = img_flat[b][mask_flat[b]]
			# valid_values = img_flat[b]

			n = len(valid_values)
			if n == 0:
				continue

			batch_mean = valid_values.mean()
			batch_var = valid_values.var(ddof=0)  # population variance

			m = global_count[b]
			mu1 = global_mean[b]
			mu2 = batch_mean
			v1 = global_var[b]
			v2 = batch_var

			# Combine means and variances
			combined_mean = (m / (m + n)) * mu1 + (n / (m + n)) * mu2 if (m + n) > 0 else mu2
			combined_var = (
				(m / (m + n)) * v1
				+ (n / (m + n)) * v2
				+ (m * n / (m + n) ** 2) * (mu1 - mu2) ** 2
				if (m + n) > 0
				else v2
			)

			global_mean[b] = combined_mean
			global_var[b] = combined_var
			global_count[b] = m + n

	means = global_mean
	stds = np.sqrt(global_var)

	print("Mean: ", means)
	print("Stds: ", stds)

	# Cache the computed statistics
	with open(means_stds_path, 'wb') as f:
		pickle.dump([means, stds], f)

	return means, stds


def print_trainable_parameters(model, detailed=False):
	"""
	Prints the number of trainable parameters in the model.
	If detailed=True, also prints breakdown by top-level module.
	"""
	trainable_params = 0
	all_param = 0
	module_params = {}  # Track params per top-level module

	for name, param in model.named_parameters():
		num_params = param.numel()
		all_param += num_params

		# Get top-level module name (e.g., 'backbone', 'head')
		top_level = name.split('.')[0] if '.' in name else name

		if top_level not in module_params:
			module_params[top_level] = {'trainable': 0, 'total': 0}
		module_params[top_level]['total'] += num_params

		if param.requires_grad:
			trainable_params += num_params
			module_params[top_level]['trainable'] += num_params

	print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}%")

	if detailed:
		print("\nParameter breakdown by module:")
		print("-" * 60)
		for module_name, counts in sorted(module_params.items()):
			trainable = counts['trainable']
			total = counts['total']
			pct = 100 * trainable / total if total > 0 else 0
			print(f"  {module_name}: {trainable:,} / {total:,} ({pct:.2f}% trainable)")


def segmentation_loss_pixels(targets, preds, device, ignore_index=-1):
	"""
	Compute regression loss for pixel dataset.
	
	Args:
		targets: (B,) tensor of ground truth labels (float)
		preds:   (B,) or (B, num_outputs) tensor of predictions
		device:  torch device
		ignore_index: value in targets to ignore (default -1)
	"""
	criterion = nn.MSELoss(reduction="sum").to(device)

	# valid mask = targets not equal to ignore_index
	valid_mask = targets != ignore_index

	if valid_mask.sum() > 0:
		valid_pred = preds[valid_mask]
		valid_target = targets[valid_mask]   # normalize like before
		loss = criterion(valid_pred, valid_target)
		return loss / valid_mask.sum().item()
	else:
		return torch.tensor(0.0, device=device)


def segmentation_loss_pixels_mae(targets, preds, device, ignore_index=-1):
	"""
	Compute MAE regression loss for pixel dataset.

	Args:
		targets: (B,) tensor of ground truth labels (float)
		preds:   (B,) or (B, num_outputs) tensor of predictions
		device:  torch device
		ignore_index: value in targets to ignore (default -1)
	"""
	criterion = nn.L1Loss(reduction="sum").to(device)

	valid_mask = targets != ignore_index

	if valid_mask.sum() > 0:
		valid_pred = preds[valid_mask]
		valid_target = targets[valid_mask]
		loss = criterion(valid_pred, valid_target)
		return loss / valid_mask.sum().item()
	else:
		return torch.tensor(0.0, device=device)


def segmentation_loss_subsampled(mask, pred, device, n_pixels=256, ignore_index=-1, loss_type="mse"):
	"""Compute loss on a random subset of valid pixels per image.

	Args:
		mask:       (B, K, H, W) ground truth
		pred:       (B, K, H, W) predictions
		device:     torch device
		n_pixels:   number of pixels to sample per image
		ignore_index: value to ignore in mask
		loss_type:  "mse" or "mae"
	"""
	mask = mask.float()
	criterion = nn.L1Loss(reduction="sum").to(device) if loss_type == "mae" else nn.MSELoss(reduction="sum").to(device)

	B, K = pred.shape[0], pred.shape[1]
	loss = 0
	total_sampled = 0

	for b in range(B):
		# Valid pixel: not ignore_index in ALL channels
		valid = torch.ones(mask.shape[2], mask.shape[3], dtype=torch.bool, device=device)
		for k in range(K):
			valid = valid & (mask[b, k] != ignore_index)

		valid_indices = valid.nonzero(as_tuple=False)  # (N_valid, 2)
		if valid_indices.shape[0] == 0:
			continue

		# Random subsample
		n_sample = min(n_pixels, valid_indices.shape[0])
		perm = torch.randperm(valid_indices.shape[0], device=device)[:n_sample]
		sampled = valid_indices[perm]  # (n_sample, 2)
		rows, cols = sampled[:, 0], sampled[:, 1]

		for k in range(K):
			loss += criterion(pred[b, k, rows, cols], mask[b, k, rows, cols])
		total_sampled += n_sample * K

	return loss / total_sampled if total_sampled > 0 else torch.tensor(0.0, device=device)


def segmentation_loss(mask, pred, device, ignore_index=-1):
	mask = mask.float()  # Convert mask to float for regression loss

	criterion = nn.MSELoss(reduction="sum").to(device)

	loss = 0
	num_channels = pred.shape[1]  # Number of output channels
	total_valid_pixels = 0  # Counter for valid pixels

	for idx in range(num_channels):
		valid_mask = mask[:, idx] != ignore_index

		if valid_mask.sum() > 0:  # Ensure there are valid pixels to compute loss

			valid_pred = pred[:, idx][valid_mask]  # Apply mask to predictions
			valid_target = mask[:, idx][valid_mask]  # Apply mask to ground truth

			loss += criterion(valid_pred, valid_target)
			total_valid_pixels += valid_mask.sum().item()

	# Normalize by total valid pixels to avoid division by zero
	return loss / total_valid_pixels if total_valid_pixels > 0 else torch.tensor(0.0, device=device)


def segmentation_loss_mae(mask, pred, device, ignore_index=-1):
	mask = mask.float()

	criterion = nn.L1Loss(reduction="sum").to(device)

	loss = 0
	num_channels = pred.shape[1]
	total_valid_pixels = 0

	for idx in range(num_channels):
		valid_mask = mask[:, idx] != ignore_index

		if valid_mask.sum() > 0:
			valid_pred = pred[:, idx][valid_mask]
			valid_target = mask[:, idx][valid_mask]

			loss += criterion(valid_pred, valid_target)
			total_valid_pixels += valid_mask.sum().item()

	return loss / total_valid_pixels if total_valid_pixels > 0 else torch.tensor(0.0, device=device)


def get_masks_paper(data="train", device="cuda"):

	data_name = "train" if data in ["train", "val"] else "test"
	test_file = f"data/LSP_{data_name}_samples.csv"
	
	data_paper_df = pd.read_csv(test_file)
	data_paper_df = data_paper_df[data_paper_df["version"] == "v1"]

	tiles_paper_masks = {}

	# Group by (year, tile)
	for (year, site_id, tile), group in data_paper_df.groupby(['years', "SiteID", 'tile']):
		# Initialize 320x320 mask with False
		mask = np.zeros((330, 330), dtype=bool)
		
		#subtract by 1 except 0 
		# Set True where (row, col) is mentioned in the group
		mask[group['row'].values, group['col'].values] = True
		
		# Store the mask with key (year, tile)

		mask = torch.Tensor(mask).bool().to(device)
		tiles_paper_masks[f"{year}_{site_id}_{tile}"] = mask
	
	return tiles_paper_masks

def compute_accuracy(gt_hls_tile, pred_hls_tile_avg, all_errors_hls_tile, hls_tile_n, paper_mask):

	for idx in range(4): 

		pred_idx = pred_hls_tile_avg[idx]
		gt_idx = gt_hls_tile[idx]

		pred_idx = pred_idx.flatten()
		gt_idx = gt_idx.flatten()

		mask = (gt_idx != -1) & paper_mask.flatten()
		pred_idx = pred_idx[mask]
		gt_idx = gt_idx[mask]

		errors = (pred_idx - gt_idx).detach().cpu().numpy() * 547
		all_errors_hls_tile[hls_tile_n][idx] = np.mean(np.abs(errors))

	return all_errors_hls_tile


def eval_data_loader(data_loader,model, device, tiles_paper_masks, loss_fn=None):

	if loss_fn is None:
		loss_fn = segmentation_loss

	model.eval()

	all_errors_hls_tile = {}

	eval_loss = 0.0
	with torch.no_grad():
		for _,data in tqdm(enumerate(data_loader), total=len(data_loader)):


			input = data["image"]
			ground_truth = data["gt_mask"].to(device)
			predictions=model(input)

			predictions = predictions[:, :, :330, :330]
			eval_loss += loss_fn(mask=data["gt_mask"].to(device),pred=predictions,device=device).item() * ground_truth.size(0)  # Multiply by batch size

			pred_hls_tile_all = predictions  # Average over the last dimension	

			for gt_hls_tile, pred_hls_tile_avg, hls_tile_n in zip(ground_truth, pred_hls_tile_all, data["hls_tile_name"]): 

				assert hls_tile_n not in all_errors_hls_tile, f"Tile {hls_tile_n} already exists in all_errors_hls_tile"
				all_errors_hls_tile[hls_tile_n] = {i:0 for i in range(4)}  # Initialize errors for each of the 4 predicted dates
				all_errors_hls_tile = compute_accuracy(gt_hls_tile, pred_hls_tile_avg, all_errors_hls_tile, hls_tile_n, tiles_paper_masks[hls_tile_n])

	all_errors_time = {i:[] for i in range(4)}
	for tile in all_errors_hls_tile:
		for i in range(4):
			all_errors_time[i].append(all_errors_hls_tile[tile][i])

	acc_dataset_val = {i:np.mean(all_errors_time[i]) for i in range(4)}
	epoch_loss_val = eval_loss / len(data_loader.dataset)

	return acc_dataset_val, all_errors_hls_tile, epoch_loss_val


def eval_data_loader_crops(data_loader, model, device, tiles_paper_masks,
						   crop_size=48, stride=48, loss_fn=None):
	"""Evaluate using sliding-window crops.

	For each tile, slide a crop_size x crop_size window with the given stride.
	Run each crop through the model, accumulate logits, and average overlapping
	predictions. The model always sees crop_size x crop_size inputs.

	Args:
		data_loader: yields tiles with "image" (C, T, H, W), "gt_mask", "hls_tile_name"
		model:       trained on crop_size x crop_size inputs
		device:      "cuda"
		tiles_paper_masks: dict of per-tile evaluation masks
		crop_size:   spatial size the model expects (default 48)
		stride:      sliding window stride (default 48 = non-overlapping)
		loss_fn:     loss function for eval loss computation
	"""
	if loss_fn is None:
		loss_fn = segmentation_loss

	model.eval()
	all_errors_hls_tile = {}
	eval_loss = 0.0
	tile_size = 330

	# Precompute crop positions
	positions = []
	r = 0
	while r + crop_size <= tile_size:
		positions.append(r)
		r += stride
	# Add edge position if last crop doesn't reach the end
	if positions[-1] + crop_size < tile_size:
		positions.append(tile_size - crop_size)

	with torch.no_grad():
		for _, data in tqdm(enumerate(data_loader), total=len(data_loader)):

			image = data["image"]             # (B, C, T, H, W) — may be padded to 336
			gt = data["gt_mask"].to(device)    # (B, 4, H, W)
			tile_name = data["hls_tile_name"]
			B = image.size(0)

			# Work on the 330x330 valid region
			image_330 = image[:, :, :, :tile_size, :tile_size]

			# Accumulators
			logit_sum = torch.zeros(B, 4, tile_size, tile_size, device=device)
			count = torch.zeros(B, 1, tile_size, tile_size, device=device)

			# Slide crop_size x crop_size window
			for r in positions:
				for c in positions:
					crop = image_330[:, :, :, r:r+crop_size, c:c+crop_size]
					pred_crop = model(crop)
					pred_crop = pred_crop[:, :, :crop_size, :crop_size]

					logit_sum[:, :, r:r+crop_size, c:c+crop_size] += pred_crop
					count[:, :, r:r+crop_size, c:c+crop_size] += 1

			# Average overlapping predictions
			predictions = logit_sum / count.clamp(min=1)

			gt_330 = gt[:, :, :tile_size, :tile_size]

			eval_loss += loss_fn(
				mask=gt_330, pred=predictions, device=device
			).item() * gt_330.size(0)

			for gt_tile, pred_tile, name in zip(gt_330, predictions, tile_name):
				assert name not in all_errors_hls_tile, f"Tile {name} already exists"
				all_errors_hls_tile[name] = {i: 0 for i in range(4)}
				all_errors_hls_tile = compute_accuracy(
					gt_tile, pred_tile, all_errors_hls_tile, name,
					tiles_paper_masks[name],
				)

	all_errors_time = {i: [] for i in range(4)}
	for tile in all_errors_hls_tile:
		for i in range(4):
			all_errors_time[i].append(all_errors_hls_tile[tile][i])

	acc_dataset = {i: np.mean(all_errors_time[i]) for i in range(4)}
	epoch_loss = eval_loss / len(data_loader.dataset)

	return acc_dataset, all_errors_hls_tile, epoch_loss


def eval_data_loader_crops_df(data_loader, model, device, tiles_paper_masks,
							  crop_size=48, stride=48):
	"""Evaluate using sliding-window crops and return per-pixel DataFrame.

	Same sliding-window logic as eval_data_loader_crops, but outputs a DataFrame
	with per-pixel predictions (matching eval_data_loader_df output format).
	"""
	model.eval()
	tile_size = 330

	# Precompute crop positions
	positions = []
	r = 0
	while r + crop_size <= tile_size:
		positions.append(r)
		r += stride
	if positions[-1] + crop_size < tile_size:
		positions.append(tile_size - crop_size)

	data_df = {
		"index":[], "years":[], "HLStile":[], "SiteID": [],
		"row":[], "col":[], "version":[],
		"G_pred_DOY":[], "M_pred_DOY":[], "S_pred_DOY":[], "D_pred_DOY":[],
		"G_truth_DOY":[], "M_truth_DOY":[], "S_truth_DOY":[], "D_truth_DOY":[],
		"n_missing_ts":[], "lat":[], "lon":[],
	}

	# Tile centroid lat/lon lookup from the dataset
	all_locations = data_loader.dataset.all_locations

	with torch.no_grad():
		for _, data in tqdm(enumerate(data_loader), total=len(data_loader)):
			image = data["image"]
			ground_truth = data["gt_mask"].to(device)
			tile_name = data["hls_tile_name"]

			image_330 = image[:, :, :, :tile_size, :tile_size]

			logit_sum = torch.zeros(image.size(0), 4, tile_size, tile_size, device=device)
			count = torch.zeros(image.size(0), 1, tile_size, tile_size, device=device)

			for ri in positions:
				for ci in positions:
					crop = image_330[:, :, :, ri:ri+crop_size, ci:ci+crop_size]
					pred_crop = model(crop)
					pred_crop = pred_crop[:, :, :crop_size, :crop_size]
					logit_sum[:, :, ri:ri+crop_size, ci:ci+crop_size] += pred_crop
					count[:, :, ri:ri+crop_size, ci:ci+crop_size] += 1

			predictions = logit_sum / count.clamp(min=1)

			img_unproc = data["image_unprocessed"].numpy() \
				if hasattr(data["image_unprocessed"], "numpy") \
				else np.array(data["image_unprocessed"])

			for b_idx, (gt_tile, pred_tile, name) in enumerate(
					zip(ground_truth, predictions, tile_name)):

				mask_tilen = tiles_paper_masks[name]
				year, siteid, hlstile = name.split("_")
				row, col = np.where(mask_tilen.cpu().numpy())

				loc = all_locations[name]
				lat, lon = loc[0], loc[1]

				sample_img = img_unproc[b_idx]

				for r, c in zip(row, col):
					pixel_ts = sample_img[:, :, r, c]
					n_missing = int((pixel_ts == 0).all(axis=0).sum())

					data_df["index"].append(len(data_df["index"]))
					data_df["years"].append(year)
					data_df["HLStile"].append(hlstile)
					data_df["SiteID"].append(siteid)
					data_df["row"].append(r)
					data_df["col"].append(c)
					data_df["version"].append("v1")
					data_df["G_pred_DOY"].append(pred_tile[0, r, c].item()*547)
					data_df["M_pred_DOY"].append(pred_tile[1, r, c].item()*547)
					data_df["S_pred_DOY"].append(pred_tile[2, r, c].item()*547)
					data_df["D_pred_DOY"].append(pred_tile[3, r, c].item()*547)
					data_df["G_truth_DOY"].append(gt_tile[0, r, c].item()*547)
					data_df["M_truth_DOY"].append(gt_tile[1, r, c].item()*547)
					data_df["S_truth_DOY"].append(gt_tile[2, r, c].item()*547)
					data_df["D_truth_DOY"].append(gt_tile[3, r, c].item()*547)
					data_df["n_missing_ts"].append(n_missing)
					data_df["lat"].append(lat)
					data_df["lon"].append(lon)

	data_df = pd.DataFrame(data_df)
	return data_df


def eval_data_loader_df(data_loader,model, device, tiles_paper_masks):

	model.eval()

	#G_pred_DOY  M_pred_DOY  S_pred_DOY  D_pred_DOY
	data_df = {
		"index":[],
		"years":[],
		"HLStile":[],
		"SiteID": [],
		"row":[],
		"col":[],
		"version":[],
		"G_pred_DOY":[],
		"M_pred_DOY":[],
		"S_pred_DOY":[],
		"D_pred_DOY":[],
		"G_truth_DOY":[],
		"M_truth_DOY":[],
		"S_truth_DOY":[],
		"D_truth_DOY":[],
		"n_missing_ts":[],   # number of all-zero timesteps for this pixel (0–12)
		"lat":[],
		"lon":[],
	}

	# Tile centroid lat/lon lookup from the dataset
	all_locations = data_loader.dataset.all_locations

	eval_loss = 0.0
	with torch.no_grad():
		for _,data in tqdm(enumerate(data_loader), total=len(data_loader)):

			input = data["image"]
			ground_truth = data["gt_mask"].to(device)
			hls_tile_name = data["hls_tile_name"]

			predictions=model(input)

			predictions = predictions[:, :, :330, :330]
			pred_hls_tile_all = predictions

			eval_loss += segmentation_loss(mask=data["gt_mask"].to(device),pred=predictions,device=device).item() * ground_truth.size(0)  # Multiply by batch size

			# image_unprocessed: (B, C, T, H, W) raw reflectance — zeros = missing
			img_unproc = data["image_unprocessed"].numpy() \
				if hasattr(data["image_unprocessed"], "numpy") \
				else np.array(data["image_unprocessed"])   # (B, C, T, H, W)

			for b_idx, (gt_hls_tile, pred_hls_tile_avg, hls_tile_n) in enumerate(
					zip(ground_truth, pred_hls_tile_all, data["hls_tile_name"])):

				mask_tilen = tiles_paper_masks[hls_tile_n]
				year, siteid, hlstile = hls_tile_n.split("_")
				#get the row and col from the mask
				row, col = np.where(mask_tilen.cpu().numpy())

				loc = all_locations[hls_tile_n]
				lat, lon = loc[0], loc[1]

				# (C, T, H, W) for this sample
				sample_img = img_unproc[b_idx]

				for r, c in zip(row, col):
					# count timesteps where all 6 bands are zero at this pixel
					pixel_ts  = sample_img[:, :, r, c]          # (C, T)
					n_missing = int((pixel_ts == 0).all(axis=0).sum())

					data_df["index"].append(len(data_df["index"]))
					data_df["years"].append(year)
					data_df["HLStile"].append(hlstile)
					data_df["SiteID"].append(siteid)
					data_df["row"].append(r)
					data_df["col"].append(c)
					data_df["version"].append("v1")
					data_df["G_pred_DOY"].append(pred_hls_tile_avg[0, r, c].item()*547)
					data_df["M_pred_DOY"].append(pred_hls_tile_avg[1, r, c].item()*547)
					data_df["S_pred_DOY"].append(pred_hls_tile_avg[2, r, c].item()*547)
					data_df["D_pred_DOY"].append(pred_hls_tile_avg[3, r, c].item()*547)
					data_df["G_truth_DOY"].append(gt_hls_tile[0, r, c].item()*547)
					data_df["M_truth_DOY"].append(gt_hls_tile[1, r, c].item()*547)
					data_df["S_truth_DOY"].append(gt_hls_tile[2, r, c].item()*547)
					data_df["D_truth_DOY"].append(gt_hls_tile[3, r, c].item()*547)
					data_df["n_missing_ts"].append(n_missing)
					data_df["lat"].append(lat)
					data_df["lon"].append(lon)

	data_df = pd.DataFrame(data_df)
	return data_df	


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filename, selected_months=None):

	checkpoint = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'train_loss': train_loss,
		'val_loss':val_loss
	}
	if selected_months is not None:
		checkpoint['selected_months'] = selected_months
	torch.save(checkpoint, filename)
	print(f"Checkpoint saved at {filename}")
	
def get_layer_lr_groups(model, head_lr, backbone_lr, layer_decay=0.75):
	"""Build per-layer param groups with geometrically decaying LR.

	Later backbone layers get higher LR (closer to backbone_lr),
	earlier layers get lower LR (backbone_lr * decay^distance_from_top).

	Works with both PrithviSegConv3D (PrithviBackbone) and mosaic models
	(MaskedPrithviBackbone) since both expose encoder blocks via
	model.backbone.model.encoder.blocks.

	Args:
		model: a PrithviSeg* model with .backbone, .reshaper, .head
		head_lr: learning rate for head (and reshaper)
		backbone_lr: peak learning rate for the last backbone block
		layer_decay: multiplicative decay per layer (0-1)

	Returns:
		list of param group dicts for the optimizer
	"""
	encoder = model.backbone.model.encoder
	num_layers = len(encoder.blocks)

	param_groups = []
	seen_params = set()

	# 1) Head parameters — full learning rate
	head_params = list(model.head.parameters())
	if head_params:
		param_groups.append({
			'params': head_params,
			'lr': head_lr,
			'name': 'head',
		})
		seen_params.update(id(p) for p in head_params)

	# 2) Reshaper parameters (PrithviReshape3D has none, but future-proof)
	reshaper_params = [p for p in model.reshaper.parameters() if id(p) not in seen_params]
	if reshaper_params:
		param_groups.append({
			'params': reshaper_params,
			'lr': head_lr,
			'name': 'reshaper',
		})
		seen_params.update(id(p) for p in reshaper_params)

	# 3) Backbone blocks — layer-wise decay
	#    Block i gets: backbone_lr * decay^(num_layers - 1 - i)
	#    Block 0 (earliest) -> smallest LR, block N-1 (latest) -> backbone_lr
	for i, block in enumerate(encoder.blocks):
		block_params = [p for p in block.parameters() if id(p) not in seen_params]
		if block_params:
			layer_lr = backbone_lr * (layer_decay ** (num_layers - 1 - i))
			param_groups.append({
				'params': block_params,
				'lr': layer_lr,
				'name': f'backbone.block.{i}',
			})
			seen_params.update(id(p) for p in block_params)

	# 4) Remaining backbone params (patch_embed, cls_token, norm) — lowest LR
	remaining = [p for p in model.backbone.parameters()
				 if id(p) not in seen_params and p.requires_grad]
	if remaining:
		lowest_lr = backbone_lr * (layer_decay ** num_layers)
		param_groups.append({
			'params': remaining,
			'lr': lowest_lr,
			'name': 'backbone.other',
		})

	return param_groups


def get_data_paths(mode, data_percentage=1.0, selected_months=None):

	if selected_months is None:
		selected_months = list(range(1, 13))

	data_dir_name = f"{data_percentage}_m{months_to_str(selected_months)}"
	checkpoint_data = f"{path_config.get_data_paths_dir()}/{data_dir_name}/"

	if os.path.exists(f'{checkpoint_data}/data_pths_{mode}.pkl'):
		with open(f'{checkpoint_data}/data_pths_{mode}.pkl', 'rb') as f:
			data_dir = pickle.load(f)

		return data_dir

	hls_path = path_config.get_data_hls_composites()
	lsp_path = path_config.get_data_lsp_ancillary()

	hls_tiles = [x for x in os.listdir(hls_path) if x.endswith('.tif')]
	lsp_tiles = []
	lsp_tiles.extend([x for x in os.listdir(f"{lsp_path}/A2019") if x.endswith('.tif')])
	lsp_tiles.extend([x for x in os.listdir(f"{lsp_path}/A2020") if x.endswith('.tif')])

	title_hls = ['_'.join(x.split('_')[3:5]).split(".")[0] for x in hls_tiles]
	title_hls = set(title_hls)

	title_hls_lsp = ["_".join(x.split('_')[3:5]) for x in lsp_tiles]
	title_hls_lsp = set(title_hls_lsp)

	hls_tiles_time = []
	lsp_tiles_time = []
	hls_tiles_name = []

	for year in ["2019", "2020"]:
		past_months = selected_months
		timesteps = [f"{year}-{str(x).zfill(2)}" for x in past_months]

		for hls_tile in tqdm(title_hls):
			hls_tile_location = hls_tile.split("_")[0]
			hls_tile_name = hls_tile.split("_")[1]
			temp_ordered = []

			for timestep in timesteps:
				temp_ordered.append(f"{hls_path}/HLS_composite_{timestep}_{hls_tile_location}_{hls_tile_name}.tif")

			lsp_filename = f"HLS_PhenoCam_A{year}_{hls_tile_location}_{hls_tile_name}_LSP_Date.tif"
			assert lsp_filename in lsp_tiles, f"GT file not found: {lsp_filename}"
			temp_lsp = f"{lsp_path}/A{year}/{lsp_filename}"

			hls_tiles_time.append(temp_ordered)
			lsp_tiles_time.append(temp_lsp)
			hls_tiles_name.append(f"{year}_{hls_tile}")

	#open training file
	with open(f"{lsp_path}/HP-LSP_train_ids.csv", 'r') as f:
		train_ids = f.readlines()[0].replace("'", "").split(",")
		train_ids = [x.strip() for x in train_ids]

	with open(f"{lsp_path}/HP-LSP_test_ids.csv", 'r') as f:
		test_ids = f.readlines()[0].replace("'", "").split(",")
		test_ids = [x.strip() for x in test_ids]


	hls_tiles_val = [
		"2019_ME-1_T19TEL",
		"2019_FL-3_T17RML",
		"2020_WI-2_T15TYL",
		"2019_AZ-5_T12SVE",
		"2020_CO-2_T13TDE",
		"2020_OR-1_T10TEQ",
		"2019_MD-1_T18SUJ",
		"2020_ND-1_T14TLS"
	]

	hls_tiles_train = [x for x in hls_tiles_name if x.split("_")[1] in train_ids]
	hls_tiles_train = [x for x in hls_tiles_train if x not in hls_tiles_val]

	# Apply data_percentage globally to all training data
	num_to_keep = int(len(hls_tiles_train) * data_percentage)
	hls_tiles_train = hls_tiles_train[:num_to_keep]
	hls_tiles_test = [x for x in hls_tiles_name if x.split("_")[1] in test_ids]
	data_dir_train = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_train]

	data_dir_val = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_val]
	data_dir_test = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_test]

	os.makedirs(checkpoint_data, exist_ok=True)
	with open(f'{checkpoint_data}/data_pths_training.pkl', 'wb') as f:
		pickle.dump(data_dir_train, f)

	with open(f'{checkpoint_data}/data_pths_validation.pkl', 'wb') as f:
		pickle.dump(data_dir_val, f)

	with open(f'{checkpoint_data}/data_pths_testing.pkl', 'wb') as f:
		pickle.dump(data_dir_test, f)

	if mode == 'training':
		return data_dir_train
	elif mode == 'validation':
		return data_dir_val
	elif mode == "testing":
		return data_dir_test
	else:
		raise ValueError(f"Unknown mode: {mode}. Expected 'training', 'validation', or 'testing'.")


def parse_param(filename, key, default=None, cast=str):
	"""Parse a parameter value from a filename like 'prefix_key-value_suffix.pth'."""
	search = f"_{key}-"
	if search not in filename:
		return default
	try:
		val = filename.split(search)[1].split("_")[0].replace(".pth", "")
		return cast(val)
	except (IndexError, ValueError):
		return default


def build_model(group, params, n_timesteps):
	"""Build the right model architecture for a given group/checkpoint.

	Args:
		group:  model group name (directory name, e.g. "prithvi_pretrained_conv3d_1.0")
		params: checkpoint filename (e.g. "prithvi_pretrained_conv3d_lr-0.0001_...pth")
		n_timesteps: number of temporal input frames

	Returns:
		(model, feed_timeloc, crop_size) where crop_size is None for
		full-tile models or an int for crop-trained models.
		Raises ValueError for unknown groups.
	"""
	import yaml

	feed_timeloc = False
	crop_size = None

	# --- Shallow transformer (patches) ---
	if "shallow_transformer_patch" in group:
		from lib.models.lsp_transformer_patches import TemporalTransformerPerPatch
		patch_size = int(group.split("patch-")[1].split("_")[0])
		model = TemporalTransformerPerPatch(
			input_channels=6, seq_len=n_timesteps, num_classes=4,
			d_model=128, nhead=4, num_layers=3, dropout=0.1,
			patch_size=(patch_size, patch_size),
		)
		return model, feed_timeloc, crop_size

	# --- Shallow transformer (pixels) ---
	if "shallow_transformer_pixels" in group:
		from lib.models.lsp_transformer_pixels import TemporalTransformer
		model = TemporalTransformer(
			input_channels=6, seq_len=n_timesteps, num_classes=4,
			d_model=128, nhead=4, num_layers=3, dropout=0.1,
		)
		return model, feed_timeloc, crop_size

	# --- All Prithvi models share the same config setup ---
	with open('configs/prithvi_300m.yaml', 'r') as f:
		prithvi_config = yaml.safe_load(f)
	prithvi_config["pretrained_cfg"]["num_frames"] = n_timesteps

	feed_timeloc = parse_param(params, "feed_timeloc", default=False, cast=str2bool)
	n_layers = parse_param(params, "n_layers", default=2, cast=int)

	# Detect crop-trained models and set img_size accordingly
	is_crop_model = "crops" in group
	if is_crop_model:
		crop_size = parse_param(params, "crop", default=48, cast=int)
		prithvi_config["pretrained_cfg"]["img_size"] = crop_size
	else:
		prithvi_config["pretrained_cfg"]["img_size"] = 336

	# --- Prithvi cathls conv3d (check before generic "conv3d") ---
	if "prithvi" in group and "cathls" in group and "conv3d" in group:
		from lib.models.prithvi_conv3d_cathls import PrithviSegConv3DCatHLS
		hls_out_channels = parse_param(params, "hlsch", default=32, cast=int)
		hls_n_layers = parse_param(params, "hlsly", default=3, cast=int)
		progressive_fusion = parse_param(params, "progfusion", default=False, cast=str2bool)
		model = PrithviSegConv3DCatHLS(
			prithvi_config["pretrained_cfg"], None,
			n_classes=4, model_size="300m",
			feed_timeloc=feed_timeloc, n_layers=n_layers,
			hls_out_channels=hls_out_channels,
			hls_n_layers=hls_n_layers,
			progressive_fusion=progressive_fusion,
		)
		return model, feed_timeloc, crop_size

	# --- Prithvi multiscale conv3d (check before generic "conv3d") ---
	if "prithvi" in group and "multiscale" in group and "conv3d" in group:
		from lib.models.prithvi_conv3d_multiscale import PrithviSegConv3DMultiScale
		feature_indices = None  # use default [5, 11, 17, 23]
		fi_str = parse_param(params, "feat", default=None, cast=str)
		if fi_str is not None:
			feature_indices = [int(x) for x in fi_str.split("-")]
		model = PrithviSegConv3DMultiScale(
			prithvi_config["pretrained_cfg"], None,
			n_classes=4, model_size="300m",
			feed_timeloc=feed_timeloc, n_layers=n_layers,
			feature_indices=feature_indices,
		)
		return model, feed_timeloc, crop_size

	# --- Prithvi conv3d (regular or crop) ---
	if "prithvi" in group and "conv3d" in group:
		from lib.models.prithvi_conv3d import PrithviSegConv3D
		model = PrithviSegConv3D(
			prithvi_config["pretrained_cfg"], None,
			n_classes=4, model_size="300m",
			feed_timeloc=feed_timeloc, n_layers=n_layers,
		)
		return model, feed_timeloc, crop_size

	raise ValueError(f"Unknown group: {group}")


def get_ndvi_data_paths(mode, data_percentage=1.0, selected_months=None):

	if selected_months is None:
		selected_months = list(range(1, 13))

	data_dir_name = f"{data_percentage}_ndvi_m{months_to_str(selected_months)}"
	checkpoint_data = f"{path_config.get_data_paths_dir()}/{data_dir_name}/"

	if os.path.exists(f'{checkpoint_data}/data_pths_{mode}.pkl'):
		with open(f'{checkpoint_data}/data_pths_{mode}.pkl', 'rb') as f:
			data_dir = pickle.load(f)

		return data_dir

	hls_path = path_config.get_data_ndvi_composites()
	lsp_path = path_config.get_data_lsp_ancillary()

	hls_tiles = [x for x in os.listdir(hls_path) if x.endswith('.tif')]
	lsp_tiles = []
	lsp_tiles.extend([x for x in os.listdir(f"{lsp_path}/A2019") if x.endswith('.tif')])
	lsp_tiles.extend([x for x in os.listdir(f"{lsp_path}/A2020") if x.endswith('.tif')])

	title_hls = ['_'.join(x.split('_')[3:5]).split(".")[0] for x in hls_tiles]
	title_hls = set(title_hls)

	title_hls_lsp = ["_".join(x.split('_')[3:5]) for x in lsp_tiles]
	title_hls_lsp = set(title_hls_lsp)

	hls_tiles_time = []
	lsp_tiles_time = []
	hls_tiles_name = []

	for year in ["2019", "2020"]:
		past_months = selected_months
		timesteps = [f"{year}-{str(x).zfill(2)}" for x in past_months]

		for hls_tile in tqdm(title_hls):
			hls_tile_location = hls_tile.split("_")[0]
			hls_tile_name = hls_tile.split("_")[1]
			temp_ordered = []

			for timestep in timesteps:
				temp_ordered.append(f"{hls_path}/HLS_composite_{timestep}_{hls_tile_location}_{hls_tile_name}_NDVI.tif")

			lsp_filename = f"HLS_PhenoCam_A{year}_{hls_tile_location}_{hls_tile_name}_LSP_Date.tif"
			assert lsp_filename in lsp_tiles, f"GT file not found: {lsp_filename}"
			temp_lsp = f"{lsp_path}/A{year}/{lsp_filename}"

			hls_tiles_time.append(temp_ordered)
			lsp_tiles_time.append(temp_lsp)
			hls_tiles_name.append(f"{year}_{hls_tile}")

	#open training file 
	with open(f"{lsp_path}/HP-LSP_train_ids.csv", 'r') as f:
		train_ids = f.readlines()[0].replace("'", "").split(",")
		train_ids = [x.strip() for x in train_ids]

	with open(f"{lsp_path}/HP-LSP_test_ids.csv", 'r') as f:
		test_ids = f.readlines()[0].replace("'", "").split(",")
		test_ids = [x.strip() for x in test_ids]


	hls_tiles_val = [
		"2019_ME-1_T19TEL",
		"2019_FL-3_T17RML",
		"2020_WI-2_T15TYL",
		"2019_AZ-5_T12SVE",
		"2020_CO-2_T13TDE",
		"2020_OR-1_T10TEQ",
		"2019_MD-1_T18SUJ",
		"2020_ND-1_T14TLS"
	]

	hls_tiles_train = [x for x in hls_tiles_name if x.split("_")[1] in train_ids]
	hls_tiles_train = [x for x in hls_tiles_train if x not in hls_tiles_val]

	# Apply data_percentage globally to all training data
	num_to_keep = int(len(hls_tiles_train) * data_percentage)
	hls_tiles_train = hls_tiles_train[:num_to_keep]
	hls_tiles_test = [x for x in hls_tiles_name if x.split("_")[1] in test_ids]
	data_dir_train = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_train]

	data_dir_val = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_val]
	data_dir_test = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_test]

	os.makedirs(checkpoint_data, exist_ok=True)
	with open(f'{checkpoint_data}/data_pths_training.pkl', 'wb') as f:
		pickle.dump(data_dir_train, f)
	
	with open(f'{checkpoint_data}/data_pths_validation.pkl', 'wb') as f:
		pickle.dump(data_dir_val, f)
	
	with open(f'{checkpoint_data}/data_pths_testing.pkl', 'wb') as f:
		pickle.dump(data_dir_test, f)

	if mode == 'training':	
		return data_dir_train
	elif mode == 'validation':
		return data_dir_val
	elif mode == "testing":
		return data_dir_test
	else: 
		raise ValueError(f"Unknown mode: {mode}. Expected 'training', 'validation', or 'testing'.")

