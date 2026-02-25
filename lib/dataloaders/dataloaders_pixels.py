import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from lib.utils import compute_or_load_means_stds, normalize_doy
import path_config

# ===== helper functions =====


def load_raster(path, crop=None):
	import rasterio
	if os.path.exists(path):
		with rasterio.open(path) as src:
			img = src.read()
			if crop:
				img = img[:, -crop[0]:, -crop[1]:]
	else:
		img = np.zeros((6, 330, 330))
	return img

def load_raster_input(path, target_size=330):
	import rasterio
	if os.path.exists(path):
		with rasterio.open(path) as src:
			img = src.read()  # (C, H, W)
		_, h, w = img.shape
		return img.astype(np.float32)
	else:
		return np.zeros((6, target_size, target_size), dtype=np.float32)

def load_raster_output(path):
	import rasterio
	if not os.path.exists(path):
		raise FileNotFoundError(f"GT raster not found: {path}")
	with rasterio.open(path) as src:
		return src.read()

# ===== Dataset =====

class CycleDatasetPixels(Dataset):
	def __init__(self, data_dir, split, cache_path=None, data_percentage=1.0, target_size=330, regenerate=False, n_timesteps=12, file_suffix=""):
		"""
		Args:
			data_dir: list of tuples [(image_paths, gt_path, hls_tile_name), ...]
			split: "train" / "val" / "test"
			cache_path: path to npz file where pixel dataset is cached (auto-derived if None)
			data_percentage: used for naming stats file (like original code)
			target_size: padded size
			regenerate: if True, rebuild dataset even if cache exists
			n_timesteps: number of monthly time steps to use (first N months)
			file_suffix: suffix for mean/std cache file naming
		"""
		self.data_dir = data_dir
		self.split = split
		self.data_percentage = data_percentage
		self.target_size = target_size
		self.n_timesteps = n_timesteps
		self.file_suffix = file_suffix
		self.cache_path = cache_path if cache_path is not None else self._get_cache_path()

		# correct gt indices
		self.correct_indices = [2, 5, 8, 11]
		self.correct_indices = [i - 1 for i in self.correct_indices]

		# load/compute means and stds
		self.get_means_stds()

		if os.path.exists(self.cache_path) and not regenerate:
			print(f"[PixelDataset] Loading preprocessed dataset from {self.cache_path}")
			data = np.load(self.cache_path, allow_pickle=True)
			self.inputs = data['inputs']   # (N, T, C)
			self.targets = data['targets'] # (N, 4)
			self.meta = data['meta']
		else:
			print(f"[PixelDataset] Preprocessing {split} split into pixels...")
			self._build_dataset()
			print(f"[PixelDataset] Saved to {self.cache_path}")


	def _get_cache_path(self):
		if self.file_suffix.startswith("_m"):
			months_sub = self.file_suffix[1:]
			cache_dir = os.path.join(path_config.get_pixels_cache_dir(), months_sub)
		else:
			cache_dir = path_config.get_pixels_cache_dir()
		os.makedirs(cache_dir, exist_ok=True)
		return f"{cache_dir}/{self.data_percentage}_pixels{self.file_suffix}.npz"

	def get_means_stds(self):
		"""
		Compute or load the mean and standard deviation for image data.
		Uses shared utility function to avoid code duplication.
		"""
		self.means, self.stds = compute_or_load_means_stds(
			data_dir=self.data_dir,
			split=self.split,
			data_percentage=self.data_percentage,
			num_bands=6,
			load_raster_fn=load_raster,
			file_suffix=self.file_suffix,
		)

	def process_gt(self,gt):
		invalid = (gt == 32767) | (gt < 0)
		gt = normalize_doy(gt)
		gt[invalid] = -1

		return gt.astype(np.float32)

	def _build_dataset(self):
		pixel_inputs, pixel_targets, pixel_meta = [], [], []

		for idx in tqdm(range(len(self.data_dir))):
			image_paths, gt_path, hls_tile_name = self.data_dir[idx]

			# load images
			imgs = [load_raster_input(p, target_size=self.target_size)[:, np.newaxis]
					for p in image_paths]
			img = np.concatenate(imgs, axis=1)  # (C, T, H, W)

			# reshape pixels (before normalization to check for dead pixels)
			C, T, H, W = img.shape
			img_reshaped = img.reshape(C, T, H*W).transpose(2, 1, 0)  # (H*W, T, C)

			# load mask
			gt_mask = load_raster_output(gt_path)[self.correct_indices, :, :]  # (4, H, W)
			labels = gt_mask.reshape(4, H*W).transpose(1, 0)  # (H*W, 4)

			# fix masks in batch
			labels = self.process_gt(labels)  # vectorized version (fixed typo: self.self -> self)

			# identify valid pixels:
			# 1. not completely invalid labels (all -1)
			valid_labels_idx = ~(labels == -1).all(axis=1)  # (H*W,)
			
			# 2. not dead pixels (all zeros across all bands and time steps)
			non_dead_idx = ~(img_reshaped == 0).all(axis=(1, 2))  # (H*W,)
			
			# combine both conditions
			valid_idx = valid_labels_idx & non_dead_idx

			# filter pixels
			img_valid = img_reshaped[valid_idx]   # (N, T, C)
			labels_valid = labels[valid_idx]      # (N, 4)

			# normalize only valid pixels
			means1 = self.means.reshape(1, 1, -1)  # (1, 1, C) for broadcasting with (N, T, C)
			stds1 = self.stds.reshape(1, 1, -1)
			img_valid = (img_valid - means1) / (stds1 + 1e-6)

			# build meta
			h_coords, w_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
			coords = np.stack([h_coords.ravel(), w_coords.ravel()], axis=1)  # (H*W, 2)
			coords = coords[valid_idx]
			meta = [(idx, h, w, hls_tile_name) for (h, w) in coords]

			pixel_inputs.append(img_valid.astype(np.float32))
			pixel_targets.append(labels_valid.astype(np.float32))
			pixel_meta.extend(meta)

		# concatenate across images
		self.inputs = np.concatenate(pixel_inputs, axis=0)   # (N, T, C)
		self.targets = np.concatenate(pixel_targets, axis=0) # (N, 4)
		self.meta = np.array(pixel_meta, dtype=object)

		os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
		np.savez_compressed(self.cache_path,
							inputs=self.inputs,
							targets=self.targets,
							meta=self.meta)
		
	# === torch dataset API ===
	def __len__(self):
		return len(self.inputs)

	def __getitem__(self, idx):
		x = torch.from_numpy(self.inputs[idx])   # e.g., (T, C)
		y = torch.from_numpy(self.targets[idx])  # e.g., (4,)

		sample = {"image": x, "gt_mask": y}

		return sample
