import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from lib.utils import normalize_doy
import path_config


CLOUD_SCORE_BAND = 7   # median_cs
CLOUD_THRESHOLD = 3000  # pixels with median_cs < 3000 are masked (zeroed out)


def load_raster_s2(path, target_size=330):
	"""Load S2 composite (8 bands, 990x990), downsample 3x to 330x330,
	mask cloudy pixels (median_cs < CLOUD_THRESHOLD), return first 6 bands."""
	import rasterio
	if os.path.exists(path):
		with rasterio.open(path) as src:
			img = src.read()  # (8, 990, 990)
		# 3x3 block average downsample all 8 bands
		C, H, W = img.shape
		new_h, new_w = H // 3, W // 3
		img = img[:, :new_h*3, :new_w*3].reshape(C, new_h, 3, new_w, 3).mean(axis=(2, 4))
		# Cloud mask: zero out spectral bands where median_cs < threshold
		cloud_mask = img[CLOUD_SCORE_BAND] < CLOUD_THRESHOLD  # (330, 330)
		img[:6, cloud_mask] = 0
		return img[:6].astype(np.float32)
	else:
		return np.zeros((6, target_size, target_size), dtype=np.float32)


def load_raster_output(path):
	import rasterio
	if not os.path.exists(path):
		raise FileNotFoundError(f"GT raster not found: {path}")
	with rasterio.open(path) as src:
		return src.read()


class CycleDatasetPixelsS2(Dataset):
	def __init__(self, data_dir, split, all_locations, cache_path=None,
				 data_percentage=1.0, target_size=330, regenerate=False,
				 n_timesteps=12, file_suffix=""):
		"""
		Pixel-level dataset for S2 composites (no mean/std normalization -- Presto normalizes internally).

		Args:
			data_dir: list of tuples [(image_paths, gt_path, tile_name), ...]
			split: "train" / "val" / "test"
			all_locations: dict mapping tile_name -> [lat, lon]
			cache_path: path to npz cache file (auto-derived if None)
			data_percentage: used for cache naming
			target_size: spatial size after downsampling
			regenerate: if True, rebuild even if cache exists
			n_timesteps: number of monthly timesteps
			file_suffix: suffix for cache naming
		"""
		self.data_dir = data_dir
		self.split = split
		self.data_percentage = data_percentage
		self.target_size = target_size
		self.n_timesteps = n_timesteps
		self.file_suffix = file_suffix
		self.all_locations = all_locations
		self.cache_path = cache_path if cache_path is not None else self._get_cache_path()

		self.correct_indices = [2, 5, 8, 11]
		self.correct_indices = [i - 1 for i in self.correct_indices]

		if os.path.exists(self.cache_path) and not regenerate:
			print(f"[PixelDatasetS2] Loading preprocessed dataset from {self.cache_path}")
			data = np.load(self.cache_path, allow_pickle=True)
			self.inputs = data['inputs']     # (N, T, 6)
			self.targets = data['targets']   # (N, 4)
			self.latlons = data['latlons']   # (N, 2)
		else:
			print(f"[PixelDatasetS2] Preprocessing {split} split into pixels...")
			self._build_dataset()
			print(f"[PixelDatasetS2] Saved to {self.cache_path}")

	def _get_cache_path(self):
		if self.file_suffix.startswith("_m"):
			months_sub = self.file_suffix[1:]
			cache_dir = os.path.join(path_config.get_pixels_cache_dir(), months_sub)
		else:
			cache_dir = path_config.get_pixels_cache_dir()
		os.makedirs(cache_dir, exist_ok=True)
		return f"{cache_dir}/{self.data_percentage}_pixels_s2{self.file_suffix}.npz"

	def process_gt(self, gt):
		invalid = (gt == 32767) | (gt < 0)
		gt = normalize_doy(gt)
		gt[invalid] = -1
		return gt.astype(np.float32)

	def _build_dataset(self):
		pixel_inputs, pixel_targets, pixel_latlons = [], [], []

		for idx in tqdm(range(len(self.data_dir))):
			image_paths, gt_path, tile_name = self.data_dir[idx]

			# load images
			imgs = [load_raster_s2(p, target_size=self.target_size)[:, np.newaxis]
					for p in image_paths]
			img = np.concatenate(imgs, axis=1)  # (6, T, H, W)

			C, T, H, W = img.shape
			img_reshaped = img.reshape(C, T, H*W).transpose(2, 1, 0)  # (H*W, T, C)

			# load GT
			gt_mask = load_raster_output(gt_path)[self.correct_indices, :, :]  # (4, H, W)
			labels = gt_mask.reshape(4, H*W).transpose(1, 0)  # (H*W, 4)
			labels = self.process_gt(labels)

			# valid pixels: not all-invalid GT and not dead (all zeros)
			valid_labels_idx = ~(labels == -1).all(axis=1)
			non_dead_idx = ~(img_reshaped == 0).all(axis=(1, 2))
			valid_idx = valid_labels_idx & non_dead_idx

			img_valid = img_reshaped[valid_idx]    # (N, T, 6)
			labels_valid = labels[valid_idx]       # (N, 4)

			# lat/lon for all valid pixels (tile-center)
			loc = self.all_locations.get(tile_name, [0.0, 0.0])
			n_valid = img_valid.shape[0]
			latlons_tile = np.tile(np.array(loc, dtype=np.float32), (n_valid, 1))  # (N, 2)

			pixel_inputs.append(img_valid.astype(np.float32))
			pixel_targets.append(labels_valid.astype(np.float32))
			pixel_latlons.append(latlons_tile)

		self.inputs = np.concatenate(pixel_inputs, axis=0)     # (N, T, 6)
		self.targets = np.concatenate(pixel_targets, axis=0)   # (N, 4)
		self.latlons = np.concatenate(pixel_latlons, axis=0)   # (N, 2)

		os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
		np.savez_compressed(self.cache_path,
							inputs=self.inputs,
							targets=self.targets,
							latlons=self.latlons)

	def __len__(self):
		return len(self.inputs)

	def __getitem__(self, idx):
		x = torch.from_numpy(self.inputs[idx])     # (T, 6)
		y = torch.from_numpy(self.targets[idx])     # (4,)
		ll = torch.from_numpy(self.latlons[idx])    # (2,)

		return {"image": x, "gt_mask": y, "latlons": ll}
