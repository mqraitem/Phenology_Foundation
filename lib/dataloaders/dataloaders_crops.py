import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from lib.utils import compute_or_load_means_stds, normalize_doy
import path_config


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
			img = src.read()
		return img[:, :target_size, :target_size].astype(np.float32)
	else:
		return np.zeros((6, target_size, target_size), dtype=np.float32)


def load_raster_output(path):
	import rasterio
	if not os.path.exists(path):
		raise FileNotFoundError(f"GT raster not found: {path}")
	with rasterio.open(path) as src:
		return src.read()


class CycleDatasetCrops(Dataset):
	"""Random crop dataset: samples random spatial crops from random tiles.

	Pre-loads all tiles into memory (normalized).
	Each __getitem__ picks a random tile and a random spatial location,
	returning a crop_size x crop_size coherent region.

	Unlike mosaics, each crop is spatially coherent — no fake boundaries.
	The model can use full spatial attention on real data.

	Args:
		data_dir:        list of (image_paths, gt_path, tile_name) tuples
		split:           "training" / "validation" / "testing"
		crop_size:       spatial size of each crop (default 48)
		data_percentage: for stats file naming
		n_timesteps:     number of monthly timesteps
		file_suffix:     suffix for mean/std cache
		epoch_length:    number of crops per epoch
	"""

	def __init__(self, data_dir, split, crop_size=48,
				 data_percentage=1.0, n_timesteps=12, file_suffix="",
				 epoch_length=5000):

		self.data_dir = data_dir
		self.split = split
		self.crop_size = crop_size
		self.data_percentage = data_percentage
		self.n_timesteps = n_timesteps
		self.file_suffix = file_suffix
		self.epoch_length = epoch_length
		self.tile_size = 330

		self.correct_indices = [2, 5, 8, 11]
		self.correct_indices = [i - 1 for i in self.correct_indices]

		self.means, self.stds = compute_or_load_means_stds(
			data_dir=self.data_dir,
			split=self.split,
			data_percentage=self.data_percentage,
			num_bands=6,
			load_raster_fn=load_raster,
			file_suffix=self.file_suffix,
		)

		self._load_or_build_tile_bank()

	def _get_cache_path(self):
		if self.file_suffix.startswith("_m"):
			months_sub = self.file_suffix[1:]
			cache_dir = os.path.join(path_config.get_pixels_cache_dir(), months_sub)
		else:
			cache_dir = path_config.get_pixels_cache_dir()
		os.makedirs(cache_dir, exist_ok=True)
		return f"{cache_dir}/{self.data_percentage}_crops{self.file_suffix}.npz"

	def _load_or_build_tile_bank(self):
		cache_path = self._get_cache_path()

		if os.path.exists(cache_path):
			print(f"[Crops] Loading tile bank from {cache_path}")
			data = np.load(cache_path)
			self.all_images = data["images"]
			self.all_gts = data["gts"]
		else:
			print(f"[Crops] Building tile bank (will cache to {cache_path})")
			self._build_tile_bank()
			np.savez(cache_path,
					 images=self.all_images,
					 gts=self.all_gts)
			print(f"[Crops] Saved tile bank to {cache_path}")

		print(f"[Crops] Tile bank: {len(self.all_images)} tiles of size "
			  f"{self.tile_size}x{self.tile_size}")
		print(f"[Crops] Crop size: {self.crop_size}x{self.crop_size}")
		print(f"[Crops] Memory: {self.all_images.nbytes / 1e6:.0f} MB images + "
			  f"{self.all_gts.nbytes / 1e6:.0f} MB GT")

	def _build_tile_bank(self):
		means = self.means.reshape(6, 1, 1, 1)
		stds = self.stds.reshape(6, 1, 1, 1)

		all_images = []
		all_gts = []

		for idx in tqdm(range(len(self.data_dir)), desc="Loading tiles"):
			image_paths, gt_path, tile_name = self.data_dir[idx]

			imgs = [load_raster_input(p, target_size=self.tile_size)[:, np.newaxis]
					for p in image_paths]
			img = np.concatenate(imgs, axis=1)  # (6, T, 330, 330)

			gt = load_raster_output(gt_path)[
				self.correct_indices, :self.tile_size, :self.tile_size]
			gt = self._process_gt(gt)

			non_zero_mask = np.any(img != 0, axis=(0, 1))  # (H, W)
			img_norm = (img.astype(np.float32) - means) / (stds + 1e-6)
			img_norm = np.where(
				non_zero_mask[np.newaxis, np.newaxis, :, :], img_norm, 0.0
			).astype(np.float32)

			all_images.append(img_norm)
			all_gts.append(gt)

		self.all_images = np.array(all_images)  # (N_tiles, 6, T, 330, 330)
		self.all_gts = np.array(all_gts)         # (N_tiles, 4, 330, 330)

	def _process_gt(self, gt):
		invalid = (gt == 32767) | (gt < 0)
		gt = normalize_doy(gt)
		gt[invalid] = -1
		return gt.astype(np.float32)

	def __len__(self):
		return self.epoch_length

	def __getitem__(self, idx):
		cs = self.crop_size

		tile_idx = np.random.randint(len(self.all_images))
		img = self.all_images[tile_idx]
		gt = self.all_gts[tile_idx]

		max_r = self.tile_size - cs
		max_c = self.tile_size - cs
		r = np.random.randint(0, max_r + 1)
		c = np.random.randint(0, max_c + 1)

		img_crop = img[:, :, r:r+cs, c:c+cs].copy()
		gt_crop = gt[:, r:r+cs, c:c+cs].copy()

		# Resample if dead crop (rare with random crops)
		attempts = 0
		while not np.any(img_crop != 0) and attempts < 5:
			tile_idx = np.random.randint(len(self.all_images))
			r = np.random.randint(0, max_r + 1)
			c = np.random.randint(0, max_c + 1)
			img_crop = self.all_images[tile_idx][:, :, r:r+cs, c:c+cs].copy()
			gt_crop = self.all_gts[tile_idx][:, r:r+cs, c:c+cs].copy()
			attempts += 1

		return {
			"image": torch.from_numpy(np.ascontiguousarray(img_crop)),
			"gt_mask": torch.from_numpy(np.ascontiguousarray(gt_crop)),
			"hls_tile_name": "crop",
		}
