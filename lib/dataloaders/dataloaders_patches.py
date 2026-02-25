import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from lib.utils import compute_or_load_means_stds, normalize_doy
import path_config

# ===== helper functions (unchanged) =====

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

def load_raster_output(path):
	import rasterio
	if not os.path.exists(path):
		raise FileNotFoundError(f"GT raster not found: {path}")
	with rasterio.open(path) as src:
		return src.read()

def load_raster_padded(path, target_size=336):
	import rasterio
	if os.path.exists(path):

		with rasterio.open(path) as src:
			img = src.read()  # shape: (C, H, W)


		_, h, w = img.shape
		pad_h = (target_size - h) if h < target_size else 0
		pad_w = (target_size - w) if w < target_size else 0

		# Pad on the bottom and right only
		padded_img = np.pad(
			img,
			pad_width=((0, 0), (0, pad_h), (0, pad_w)),
			mode='constant',
			constant_values=0  # or np.nan or other fill value
		)
		
		# Ensure consistent dtype
		padded_img = padded_img.astype(np.float32)

	else: 
		padded_img = np.zeros((6, target_size, target_size)).astype(np.float32)

	return padded_img

# ===== Patch Dataset =====

class CycleDatasetPatches(Dataset):
	"""
	Build a dataset of spatio-temporal patches.

	Output shapes:
		inputs: (N, T, C, H, W)
		targets: (N, 4, H, W)
		meta rows: (image_idx, top_h, top_w, tile_name)

	Notes:
	  - Non-overlapping patches by default (stride = patch_size).
	  - Partial edge patches are dropped (use exact tiling or prepad the rasters if you need full coverage).
	"""
	def __init__(
		self,
		data_dir,
		split,
		cache_path=None,
		data_percentage=1.0,
		target_size=336,
		regenerate=False,
		h5_path=None,
		patch_size=(32, 32),   # <---- NEW
		stride=None,           # <---- optional; defaults to patch_size (non-overlap)
		n_timesteps=12,        # <---- number of monthly time steps to use
		file_suffix="",        # <---- suffix for mean/std cache file naming
	):
		"""
		Args:
			data_dir: list of tuples [(image_paths, gt_path, hls_tile_name), ...]
			split: "train" / "val" / "test"
			cache_path: path to npz file where dataset is cached (auto-derived if None)
			data_percentage: kept for filename compatibility
			target_size: unused here for read; kept for API compatibility
			regenerate: if True, rebuild dataset even if cache exists
			h5_path: optional features file; if provided, returns per-pixel feats averaged over patch
			patch_size: (H, W) of each patch
			stride: step for sliding window; if None, stride=patch_size (non-overlapping)
		"""
		self.data_dir = data_dir
		self.split = split
		self.data_percentage = data_percentage
		self.target_size = target_size
		self.n_timesteps = n_timesteps
		self.file_suffix = file_suffix

		self.h5_path = h5_path
		self.patch_h, self.patch_w = patch_size
		self.stride_h = self.patch_h if stride is None else stride[0]
		self.stride_w = self.patch_w if stride is None else stride[1]

		self.cache_path = cache_path if cache_path is not None else self._get_cache_path()

		# correct gt indices
		self.correct_indices = [2, 5, 8, 11]
		self.correct_indices = [i - 1 for i in self.correct_indices]

		# load/compute means and stds (per-channel, same as pixel dataset)
		self.get_means_stds()

		if os.path.exists(self.cache_path) and not regenerate:
			print(f"[PatchDataset] Loading preprocessed dataset from {self.cache_path}")
			data = np.load(self.cache_path, allow_pickle=True)
			self.inputs = data['inputs']   # (N, T, C, H, W)
			self.targets = data['targets'] # (N, 4, H, W)
			self.meta = data['meta']       # (N, 4) objects
		else:
			print(f"[PatchDataset] Preprocessing {split} split into patches...")
			self._build_dataset()
			print(f"[PatchDataset] Saved to {self.cache_path}")

	# ---------- cache path ----------
	def _get_cache_path(self):
		if self.file_suffix.startswith("_m"):
			months_sub = self.file_suffix[1:]
			cache_dir = os.path.join(path_config.get_pixels_cache_dir(), months_sub)
		else:
			cache_dir = path_config.get_pixels_cache_dir()
		os.makedirs(cache_dir, exist_ok=True)
		return f"{cache_dir}/{self.data_percentage}_patch-{self.patch_h}{self.file_suffix}.npz"

	# ---------- stats ----------
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

	# ---------- dataset build (patch version) ----------
	def _edge_cover_starts(self, L: int, p: int, s: int):
		"""
		Generate non-overlapping starts with stride s, except:
		- If (L - p) % s != 0, append a final start at L - p (edge-only overlap).
		- If L < p, return [0] and let caller pad.
		"""
		if L <= p:
			return [0]
		starts = list(range(0, L - p + 1, s))
		last = L - p
		if starts[-1] != last:
			starts.append(last)  # minimal overlap for the edge
		return starts

	def _build_dataset(self):
		patch_inputs, patch_targets, patch_meta = [], [], []

		for img_idx in tqdm(range(len(self.data_dir))):
			image_paths, gt_path, hls_tile_name = self.data_dir[img_idx]

			# load and stack times at native resolution: (C, T, H, W)
			imgs = [load_raster(p)[:, np.newaxis] for p in image_paths]
			img = np.concatenate(imgs, axis=1)

			# Create mask for dead pixels (all zeros across all bands and time steps)
			alive_mask = np.any(img != 0, axis=(0, 1))  # (H, W)

			# normalize per channel, only for alive pixels
			means1 = self.means.reshape(-1, 1, 1, 1)
			stds1  = self.stds.reshape(-1, 1, 1, 1)
			img_normalized = (img - means1) / (stds1 + 1e-6)
			img = np.where(alive_mask[np.newaxis, np.newaxis, :, :], img_normalized, 0.0)

			C, T, H, W = img.shape
			ph, pw = self.patch_h, self.patch_w
			sh, sw = self.stride_h, self.stride_w

			# load gt mask at native resolution (4, H, W)
			gt_mask_full = load_raster_output(gt_path)[self.correct_indices, :, :]

			# Compute starts: non-overlap except last patch aligns to the edge if needed
			starts_h = self._edge_cover_starts(H, ph, sh)
			starts_w = self._edge_cover_starts(W, pw, sw)

			for top_h in starts_h:
				for top_w in starts_w:
					img_patch = img[:, :, top_h:top_h+ph, top_w:top_w+pw]      # (C, T, ph, pw)
					gt_patch  = gt_mask_full[:, top_h:top_h+ph, top_w:top_w+pw] # (4, ph, pw)

					gt_patch_fixed = self.process_gt(gt_patch)         # (4, ph, pw)

					# skip patches that are entirely invalid (all -1 after processing)
					if (gt_patch_fixed == -1).all():
						continue

					# (C, T, ph, pw) -> (T, C, ph, pw)
					img_patch_tc = np.transpose(img_patch, (1, 0, 2, 3)).astype(np.float32)

					patch_inputs.append(img_patch_tc)          # (T, C, ph, pw)
					patch_targets.append(gt_patch_fixed)       # (4, ph, pw)
					patch_meta.append((img_idx, int(top_h), int(top_w), hls_tile_name))

		if len(patch_inputs) == 0:
			raise RuntimeError("No valid patches were found. Check masks/paths/patch_size.")

		self.inputs  = np.stack(patch_inputs, axis=0)   # (N, T, C, H, W)
		self.targets = np.stack(patch_targets, axis=0)  # (N, 4, H, W)
		self.meta    = np.array(patch_meta, dtype=object)

		# cache
		os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
		np.savez_compressed(
			self.cache_path,
			inputs=self.inputs,
			targets=self.targets,
			meta=self.meta
		)


	# ---------- torch dataset API ----------
	def __len__(self):
		return len(self.inputs)

	def __getitem__(self, idx):
		x = torch.from_numpy(self.inputs[idx])    # (T, C, H, W)
		y = torch.from_numpy(self.targets[idx])   # (4, H, W)
		_, top_h, top_w, tile = self.meta[idx]
		top_h, top_w, tile = int(top_h), int(top_w), str(tile)

		sample = {"image": x, "gt_mask": y, "patch_origin": (top_h, top_w), "tile": tile}

		return sample