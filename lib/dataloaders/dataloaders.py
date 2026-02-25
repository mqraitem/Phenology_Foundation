import rasterio
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import geopandas as gpd
import path_config
from lib.utils import compute_or_load_means_stds, normalize_doy
from datetime import datetime


def load_raster(path,crop=None):

	if os.path.exists(path):
		with rasterio.open(path) as src:
			img = src.read()
			if crop:
				img = img[:, -crop[0]:, -crop[1]:]

	else:
		img = np.zeros((6, 330, 330))

	return img

def load_raster_input(path, target_size=336):

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

def load_raster_output(path):
		if not os.path.exists(path):
			raise FileNotFoundError(f"GT raster not found: {path}")
		with rasterio.open(path) as src:
			img = src.read()

		return img


class CycleDataset(Dataset):
	def __init__(self,path,split, data_percentage=1.0, means=None, stds=None, feed_timeloc=False, n_timesteps=12, file_suffix=""):


		self.data_dir=path
		self.split=split

		self.total_below_0 = 0
		self.total_above_365 = 0
		self.total_nan = 0
		self.total = 0
		self.data_percentage = data_percentage
		self.n_timesteps = n_timesteps
		self.file_suffix = file_suffix
		# Region filtering removed - hardcoded to "all"

		self.correct_indices = [2, 5, 8, 11]
		self.correct_indices = [i - 1 for i in self.correct_indices]  # Convert to zero-based index

		if means is None or stds is None:
			self.get_means_stds()
		else:
			self.means = np.array(means)
			self.stds = np.array(stds)

			print("Using precomputed means and stds")

		self.assign_location_time_info()
		self.feed_timeloc = feed_timeloc

		# self.test_theory()

	def set_feed_timeloc(self, feed_timeloc): 
		self.feed_timeloc = feed_timeloc

	def assign_location_time_info(self): 
		geo_path = path_config.get_data_geojson()
		geo_gdf = gpd.read_file(geo_path)
		geo_gdf = geo_gdf.rename(columns={"Site_ID": "SiteID"})
		geo_gdf["HLStile"] = "T" + geo_gdf["name"]
		geo_gdf = geo_gdf.set_crs("EPSG:4326")
		geo_gdf["centroid"] = geo_gdf.geometry.representative_point()

		self.all_locations = {} 
		self.all_times = {}

		for input in self.data_dir: 
			full_id = input[2]
			hls_tile = full_id.split("_")[-1]
			site_id = full_id.split("_")[-2]
			centroid = geo_gdf[(geo_gdf["HLStile"] == hls_tile) & (geo_gdf["SiteID"] == site_id)]["centroid"].iloc[0]
			location_coords = [centroid.y, centroid.x]  # [lat, lon]

			self.all_locations[full_id] =  location_coords

			all_input_images_times = [x.split("/")[-1].split("_")[2].split("-") for x in input[0]]
			temp_coords = [[int(x[0]), int(x[1])] for x in all_input_images_times]
			temp_coords = [[x[0], datetime(x[0], x[1], 15).timetuple().tm_yday] for x in temp_coords]

			self.all_times[full_id] = temp_coords


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

	def __len__(self):
		return len(self.data_dir)

	def normalize_image(self, image, means, stds):
		"""
		Normalize a (bands, time, H, W) image using per-band means/stds (across all time steps).
		`image` is (bands, time, H, W)
		Pixels where all bands/time steps are zero are left as zero (not normalized).
		Returns a torch tensor with shape (bands, time, H, W)
		"""
		number_of_channels = image.shape[0]  # bands
		number_of_time_steps = image.shape[1]
		bands, time, H, W = image.shape
		vh, vw = (330, 330)  # e.g. 330, 330

		# Reshape for broadcasting
		means1 = means.reshape(bands, 1, 1, 1)
		stds1 = stds.reshape(bands, 1, 1, 1)

		# Initialize output with zeros (preserve padding)
		normalized = np.zeros_like(image, dtype=np.float32)

		# Create mask for pixels where all bands/time steps are zero
		# Shape: (H, W) - True where pixel should be normalized
		valid_region = image[:, :, :vh, :vw]
		non_zero_mask = np.any(valid_region != 0, axis=(0, 1))  # (vh, vw)

		# Normalize only valid region where pixels are non-zero
		normalized_valid = (
			(valid_region.astype(np.float32) - means1) / (stds1 + 1e-6)
		)
		
		# Apply mask: keep normalized values only where pixels are non-zero
		# Broadcast mask from (vh, vw) to (bands, time, vh, vw)
		normalized[:, :, :vh, :vw] = np.where(
			non_zero_mask[np.newaxis, np.newaxis, :, :],
			normalized_valid,
			0.0
		)

		# Convert to torch tensor with batch dimension
		normalized_tensor = torch.from_numpy(
			normalized.reshape(number_of_channels, number_of_time_steps, *image.shape[-2:])
		).to(torch.float32)

		return normalized_tensor

	def process_gt(self,gt):
		invalid = (gt == 32767) | (gt < 0)
		gt = normalize_doy(gt)
		gt[invalid] = -1

		return gt.astype(np.float32)

	def __getitem__(self,idx):

		image_path=self.data_dir[idx][0]
		output_path=self.data_dir[idx][1]
		hls_tile_name = self.data_dir[idx][2]

		images = []
		for path in image_path:
			images.append(load_raster_input(path)[:, np.newaxis])

		gt_mask=load_raster_output(output_path)
		gt_mask = gt_mask[self.correct_indices, :, :]

		image = np.concatenate(images, axis=1)
		final_image=self.normalize_image(image, self.means, self.stds)
		gt_mask = self.process_gt(gt_mask)

		temporal_coords = self.all_times[hls_tile_name]
		location_coords = self.all_locations[hls_tile_name]


		if self.feed_timeloc: 
			to_return = {
				"image": {
					"chip": final_image, 
					"temporal_coords": torch.tensor(temporal_coords, dtype=torch.float32),
					"location_coords": torch.tensor(location_coords, dtype=torch.float32),
				},
				"image_unprocessed": image,
				"gt_mask": gt_mask,
				"hls_tile_name": hls_tile_name,

			}
		else: 
			to_return = {
				"image": final_image,
				"image_unprocessed": image,
				"gt_mask": gt_mask,
				"hls_tile_name": hls_tile_name,
			}

		return to_return


# def test_theory(self): 

# 	total_all_zeros_pixels = 0
# 	total_nans_imgs = 0
# 	total_nans_gts = 0 
# 	total_negative_input = 0
# 	total_all_high_pixels = 0
# 	for i in tqdm(range(len(self.data_dir))):

# 		image_path = self.data_dir[i][0]
# 		gt_path = self.data_dir[i][1]
			
# 		images = []
# 		for path in image_path:
# 			images.append(load_raster_input(path)[:, np.newaxis])

# 		img = np.concatenate(images, axis=1)  # shape: (num_bands, time_steps, H, W)
# 		gt_mask = load_raster_output(gt_path)
# 		img = img[:, :, :330, :330]

# 		total_all_zeros_pixels += np.sum(np.all(img == 0, axis=(0, 1)))
# 		total_nans_imgs += np.sum(np.isnan(img)) 
# 		total_nans_gts += np.sum(gt_mask==32767) 
# 		total_negative_input += np.sum(img == -9999)
# 		total_all_high_pixels += np.sum(np.all(img == -9999, axis=(0, 1)))

# 		img = self.normalize_image(img, self.means, self.stds)

# 	print(total_all_zeros_pixels)
# 	print(total_nans_imgs)
# 	print(total_nans_gts)
# 	print(total_negative_input)
# 	print(total_all_high_pixels)

# 	quit()

