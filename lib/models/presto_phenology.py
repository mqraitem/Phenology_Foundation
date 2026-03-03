import sys
import os
import torch
import torch.nn as nn

# Add presto/ to path so we can import from it
_presto_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'presto')
if _presto_dir not in sys.path:
	sys.path.insert(0, os.path.abspath(_presto_dir))

from presto.presto import Presto

# S2 composite bands -> NORMED_BANDS indices
# S2 composites: [B2, B3, B4, B8A, B11, B12, mean_cs, median_cs] (we use first 6)
# NORMED_BANDS: [VV(0), VH(1), B2(2), B3(3), B4(4), B5(5), B6(6), B7(7), B8(8), B8A(9), B11(10), B12(11),
#                temperature_2m(12), total_precipitation(13), elevation(14), slope(15), NDVI(16)]
S2_TO_NORMED = [2, 3, 4, 9, 10, 11]  # B2->2, B3->3, B4->4, B8A->9, B11->10, B12->11
NUM_NORMED_BANDS = 17
DW_MISSING = 9  # DynamicWorld2020_2021.class_amount


class PrestoPhenologyModel(nn.Module):
	def __init__(self, num_classes=4, freeze_encoder=False):
		super().__init__()

		pretrained = Presto.load_pretrained()
		self.model = pretrained.construct_finetuning_model(
			num_outputs=num_classes, regression=True
		)

		if freeze_encoder:
			for param in self.model.encoder.parameters():
				param.requires_grad_(False)
			# Head stays trainable
			for param in self.model.head.parameters():
				param.requires_grad_(True)

	def _prepare_presto_inputs(self, s2_data, latlons, month=None):
		"""Convert S2 pixel data to Presto input format.

		Args:
			s2_data: (B, T, 6) raw S2 values (not normalized)
			latlons: (B, 2) lat/lon in degrees
			month: (T,) tensor of 0-indexed months, or int, or None (defaults to 0)

		Returns:
			x, dynamic_world, latlons, mask, month
		"""
		B, T, C = s2_data.shape
		device = s2_data.device

		# Build 17-band tensor, normalized by /10000 for S2
		x = torch.zeros(B, T, NUM_NORMED_BANDS, device=device, dtype=s2_data.dtype)
		x[:, :, S2_TO_NORMED] = s2_data / 10000.0

		# Mask: 1 = masked (missing), 0 = valid
		mask = torch.ones(B, T, NUM_NORMED_BANDS, device=device, dtype=s2_data.dtype)
		mask[:, :, S2_TO_NORMED] = 0.0

		# Also mask dead timesteps (all S2 bands zero)
		dead_ts = (s2_data == 0).all(dim=-1)  # (B, T)
		mask[dead_ts] = 1.0  # mask all bands for dead timesteps

		# Dynamic world: all missing (9)
		dynamic_world = torch.full((B, T), DW_MISSING, dtype=torch.long, device=device)

		# month_to_tensor expects:
		#   int -> consecutive months starting from that month
		#   1D (batch,) -> per-sample starting month, expanded to seq_len
		#   2D (batch, T) -> passed through unchanged
		# For non-consecutive months we need a 2D tensor (B, T).
		if month is None:
			month = 0
		elif isinstance(month, torch.Tensor) and month.dim() == 1 and month.shape[0] == T:
			# (T,) per-timestep months -> expand to (B, T)
			month = month.unsqueeze(0).expand(B, T).to(device)

		return x, dynamic_world, latlons, mask, month

	def forward(self, x, processing_images=True, latlons=None, month=None, chunk_size=2048):
		"""Forward pass with dual mode matching TemporalTransformer interface.

		Args:
			x: if processing_images=True: (B, 6, T, H, W) raw S2 values
			   if processing_images=False: (B, T, 6) raw S2 values
			latlons: (B, 2) or (1, 2) lat/lon in degrees
			month: (T,) 0-indexed month tensor, int, or None
			chunk_size: pixels per forward pass for memory efficiency
		"""
		device = next(self.parameters()).device

		if latlons is None:
			latlons = torch.zeros(x.shape[0], 2, device=device)

		if processing_images:
			# (B, 6, T, H, W) -> per-pixel processing
			x = x.to(device)
			latlons = latlons.to(device)
			B, C, T, H, W = x.shape

			# Reshape to pixels: (B*H*W, T, 6)
			x_pixels = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, C)

			# Expand latlons: (B, 2) -> (B*H*W, 2)
			latlons_expanded = latlons.unsqueeze(1).unsqueeze(1).expand(B, H, W, 2).reshape(B * H * W, 2)

			# Process in chunks
			outputs = []
			N = x_pixels.shape[0]
			for i in range(0, N, chunk_size):
				chunk = x_pixels[i:i + chunk_size]
				ll_chunk = latlons_expanded[i:i + chunk_size]
				px, dw, ll, mk, mo = self._prepare_presto_inputs(chunk, ll_chunk, month)
				out = self.model(x=px, dynamic_world=dw, latlons=ll, mask=mk, month=mo)
				outputs.append(out)

			out = torch.cat(outputs, dim=0)  # (B*H*W, num_classes)
			out = out.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, num_classes, H, W)
			return out

		else:
			# Pixel mode: (B, T, 6)
			x = x.to(device)
			latlons = latlons.to(device)
			px, dw, ll, mk, mo = self._prepare_presto_inputs(x, latlons, month)
			return self.model(x=px, dynamic_world=dw, latlons=ll, mask=mk, month=mo)
