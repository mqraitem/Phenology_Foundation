import torch
import torch.nn as nn

from lib.models.prithvi_conv3d import PrithviReshape3D, Upscaler3D
from lib.models.prithvi_mae import PrithviMAE


# Default feature indices for 24-block (300M) model: evenly spaced
DEFAULT_FEATURE_INDICES_300M = [5, 11, 17, 23]


class PrithviBackboneMultiLevel(nn.Module):
	"""Prithvi backbone that returns features from multiple transformer blocks."""

	def __init__(self, prithvi_params: dict, prithvi_ckpt_path: str = None,
				 feature_indices: list[int] = None):
		super().__init__()
		self.prithvi_params = prithvi_params
		self.feature_indices = feature_indices or DEFAULT_FEATURE_INDICES_300M

		self.model = PrithviMAE(**prithvi_params)

		if prithvi_ckpt_path is not None:
			checkpoint = torch.load(prithvi_ckpt_path, weights_only=False)

			if "encoder.pos_embed" not in checkpoint.keys():
				key = "model" if "model" in checkpoint.keys() else "state_dict"
				keys = list(checkpoint[key].keys())
				checkpoint = checkpoint[key]
			else:
				keys = list(checkpoint.keys())

			for k in keys:
				if ((prithvi_params["encoder_only"]) and ("decoder" in k)) or "pos_embed" in k:
					del checkpoint[k]
				elif "prithvi" in k:
					print(f"Warning: renaming prithvi layer {k}")
					new_k = k.replace("prithvi.", "")
					checkpoint[new_k] = checkpoint[k]
				elif k in self.model.state_dict() and checkpoint[k].shape != self.model.state_dict()[k].shape:
					print(f"Warning: size mismatch for layer {k}, deleting: "
						  f"{checkpoint[k].shape} != {self.model.state_dict()[k].shape}")
					del checkpoint[k]

			_ = self.model.load_state_dict(checkpoint, strict=False)

	def forward(self, data):
		if isinstance(data, dict):
			chip = data.get("chip")
			temporal = data.get("temporal_coords")
			location = data.get("location_coords")
		else:
			chip = data
			temporal = None
			location = None

		return self.model.forward_features_multilevel(
			chip, temporal, location, self.feature_indices
		)


class MultiScaleFusion3D(nn.Module):
	"""Fuse multi-level ViT features into a single 3D tensor.

	Each level (B, 1+T*H*W, embed_dim) is independently normalized,
	projected to embed_dim // n_levels, and reshaped to (B, proj_dim, T, H, W).
	All levels are concatenated along channels -> (B, embed_dim, T, H, W).
	"""

	def __init__(self, embed_dim, n_levels, patch_size, img_size, num_frames):
		super().__init__()

		proj_dim = embed_dim // n_levels
		self.proj_dim = proj_dim
		self.n_levels = n_levels

		self.reshaper = PrithviReshape3D(patch_size, img_size, num_frames)

		# Per-level normalization and projection
		self.level_norms = nn.ModuleList([
			nn.LayerNorm(embed_dim) for _ in range(n_levels)
		])
		self.level_projs = nn.ModuleList([
			nn.Linear(embed_dim, proj_dim) for _ in range(n_levels)
		])

	def forward(self, features):
		"""
		Args:
			features: list of n_levels tensors, each (B, 1+T*H*W, embed_dim)
		Returns:
			(B, n_levels * proj_dim, T, H, W) tensor
		"""
		fused = []
		for feat, norm, proj in zip(features, self.level_norms, self.level_projs):
			feat = norm(feat)              # (B, 1+T*H*W, embed_dim)
			feat = proj(feat)              # (B, 1+T*H*W, proj_dim)
			feat = self.reshaper(feat)     # (B, proj_dim, T, H, W)
			fused.append(feat)

		return torch.cat(fused, dim=1)     # (B, embed_dim, T, H, W)


class PrithviSegConv3DMultiScale(nn.Module):
	"""Prithvi with multi-scale feature fusion + Conv3D upscaler.

	Extracts features from 4 intermediate ViT blocks, projects each to
	embed_dim // 4, concatenates back to embed_dim, and feeds through
	the same Conv3D upscaler head used in PrithviSegConv3D.

	Default feature indices for 300M (24 blocks): [5, 11, 17, 23]
	"""

	def __init__(self,
				 prithvi_params: dict,
				 prithvi_ckpt_path: str = None,
				 n_classes: int = 4,
				 model_size: str = "300m",
				 feed_timeloc: bool = False,
				 n_layers: int = 2,
				 feature_indices: list[int] = None):
		super().__init__()

		if feature_indices is None:
			depth = prithvi_params.get("depth", 24)
			n_features = 4
			step = depth // n_features
			feature_indices = [step * (i + 1) - 1 for i in range(n_features)]

		self.feature_indices = feature_indices

		if feed_timeloc:
			prithvi_params["coords_encoding"] = ["time", "location"]

		self.backbone = PrithviBackboneMultiLevel(
			prithvi_params, prithvi_ckpt_path, feature_indices
		)

		embed_dim = prithvi_params["embed_dim"]
		n_levels = len(feature_indices)

		self.fusion = MultiScaleFusion3D(
			embed_dim, n_levels,
			prithvi_params["patch_size"],
			prithvi_params["img_size"],
			prithvi_params["num_frames"],
		)

		# After fusion: n_levels * (embed_dim // n_levels) = embed_dim
		num_frames = prithvi_params["num_frames"]

		if model_size == "300m":
			self.head = Upscaler3D(embed_dim, n_classes=n_classes, n_layers=n_layers, num_frames=num_frames)
		else:
			raise ValueError(f"model_size {model_size} not supported")

	def forward(self, x):
		if isinstance(x, dict):
			x = {k: v.cuda() for k, v in x.items()}
		else:
			x = x.cuda()

		features = self.backbone(x)       # list of (B, 1+T*H*W, D)
		x = self.fusion(features)          # (B, embed_dim, T, H, W)
		x = self.head(x)                   # (B, n_classes, H_out, W_out)
		return x
