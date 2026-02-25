import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.prithvi_conv3d import (
	PrithviBackbone,
	PrithviReshape3D,
	Conv3DTemporalSpatialBlock,
	ChannelLayerNorm3D,
	TemporalFusionHead,
)


class HLSBranch3D(nn.Module):
	"""Per-pixel temporal encoder at full spatial resolution.

	Uses (3,1,1) kernels — temporal-only convolutions — to capture
	fine-grained per-pixel spectral-temporal patterns that Prithvi's
	16x16 patch embedding loses.

	Args:
		in_channels:  number of input bands (default 6 for HLS)
		out_channels: output feature channels (default 32)
		n_layers:     number of Conv3d blocks (default 3)
	"""

	def __init__(self, in_channels=6, out_channels=32, n_layers=3):
		super().__init__()
		layers = []
		ch = in_channels
		for i in range(n_layers):
			next_ch = out_channels
			layers.extend([
				nn.Conv3d(ch, next_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
				ChannelLayerNorm3D(next_ch),
				nn.GELU(),
			])
			ch = next_ch
		self.net = nn.Sequential(*layers)

	def forward(self, x):
		"""x: (B, C, T, H, W) -> (B, out_channels, T, H, W)"""
		return self.net(x)


class FiLMModulation(nn.Module):
	"""Feature-wise Linear Modulation: HLS features modulate Prithvi features.

	gamma, beta are derived from the HLS branch via 1x1x1 conv.
	Initialized to identity (gamma=1, beta=0) so the model starts
	equivalent to the baseline.

	Args:
		hls_channels:     channels from HLS branch
		prithvi_channels: channels from Prithvi upscaler output
	"""

	def __init__(self, hls_channels, prithvi_channels):
		super().__init__()
		self.gamma_proj = nn.Conv3d(hls_channels, prithvi_channels, kernel_size=1)
		self.beta_proj = nn.Conv3d(hls_channels, prithvi_channels, kernel_size=1)

		# Initialize to identity: gamma=1, beta=0
		nn.init.zeros_(self.gamma_proj.weight)
		nn.init.ones_(self.gamma_proj.bias)
		nn.init.zeros_(self.beta_proj.weight)
		nn.init.zeros_(self.beta_proj.bias)

	def forward(self, prithvi_feat, hls_feat):
		"""
		prithvi_feat: (B, prithvi_channels, T, H, W)
		hls_feat:     (B, hls_channels, T, H, W)
		returns:      (B, prithvi_channels, T, H, W)
		"""
		gamma = self.gamma_proj(hls_feat)  # (B, prithvi_channels, T, H, W)
		beta = self.beta_proj(hls_feat)
		return prithvi_feat * gamma + beta


class Upscaler3DCatHLS(nn.Module):
	"""Conv3D upscaler with parallel HLS branch and FiLM fusion.

	Prithvi path: 4x ConvTranspose spatial upscaling blocks
	  (B, embed_dim, T, 3, 3) -> (B, embed_dim//8, T, 48, 48)

	HLS path: per-pixel temporal encoder at full resolution
	  (B, 6, T, 48, 48) -> (B, hls_out_channels, T, 48, 48)

	Fusion modes:
	  - progressive=False: FiLM at the end only (after all upscaling)
	  - progressive=True:  FiLM at each upscaler stage, with HLS features
	    pooled to match spatial resolution. Guides spatial reconstruction
	    at every scale.

	Args:
		embed_dim:          Prithvi embedding dimension (1024 for 300m)
		n_classes:          number of output channels (default 4)
		n_layers:           layers in TemporalFusionHead (default 2)
		dropout:            use dropout in upscale blocks (default True)
		in_channels:        HLS input bands (default 6)
		hls_out_channels:   HLS branch output channels (default 32)
		hls_n_layers:       number of Conv3d blocks in HLS branch (default 3)
		num_frames:         number of temporal frames
		progressive_fusion: FiLM at each upscaler stage (default False)
	"""

	def __init__(self, embed_dim, n_classes=4, n_layers=2, dropout=True,
				 in_channels=6, hls_out_channels=32, hls_n_layers=3,
				 num_frames=4, progressive_fusion=False):
		super().__init__()
		self.progressive_fusion = progressive_fusion

		# Prithvi upscaling blocks (ModuleList for interleaved FiLM)
		self.block1 = Conv3DTemporalSpatialBlock(embed_dim, embed_dim // 2, dropout=dropout)
		self.block2 = Conv3DTemporalSpatialBlock(embed_dim // 2, embed_dim // 4, dropout=dropout)
		self.block3 = Conv3DTemporalSpatialBlock(embed_dim // 4, embed_dim // 8, dropout=dropout)
		self.block4 = Conv3DTemporalSpatialBlock(embed_dim // 8, embed_dim // 8, dropout=dropout)

		prithvi_final_ch = embed_dim // 8

		# HLS branch
		self.hls_branch = HLSBranch3D(in_channels, hls_out_channels, hls_n_layers)

		if progressive_fusion:
			# FiLM at each upscaler stage
			self.film1 = FiLMModulation(hls_out_channels, embed_dim // 2)
			self.film2 = FiLMModulation(hls_out_channels, embed_dim // 4)
			self.film3 = FiLMModulation(hls_out_channels, embed_dim // 8)
			self.film4 = FiLMModulation(hls_out_channels, embed_dim // 8)
		else:
			# FiLM only at the end
			self.film = FiLMModulation(hls_out_channels, prithvi_final_ch)

		self.head = TemporalFusionHead(prithvi_final_ch, n_classes, n_layers, num_frames=num_frames)

	def forward(self, prithvi_latent, hls_input):
		"""
		prithvi_latent: (B, embed_dim, T, H_patch, W_patch) from PrithviReshape3D
		hls_input:      (B, C, T, H, W) same normalized input at full resolution
		"""
		hls_feat = self.hls_branch(hls_input)  # (B, hls_out_channels, T, H, W)

		if self.progressive_fusion:
			x = self.block1(prithvi_latent)
			x = self.film1(x, F.adaptive_avg_pool3d(hls_feat, x.shape[2:]))

			x = self.block2(x)
			x = self.film2(x, F.adaptive_avg_pool3d(hls_feat, x.shape[2:]))

			x = self.block3(x)
			x = self.film3(x, F.adaptive_avg_pool3d(hls_feat, x.shape[2:]))

			x = self.block4(x)
			x = self.film4(x, hls_feat)
		else:
			x = self.block1(prithvi_latent)
			x = self.block2(x)
			x = self.block3(x)
			x = self.block4(x)
			x = self.film(x, hls_feat)

		return self.head(x)


class PrithviSegConv3DCatHLS(nn.Module):
	"""Prithvi + HLS skip-connection model for phenology prediction.

	Both branches receive the same normalized input. The Prithvi branch
	processes it through the ViT backbone and 3D upscaler, while the HLS
	branch extracts per-pixel temporal features at full resolution.
	FiLM modulation fuses HLS features into the Prithvi path.

	Args:
		prithvi_params:     dict of Prithvi config
		prithvi_ckpt_path:  path to pretrained weights (or None)
		n_classes:          number of output classes (default 4)
		model_size:         "300m" (default)
		feed_timeloc:       feed time/location coords to Prithvi
		n_layers:           layers in TemporalFusionHead
		hls_out_channels:   HLS branch output channels (default 32)
		hls_n_layers:       HLS branch depth (default 3)
		progressive_fusion: FiLM at each upscaler stage (default False)
	"""

	def __init__(self,
				 prithvi_params: dict,
				 prithvi_ckpt_path: str = None,
				 n_classes: int = 4,
				 model_size: str = "300m",
				 feed_timeloc: bool = False,
				 n_layers: int = 2,
				 hls_out_channels: int = 32,
				 hls_n_layers: int = 3,
				 progressive_fusion: bool = False):
		super().__init__()

		if feed_timeloc:
			prithvi_params["coords_encoding"] = ["time", "location"]

		self.backbone = PrithviBackbone(prithvi_params, prithvi_ckpt_path)

		self.reshaper = PrithviReshape3D(
			prithvi_params["patch_size"],
			prithvi_params["img_size"],
			prithvi_params["num_frames"],
		)

		embed_dim = prithvi_params["embed_dim"]
		in_channels = prithvi_params.get("in_chans", 6)
		num_frames = prithvi_params["num_frames"]

		if model_size == "300m":
			self.head = Upscaler3DCatHLS(
				embed_dim, n_classes=n_classes, n_layers=n_layers,
				in_channels=in_channels,
				hls_out_channels=hls_out_channels,
				hls_n_layers=hls_n_layers,
				num_frames=num_frames,
				progressive_fusion=progressive_fusion,
			)
		else:
			raise ValueError(f"model_size {model_size} not supported")

	def forward(self, x):
		if isinstance(x, dict):
			chip = x.get("chip")
			hls_input = chip.cuda()
			x = {k: v.cuda() for k, v in x.items()}
		else:
			hls_input = x.cuda()
			x = x.cuda()

		latent = self.backbone(x)      # (B, 1+T*H*W, D)
		latent = self.reshaper(latent)  # (B, D, T, H_patch, W_patch)
		out = self.head(latent, hls_input)  # (B, n_classes, H, W)
		return out
