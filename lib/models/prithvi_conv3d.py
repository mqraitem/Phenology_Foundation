import torch
import torch.nn as nn


class ChannelLayerNorm3D(nn.Module):
    """LayerNorm over channels for 5D tensors (B, C, T, H, W).

    Normalizes over C at each (t, h, w) position independently.
    """

    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        return x


class PrithviBackbone(nn.Module):
    def __init__(self,
                 prithvi_params: dict,
                 prithvi_ckpt_path: str = None):
        super().__init__()
        self.prithvi_ckpt_path = prithvi_ckpt_path
        self.prithvi_params = prithvi_params

        from lib.models.prithvi_mae import PrithviMAE
        self.model = PrithviMAE(**self.prithvi_params)
        if self.prithvi_ckpt_path is not None:
            checkpoint = torch.load(self.prithvi_ckpt_path, weights_only=False)

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
                    print(f"Warning: size mismatch for layer {k}, deleting: {checkpoint[k].shape} != {self.model.state_dict()[k].shape}")
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

        if self.prithvi_params["encoder_only"]:
            return self.model.forward_features(chip, temporal, location)
        else:
            latent, mask, ids_restore = self.model.encoder(chip, temporal, location, 0.0)
            return self.model.decoder(latent,
                                ids_restore,
                                temporal,
                                location,
                                input_size=(self.prithvi_params["num_frames"], self.prithvi_params["img_size"], self.prithvi_params["img_size"]))


class PrithviReshape3D(nn.Module):
    """Reshape backbone output to (B, embed_dim, T, H_patches, W_patches)
    keeping the temporal dimension separate from channels."""

    def __init__(self, patch_size, input_size, num_frames):
        super().__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        self.num_frames = num_frames
        self.spatial_size = int(self.input_size / self.patch_size[-1])

    def forward(self, latent):
        # latent: (B, 1 + T*H*W, embed_dim) from ViT
        latent = latent[:, 1:, :]  # remove CLS token: (B, T*H*W, D)
        B, N, D = latent.shape
        H = W = self.spatial_size
        T = self.num_frames

        # Reshape to (B, T, H, W, D) then permute to (B, D, T, H, W)
        latent = latent.reshape(B, T, H, W, D)
        latent = latent.permute(0, 4, 1, 2, 3)  # (B, D, T, H, W)
        return latent


class Conv3DTemporalSpatialBlock(nn.Module):
    """Upscale spatial dims by 2x and reduce channels."""

    def __init__(self, in_ch, out_ch, dropout=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            ChannelLayerNorm3D(out_ch),
            nn.GELU(),
            nn.Dropout(0.1) if dropout else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ChannelLayerNorm3D(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TemporalFusionHead(nn.Module):
    """3D conv fusion head that preserves temporal dimension, then collapses T
    with a learned Conv3d(kernel_size=(T,1,1)) — each output class gets its own
    temporal weights.

    Input:  (B, C, T, H, W)
    Output: (B, n_classes, H, W)
    """

    def __init__(self, in_ch, n_classes=4, n_layers=2, num_frames=4):
        super().__init__()
        kernel = (3, 3, 3)
        pad = (1, 1, 1)
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Conv3d(in_ch, in_ch, kernel_size=kernel, padding=pad),
                ChannelLayerNorm3D(in_ch),
                nn.ReLU(inplace=True),
            ])
        self.layers = nn.Sequential(*layers)
        # Collapse T and project to n_classes in one step:
        # each class learns its own (in_ch, T) weighting
        self.temporal_proj = nn.Conv3d(in_ch, n_classes, kernel_size=(num_frames, 1, 1))

    def forward(self, x):
        x = self.layers(x)              # (B, in_ch, T, H, W)
        x = self.temporal_proj(x)        # (B, n_classes, 1, H, W)
        x = x.squeeze(2)                # (B, n_classes, H, W)
        return x


class Upscaler3D(nn.Module):
    """Conv3D temporal + ConvTranspose spatial upscaler.

    Takes (B, embed_dim, T, H, W) and produces (B, n_classes, H_out, W_out).
    T is preserved through all 3D blocks and fusion layers,
    then mean-pooled before a final 1x1 projection.

    Fully convolutional — works at any spatial resolution.
    For 48x48 crops: 3 -> 6 -> 12 -> 24 -> 48
    For 336x336 tiles: 21 -> 42 -> 84 -> 168 -> 336
    """

    def __init__(self, embed_dim, n_classes=4, n_layers=2, dropout=True, num_frames=4):
        super().__init__()

        self.blocks = nn.Sequential(
            Conv3DTemporalSpatialBlock(embed_dim, embed_dim // 2, dropout=dropout),
            Conv3DTemporalSpatialBlock(embed_dim // 2, embed_dim // 4, dropout=dropout),
            Conv3DTemporalSpatialBlock(embed_dim // 4, embed_dim // 8, dropout=dropout),
            Conv3DTemporalSpatialBlock(embed_dim // 8, embed_dim // 8, dropout=dropout),
        )

        final_ch = embed_dim // 8
        self.head = TemporalFusionHead(final_ch, n_classes, n_layers, num_frames=num_frames)

    def forward(self, x):
        x = self.blocks(x)
        x = self.head(x)
        return x


class PrithviSegConv3D(nn.Module):
    def __init__(self,
                 prithvi_params: dict,
                 prithvi_ckpt_path: str = None,
                 n_classes: int = 4,
                 model_size: str = "300m",
                 feed_timeloc: bool = False,
                 n_layers: int = 2):
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

        num_frames = prithvi_params["num_frames"]

        if model_size == "300m":
            self.head = Upscaler3D(embed_dim, n_classes=n_classes,
                                   n_layers=n_layers, num_frames=num_frames)
        else:
            raise ValueError(f"model_size {model_size} not supported")

    def forward(self, x):
        if isinstance(x, dict):
            x = {k: v.cuda() for k, v in x.items()}
        else:
            x = x.cuda()

        x = self.backbone(x)         # (B, 1+T*H*W, D)
        x = self.reshaper(x)          # (B, D, T, H, W)
        x = self.head(x)              # (B, n_classes, H_out, W_out)
        return x
