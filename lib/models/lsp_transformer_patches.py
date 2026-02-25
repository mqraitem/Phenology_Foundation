import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (N, T, d_model)
        return x + self.pe[:, :x.size(1), :]


def _edge_cover_starts(L: int, p: int):
    """
    Non-overlap starts (0, p, 2p, ...) plus, if needed, one extra start at L - p
    so the last patch touches the edge (minimal overlap). If L <= p -> [0].
    """
    if L <= p:
        return [0]
    starts = list(range(0, L - p + 1, p))
    last = L - p
    if starts[-1] != last:
        starts.append(last)   # edge-only overlap
    return starts


class TemporalTransformerPerPatch(nn.Module):
    """
    Temporal-only transformer that operates per patch.
    - processing_images=True: (B, C, T, H, W) -> (B, num_classes, H, W)
      * Interior: stride=patch (no overlap)
      * Edge-only overlap when H or W not divisible by patch size (last start at L-p)
      * Edge patches overwrite any tiny overlap from previous patch
    - processing_images=False: (B, T, C, H, W) where (H,W)==patch -> (B, num_classes, H, W)
    """
    def __init__(
        self,
        input_channels=6,
        seq_len=4,
        num_classes=4,
        d_model=256,
        nhead=8,
        num_layers=3,
        dropout=0.1,
        patch_size=(32, 32),
        pad_mode="replicate",   # used only if image smaller than patch
        pad_value=0.0,
    ):
        super().__init__()
        self.C = input_channels
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.ph, self.pw = patch_size
        self.pad_mode = pad_mode
        self.pad_value = pad_value

        patch_feat = input_channels * self.ph * self.pw  # feature length per time step
        self.input_proj = nn.Linear(patch_feat, d_model, bias=False)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        # Output per-pixel predictions for the entire patch
        self.output_proj = nn.Linear(d_model, num_classes * self.ph * self.pw)

    # ---- internals ----
    def _temporal_patch_encode(self, x_seq, chunk_size=2048):
        """
        x_seq: (Npatches, T, C*ph*pw) -> (Npatches, num_classes, ph, pw)
        """
        x_seq = self.input_proj(x_seq)         # (N, T, d_model)
        x_seq = self.pos_encoder(x_seq)        # (N, T, d_model)
        outs = []
        N = x_seq.size(0)

        for i in range(0, N, chunk_size):
            z = self.encoder(x_seq[i:i+chunk_size])  # (chunk, T, d_model)
            z = z.mean(dim=1)                        # temporal pooling
            z = self.dropout(z)
            out = self.output_proj(z)                # (chunk, K*ph*pw)
            out = out.view(-1, self.num_classes, self.ph, self.pw)  # (chunk, K, ph, pw)
            outs.append(out)
        return torch.cat(outs, dim=0)


    # ---- forward ----
    def forward(self, x, processing_images=True, chunk_size=2048):

        x = x.cuda()

        if processing_images:
            # x: (B, C, T, H, W)
            assert x.dim() == 5, "processing_images=True expects (B, C, T, H, W)"

            # Crop to native resolution to match training patch positions
            x = x[:, :, :, :330, :330]

            B, C, T, Huse, Wuse = x.shape
            assert C == self.C, f"input_channels mismatch: {C} vs {self.C}"
            assert T == self.seq_len, f"T={T} vs seq_len={self.seq_len}"

            # compute starts with edge-only overlap
            starts_h = _edge_cover_starts(Huse, self.ph)
            starts_w = _edge_cover_starts(Wuse, self.pw)
            positions = [(top, left) for top in starts_h for left in starts_w]
            P = len(positions)

            # Extract all patches: (P, B, C, T, ph, pw)
            patches = torch.stack([
                x[:, :, :, top:top+self.ph, left:left+self.pw]
                for top, left in positions
            ], dim=0)

            # Reshape: (P, B, C, T, ph, pw) -> (B*P, T, C*ph*pw)
            patches = patches.permute(1, 0, 3, 2, 4, 5).reshape(B * P, T, -1)

            # Process all patches; chunk_size controls batching through encoder
            logits = self._temporal_patch_encode(patches, chunk_size=chunk_size)  # (B*P, K, ph, pw)
            logits = logits.view(B, P, self.num_classes, self.ph, self.pw)  # (B, P, K, ph, pw)

            # Scatter results back to spatial output, averaging overlapping patches
            out = x.new_zeros((B, self.num_classes, Huse, Wuse))
            count = x.new_zeros((1, 1, Huse, Wuse))
            for i, (top, left) in enumerate(positions):
                out[:, :, top:top+self.ph, left:left+self.pw] += logits[:, i]
                count[:, :, top:top+self.ph, left:left+self.pw] += 1
            out = out / count

            return out  # (B, K, H, W)

        # processing_images == False
        # x: (B, T, C, H, W) where (H,W) == patch_size
        assert x.dim() == 5, "processing_images=False expects (B, T, C, H, W)"
        B, T, C, H, W = x.shape
        assert C == self.C, f"input_channels mismatch: {C} vs {self.C}"
        assert T == self.seq_len, f"T={T} vs seq_len={self.seq_len}"
        assert (H, W) == (self.ph, self.pw), f"Patch size mismatch: ({H},{W}) vs ({self.ph},{self.pw})"

        seq = x.reshape(B, T, C * H * W)                      # (B, T, C*H*W)
        logits = self._temporal_patch_encode(seq, chunk_size=chunk_size)  # (B, K, ph, pw)
        return logits
