"""Generate output heatmap PNGs for the multiscale architecture figure.

Each phenology band (G, M, S, D) gets its own colour range (vmin/vmax)
so individual structure is visible, rather than stretching one range
across all four.

Saves: figures/output_g.png, output_m.png, output_s.png, output_d.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Config ---
CSV = "results/m3-6-9-12/prithvi_pretrained_multiscale_crops_conv3d_1.0_test.csv"
OUT_DIR = "figures"
TILE, SITE, YEAR = "T14TMT", "ND-3", 2019
BANDS = ["G", "M", "S", "D"]
CMAP = "viridis"
DPI = 150


def reshape_doy(df, band, col_suffix="_pred_DOY"):
    """Reshape flat rows into a 2-D image."""
    max_r = int(df["row"].max() // 10) + 1
    max_c = int(df["col"].max() // 10) + 1
    image = np.full((max_r, max_c), np.nan)
    for _, row in df.iterrows():
        r = int(row["row"] // 10)
        c = int(row["col"] // 10)
        image[r, c] = row[f"{band}{col_suffix}"]
    return image


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV)
    sub = df[(df["HLStile"] == TILE) & (df["SiteID"] == SITE) & (df["years"] == YEAR)]
    print(f"Pixels for {TILE}/{SITE}/{YEAR}: {len(sub)}")

    for band in BANDS:
        img = reshape_doy(sub, band)
        vmin, vmax = np.nanmin(img), np.nanmax(img)
        print(f"  {band}: vmin={vmin:.1f}  vmax={vmax:.1f}")

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(img, cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        out_path = os.path.join(OUT_DIR, f"output_{band.lower()}.png")
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
