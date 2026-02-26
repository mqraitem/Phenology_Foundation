"""Generate all images for the task definition figure.

Produces:
  - figures/task_input_MM.png  for each month (01..12) from NH-2 2019
  - figures/task_gt_{g,m,s,d}.png  ground truth heatmaps for NH-2 2019
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- Config ---
HLS_DIR = "/projectnb/hlsfm/applications/lsp/outputs/HLS_composites_HP-LSP"
CSV = "results/m3-6-9-12/prithvi_pretrained_multiscale_crops_conv3d_1.0_test.csv"
OUT_DIR = "figures"
SITE, TILE, YEAR = "NH-2", "T18TYP", 2019
BANDS = ["G", "M", "S", "D"]
BAND_NAMES = {"G": "Greenup", "M": "Maturity", "S": "Senescence", "D": "Dormancy"}
CMAP = "viridis"
DPI = 150


def crop_and_save_input(month: int):
    """Load an HLS composite PNG, crop off whitespace/borders, and save."""
    fname = f"HLS_composite_{YEAR}-{month:02d}_{SITE}_{TILE}.png"
    src = os.path.join(HLS_DIR, fname)
    img = Image.open(src)
    arr = np.array(img)

    # Auto-crop: find bounding box of non-white pixels
    if arr.ndim == 3:
        mask = np.any(arr[:, :, :3] < 250, axis=2)
    else:
        mask = arr < 250
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) and len(cols):
        arr = arr[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(arr)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out = os.path.join(OUT_DIR, f"task_input_{month:02d}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  Saved {out}")


def reshape_doy(df, band, col_suffix="_truth_DOY"):
    """Reshape flat rows into a 2-D image."""
    max_r = int(df["row"].max() // 10) + 1
    max_c = int(df["col"].max() // 10) + 1
    image = np.full((max_r, max_c), np.nan)
    for _, row in df.iterrows():
        r = int(row["row"] // 10)
        c = int(row["col"] // 10)
        image[r, c] = row[f"{band}{col_suffix}"]
    return image


def gen_gt():
    """Generate ground truth phenology heatmaps."""
    df = pd.read_csv(CSV)
    sub = df[(df["HLStile"] == TILE) & (df["SiteID"] == SITE) & (df["years"] == YEAR)]
    print(f"Pixels for {TILE}/{SITE}/{YEAR}: {len(sub)}")

    for band in BANDS:
        img = reshape_doy(sub, band, col_suffix="_truth_DOY")
        vmin, vmax = np.nanmin(img), np.nanmax(img)
        print(f"  {band}: vmin={vmin:.1f}  vmax={vmax:.1f}")

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(img, cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        out = os.path.join(OUT_DIR, f"task_gt_{band.lower()}.png")
        fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"  Saved {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating input images (all 12 months)...")
    for m in range(1, 13):
        crop_and_save_input(m)

    print("\nGenerating ground truth maps...")
    gen_gt()


if __name__ == "__main__":
    main()
