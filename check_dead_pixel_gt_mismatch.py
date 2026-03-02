"""Check how many dead image pixels have a valid (non -1) GT value.

A dead pixel is one where all bands are zero across all selected timesteps.
A GT is valid if at least one of the 4 phenological dates is not -1 (i.e. not
marked as 32767 or negative in the raw raster).

Usage:
    python check_dead_pixel_gt_mismatch.py --selected_months 3 6 9 12
    python check_dead_pixel_gt_mismatch.py --selected_months 3 6 9 12 --split test
"""

import argparse
import numpy as np
import rasterio
from tqdm import tqdm

from lib.utils import get_data_paths, normalize_doy


CORRECT_INDICES = [i - 1 for i in [2, 5, 8, 11]]  # 0-based GT band indices


def load_image(paths, target_size=330):
    """Load multi-temporal image as (6, T, H, W)."""
    imgs = []
    for p in paths:
        try:
            with rasterio.open(p) as src:
                img = src.read()[:, :target_size, :target_size].astype(np.float32)
        except Exception:
            img = np.zeros((6, target_size, target_size), dtype=np.float32)
        imgs.append(img[:, np.newaxis])
    return np.concatenate(imgs, axis=1)  # (6, T, H, W)


def load_gt(path, target_size=330):
    """Load GT and return (4, H, W) with -1 for invalid pixels."""
    with rasterio.open(path) as src:
        gt = src.read()[CORRECT_INDICES, :target_size, :target_size].astype(np.float32)
    invalid = (gt == 32767) | (gt < 0)
    gt = normalize_doy(gt)
    gt[invalid] = -1
    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_months", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument("--split", type=str, default="training",
                        choices=["training", "validation", "testing"],
                        help="Which data split to check")
    args = parser.parse_args()

    data_paths = get_data_paths(args.split, data_percentage=1.0,
                                selected_months=args.selected_months)
    print(f"Split: {args.split}  |  Tiles: {len(data_paths)}  |  "
          f"Months: {args.selected_months}\n")

    total_pixels = 0
    total_dead = 0
    total_dead_with_valid_gt = 0  # dead image but valid GT (the mismatch)

    per_date_dead_valid = np.zeros(4, dtype=np.int64)  # per phenological date

    for image_paths, gt_path, tile_name in tqdm(data_paths, desc="Tiles"):
        img = load_image(image_paths)          # (6, T, H, W)
        gt = load_gt(gt_path)                  # (4, H, W)

        H, W = img.shape[2], img.shape[3]
        total_pixels += H * W

        # Dead: all bands × all timesteps are zero for a pixel
        dead_mask = (img == 0).all(axis=(0, 1))  # (H, W)
        n_dead = dead_mask.sum()
        total_dead += int(n_dead)

        if n_dead == 0:
            continue

        # Valid GT: at least one date is not -1
        gt_valid_mask = (gt != -1).any(axis=0)  # (H, W)

        mismatch = dead_mask & gt_valid_mask     # dead image but valid GT
        total_dead_with_valid_gt += int(mismatch.sum())

        # Per-date breakdown: how many dead pixels have that specific date valid
        for d in range(4):
            per_date_dead_valid[d] += int((dead_mask & (gt[d] != -1)).sum())

    date_names = ["G (Greenup)", "M (Maturity)", "S (Senescence)", "D (Dormancy)"]

    print(f"{'Total pixels':<40s}: {total_pixels:>10,}")
    print(f"{'Dead pixels (all-zero image)':<40s}: {total_dead:>10,}  "
          f"({100 * total_dead / total_pixels:.2f}%)")
    print(f"{'Dead pixels WITH valid GT (mismatch)':<40s}: {total_dead_with_valid_gt:>10,}  "
          f"({100 * total_dead_with_valid_gt / max(total_dead, 1):.2f}% of dead pixels)")
    print()
    print("Per-date breakdown (dead pixels with that date valid):")
    for d, name in enumerate(date_names):
        print(f"  {name:<20s}: {per_date_dead_valid[d]:>10,}  "
              f"({100 * per_date_dead_valid[d] / max(total_dead, 1):.2f}% of dead pixels)")


if __name__ == "__main__":
    main()
