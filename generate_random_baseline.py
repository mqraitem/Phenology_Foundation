"""Generate a random baseline that predicts by sampling from the training set DOY distribution.

For each pixel in the test/val set, the predicted DOY for each date (G, M, S, D)
is drawn independently from the empirical distribution of that date in the training set.

Outputs CSVs in the same format as other model results so they are picked up
by results_file() in the notebook.
"""
import numpy as np
import pandas as pd
import os
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

import path_config
from lib.utils import (get_data_paths, get_masks_paper, months_to_str,
                       get_months_subdir, get_results_dir)
from lib.dataloaders.dataloaders import CycleDataset


def collect_train_doy_distribution(selected_months):
    """Load training data and return arrays of truth DOY values per date."""
    n_timesteps = len(selected_months)
    m_str = months_to_str(selected_months)
    file_suffix = f"_m{m_str}"

    path_train = get_data_paths("training", 1.0, selected_months)
    dataset = CycleDataset(path_train, split="training", data_percentage=1.0,
                           n_timesteps=n_timesteps, file_suffix=file_suffix)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
    masks = get_masks_paper("train")

    date_names = ["G", "M", "S", "D"]
    doy_values = {d: [] for d in date_names}

    print("Collecting training set DOY distribution...")
    for _, data in tqdm(enumerate(loader), total=len(loader)):
        gt = data["gt_mask"]  # (B, 4, H, W)
        for b_idx, hls_tile_n in enumerate(data["hls_tile_name"]):
            mask = masks[hls_tile_n]
            rows, cols = np.where(mask.cpu().numpy())
            for i, d in enumerate(date_names):
                vals = gt[b_idx, i, rows, cols].numpy() * 547
                doy_values[d].extend(vals.tolist())

    for d in date_names:
        doy_values[d] = np.array(doy_values[d])
        print(f"  {d}: {len(doy_values[d])} pixels, "
              f"mean={doy_values[d].mean():.1f}, std={doy_values[d].std():.1f}")

    return doy_values


def generate_random_predictions(template_csv, doy_dist, seed=42):
    """Replace pred columns in a template CSV with random samples from training distribution."""
    df = pd.read_csv(template_csv)
    rng = np.random.RandomState(seed)
    n = len(df)

    for d in ["G", "M", "S", "D"]:
        col = f"{d}_pred_DOY"
        df[col] = rng.choice(doy_dist[d], size=n, replace=True)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_months", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected_months = args.selected_months
    results_dir = get_results_dir(selected_months)

    # Collect training DOY distribution
    doy_dist = collect_train_doy_distribution(selected_months)

    # Find an existing results CSV to use as template (need the metadata + truth columns)
    # Use any existing test/val CSV
    template_base = None
    for f in os.listdir(results_dir):
        if f.endswith("_test.csv") and "random_baseline" not in f:
            template_base = f.replace("_test.csv", "")
            break

    if template_base is None:
        print("No existing results CSV found to use as template!")
        return

    out_name = "random_baseline_1.0"
    for split in ["test", "val"]:
        template_path = os.path.join(results_dir, f"{template_base}_{split}.csv")
        if not os.path.exists(template_path):
            print(f"Template {template_path} not found, skipping {split}")
            continue

        out_path = os.path.join(results_dir, f"{out_name}_{split}.csv")
        df = generate_random_predictions(template_path, doy_dist, seed=args.seed)
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
