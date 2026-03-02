"""Visualize per-tile predictions for multiple models side-by-side with ground truth.

Produces one figure per tile as a (n_rows) x 4 grid of heatmaps.
Rows = models [+ ensemble] + GT, columns = G / M / S / D phenological dates.

Usage:
    python visualize_tile_predictions.py \
        --models prithvi_pretrained_crops_conv3d_1.0 \
                 prithvi_pretrained_multiscale_crops_conv3d_crop48_1.0 \
        --n_samples 4 \
        --selected_months 3 6 9 12

    # With a pre-computed ensemble from ensemble_from_csvs.py:
    python visualize_tile_predictions.py \
        --models prithvi_pretrained_multiscale_crops_conv3d_crop48_1.0 \
                 shallow_transformer_pixels_1.0 \
        --n_samples 4 \
        --ensemble_file data/ensembles/m3-6-9-12/my_ensemble.json \
        --selected_months 3 6 9 12
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from tqdm import tqdm

import path_config
from lib.utils import (
    build_model, get_data_paths, get_months_subdir, months_to_str,
    get_results_dir, batched_sliding_window,
)
from lib.dataloaders.dataloaders import CycleDataset

PHASE_NAMES = ["Greenup", "Maturity", "Senescence", "Dormancy"]
TILE_SIZE = 330


def full_tile_predict(model, image, device):
    """Run full-tile inference (no cropping)."""
    with torch.no_grad():
        pred = model(image)
    pred = pred[:, :, :TILE_SIZE, :TILE_SIZE].squeeze(0)  # (4, 330, 330)
    return pred.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Visualize tile predictions for multiple models")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Model group names from best_params.csv")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of tiles to visualize")
    parser.add_argument("--selected_months", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble_file", type=str, default=None,
                        help="Path to ensemble JSON from ensemble_from_csvs.py")
    parser.add_argument("--output_dir", type=str, default="figures/viz")
    args = parser.parse_args()

    selected_months = args.selected_months
    n_timesteps = len(selected_months)
    months_sub = get_months_subdir(selected_months)
    months_str = months_to_str(selected_months)
    file_suffix = f"_m{months_str}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load best params
    results_dir = get_results_dir(selected_months)
    best_df = pd.read_csv(os.path.join(results_dir, "best_params.csv"))
    best_lookup = dict(zip(best_df["Model Name"], best_df["Best Param"]))

    for m in args.models:
        if m not in best_lookup:
            raise ValueError(f"Model '{m}' not found in best_params.csv. "
                             f"Available: {list(best_lookup.keys())}")

    # Load test dataset
    data_path = get_data_paths("testing", 1.0, selected_months)
    dataset = CycleDataset(data_path, split="testing", data_percentage=1.0,
                           n_timesteps=n_timesteps, file_suffix=file_suffix)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    n_tiles = len(dataset)
    rng = np.random.RandomState(args.seed)
    sample_indices = sorted(rng.choice(n_tiles, size=min(args.n_samples, n_tiles), replace=False))
    sample_set = set(sample_indices)

    print(f"Selected {len(sample_indices)} tiles out of {n_tiles}")

    # First pass: collect GT for selected tiles and compute dataset-level min/max
    gt_dict = {}       # tile_name -> (4, 330, 330) numpy
    images_dict = {}   # tile_name -> (1, C, T, H, W) tensor
    tile_names_ordered = []

    # We also need global min/max across ALL test GT for consistent colormaps
    global_min = np.full(4, np.inf)
    global_max = np.full(4, -np.inf)

    print("Collecting ground truth and computing min/max...")
    for idx, data in tqdm(enumerate(loader), total=n_tiles):
        gt = data["gt_mask"].squeeze(0).numpy()  # (4, H, W)
        gt_330 = gt[:, :TILE_SIZE, :TILE_SIZE]

        # Update global min/max from valid pixels
        for d in range(4):
            valid = gt_330[d][gt_330[d] != -1]
            if len(valid) > 0:
                global_min[d] = min(global_min[d], valid.min())
                global_max[d] = max(global_max[d], valid.max())

        if idx in sample_set:
            tile_name = data["hls_tile_name"][0]
            tile_names_ordered.append(tile_name)
            gt_dict[tile_name] = gt_330
            images_dict[tile_name] = data["image"]

    # Convert normalized DOY to actual DOY for display
    DOY_SCALE = 547
    global_min_doy = global_min * DOY_SCALE
    global_max_doy = global_max * DOY_SCALE

    print(f"DOY ranges: {list(zip(PHASE_NAMES, global_min_doy.astype(int), global_max_doy.astype(int)))}")

    # Load ensemble weights if provided
    PRED_COLS = ["G_pred_DOY", "M_pred_DOY", "S_pred_DOY", "D_pred_DOY"]
    ensemble_info = None
    if args.ensemble_file is not None:
        with open(args.ensemble_file, "r") as f:
            ensemble_info = json.load(f)
        # Verify that all ensemble methods are in --models
        for m in ensemble_info["methods"]:
            if m not in args.models:
                raise ValueError(
                    f"Ensemble method '{m}' from {args.ensemble_file} "
                    f"is not in --models. Add it to --models."
                )
        print(f"\nLoaded ensemble weights from {args.ensemble_file}")
        for col in PRED_COLS:
            date = col.split("_")[0]
            w = ensemble_info["weights"][col]
            w_str = "  ".join(
                f"{ensemble_info['methods'][i]}: {w[i]:.3f}"
                for i in range(len(w))
            )
            print(f"  {date}: {w_str}")

    # Run inference for each model
    # model_preds[model_name][tile_name] = (4, 330, 330) numpy  (test, selected tiles only)
    model_preds = {}

    ckpt_root = path_config.get_checkpoint_root()

    for model_name in args.models:
        print(f"\nLoading model: {model_name}")
        best_param = best_lookup[model_name]
        ckpt_path = os.path.join(ckpt_root, months_sub, model_name, best_param)
        ckpt = torch.load(ckpt_path, map_location=device)

        model, feed_timeloc, crop_size = build_model(model_name, best_param, n_timesteps)
        model = model.to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Test inference (selected tiles only)
        model_preds[model_name] = {}
        print(f"  Running test inference (crop_size={crop_size})...")
        for tile_name in tqdm(tile_names_ordered, desc="    test tiles"):
            image = images_dict[tile_name]
            if crop_size is not None:
                with torch.no_grad():
                    pred = batched_sliding_window(
                        model, image, crop_size, device,
                    ).cpu().numpy()
            else:
                pred = full_tile_predict(model, image, device)
            model_preds[model_name][tile_name] = pred

        del model
        torch.cuda.empty_cache()

    # Apply pre-computed ensemble weights to test predictions
    ENSEMBLE_KEY = "__ensemble__"
    if ensemble_info is not None:
        print("\nApplying ensemble weights to predictions...")
        ensemble_methods = ensemble_info["methods"]
        # weights is {pred_col: [w0, w1, ...]} — one weight per date per method
        # Convert to (4, n_methods) array matching PRED_COLS order
        ensemble_weights = np.array([ensemble_info["weights"][col] for col in PRED_COLS])

        model_preds[ENSEMBLE_KEY] = {}
        for tile_name in tile_names_ordered:
            ens_pred = np.zeros((4, TILE_SIZE, TILE_SIZE))
            for d in range(4):
                for i, m in enumerate(ensemble_methods):
                    ens_pred[d] += ensemble_weights[d, i] * model_preds[m][tile_name][d]
            model_preds[ENSEMBLE_KEY][tile_name] = ens_pred

    # Build column order: models [+ ensemble] + GT
    col_keys = list(args.models)
    if ensemble_info is not None:
        col_keys.append(ENSEMBLE_KEY)

    n_cols = len(col_keys) + 1  # + GT
    n_rows = 4  # G, M, S, D

    # Build short display names for columns
    display_names = []
    for key in col_keys:
        if key == ENSEMBLE_KEY:
            display_names.append(ensemble_info.get("name", "Ensemble"))
        else:
            short = key.replace("prithvi_pretrained_", "").replace("_1.0", "").replace("_", " ")
            display_names.append(short)
    display_names.append("Ground Truth")

    cmap = plt.cm.RdYlGn_r
    bad_color = "0.85"

    for tile_name in tile_names_ordered:
        # Gridspec: 4 date rows, model columns + one thin column for colorbars on the right
        fig = plt.figure(figsize=(n_cols * 3.5 + 0.6, 4 * 3.5))
        gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.05],
                              hspace=0.15, wspace=0.08)

        axes = np.empty((n_rows, n_cols), dtype=object)
        for r in range(n_rows):
            for c in range(n_cols):
                axes[r, c] = fig.add_subplot(gs[r, c])

        last_im_per_row = [None] * n_rows  # track last imshow per row for colorbar

        gt_330 = gt_dict[tile_name]
        invalid_mask = gt_330[0] == -1  # same mask across all dates

        for row, phase in enumerate(PHASE_NAMES):
            vmin = global_min_doy[row]
            vmax = global_max_doy[row]

            cmap_copy = cmap.copy()
            cmap_copy.set_bad(bad_color)

            # Model columns
            for col_idx, col_key in enumerate(col_keys):
                ax = axes[row, col_idx]
                pred = model_preds[col_key][tile_name][row] * DOY_SCALE
                pred_masked = np.ma.masked_where(invalid_mask, pred)

                im = ax.imshow(pred_masked, vmin=vmin, vmax=vmax, cmap=cmap_copy)
                ax.set_xticks([])
                ax.set_yticks([])
                last_im_per_row[row] = im

                if row == 0:
                    ax.set_title(display_names[col_idx], fontsize=10, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(phase, fontsize=12, fontweight="bold")

            # GT column (last column)
            gt_col = len(col_keys)
            ax = axes[row, gt_col]
            gt_doy = gt_330[row] * DOY_SCALE
            gt_masked = np.ma.masked_where(invalid_mask, gt_doy)

            im = ax.imshow(gt_masked, vmin=vmin, vmax=vmax, cmap=cmap_copy)
            ax.set_xticks([])
            ax.set_yticks([])
            last_im_per_row[row] = im

            if row == 0:
                ax.set_title("Ground Truth", fontsize=10, fontweight="bold")

        # Per-row vertical colorbar on the right
        for row in range(n_rows):
            cbar_ax = fig.add_subplot(gs[row, n_cols])
            fig.colorbar(last_im_per_row[row], cax=cbar_ax, orientation="vertical",
                         label="DOY")

        fig.suptitle(tile_name, fontsize=14, fontweight="bold")

        out_path = os.path.join(args.output_dir, f"{tile_name}.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")

    print(f"\nDone. {len(tile_names_ordered)} figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
