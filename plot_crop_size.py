"""Plot MAE vs crop size for a given model group.

Reads cached evaluation results from analyze_hparams.py.
For each crop size, selects the best checkpoint (lowest mean MAE)
and plots a line chart of MAE (days) vs crop size.

Usage:
    python plot_crop_size.py --group prithvi_pretrained_crops_conv3d_1.0
    python plot_crop_size.py --group prithvi_pretrained_crops_cathls_conv3d_1.0
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from lib.utils import get_months_subdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True,
                        help="Model group name (e.g. prithvi_pretrained_crops_conv3d_1.0)")
    parser.add_argument("--selected_months", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument("--cache_dir", type=str, default="results",
                        help="Directory containing hparam_analysis_cache.csv")
    args = parser.parse_args()

    months_sub = get_months_subdir(args.selected_months)
    cache_path = os.path.join(args.cache_dir, months_sub, "hparam_analysis_cache.csv")

    if not os.path.exists(cache_path):
        print(f"Cache not found at {cache_path}")
        print("Run 'python analyze_hparams.py --eval' first to generate cached results.")
        return

    df = pd.read_csv(cache_path)

    # Filter to the requested group
    df_group = df[df["group"] == args.group]
    if df_group.empty:
        print(f"No results found for group '{args.group}'")
        print(f"Available groups: {sorted(df['group'].unique())}")
        return

    # For each crop size, pick the checkpoint with the lowest mean MAE
    dates = ["G", "M", "S", "D"]
    best_per_crop = []
    for crop_size, crop_df in df_group.groupby("crop"):
        best_idx = crop_df["Mean"].idxmin()
        best_row = crop_df.loc[best_idx]
        best_per_crop.append({
            "crop_size": int(crop_size),
            **{d: best_row[d] for d in dates},
            "Mean": best_row["Mean"],
            "checkpoint": best_row["checkpoint"],
        })

    best_df = pd.DataFrame(best_per_crop).sort_values("crop_size")
    print(best_df[["crop_size", "G", "M", "S", "D", "Mean"]].to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    for date in dates:
        ax.plot(best_df["crop_size"], best_df[date], marker="o", label=date)
    ax.plot(best_df["crop_size"], best_df["Mean"], marker="s", linewidth=2.5,
            color="black", label="Mean")

    ax.set_xlabel("Crop Size (pixels)", fontsize=13)
    ax.set_ylabel("MAE (days)", fontsize=13)
    ax.set_title(args.group, fontsize=14)
    ax.set_xticks(best_df["crop_size"].values)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs("figures", exist_ok=True)
    out_path = f"figures/crop_size_mae_{args.group}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
