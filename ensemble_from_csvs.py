"""Fit a convex ensemble from per-pixel result CSVs and save weights + test predictions.

Reads val CSVs to learn per-date convex weights, applies them to test CSVs,
and saves:
  1. Ensemble weights JSON to data/ensembles/{ensemble_name}.json
  2. Blended test CSV to results/{months_sub}/{ensemble_name}_test.csv
  3. Blended val CSV  to results/{months_sub}/{ensemble_name}_val.csv

The saved JSON can be loaded by visualize_tile_predictions.py with --ensemble_file.

Usage:
    python ensemble_from_csvs.py \
        --methods prithvi_pretrained_multiscale_crops_conv3d_crop48_1.0 \
                  shallow_transformer_pixels_1.0 \
        --selected_months 3 6 9 12 \
        --name my_ensemble
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from lib.utils import get_results_dir, months_to_str


PRED_COLS = ["G_pred_DOY", "M_pred_DOY", "S_pred_DOY", "D_pred_DOY"]
TRUTH_COLS = ["G_truth_DOY", "M_truth_DOY", "S_truth_DOY", "D_truth_DOY"]


def _fit_convex_weights(X, y):
    """Find w >= 0, sum(w) = 1 that minimizes MSE."""
    n = X.shape[1]
    w0 = np.full(n, 1.0 / n)
    res = minimize(
        fun=lambda w: np.mean((X @ w - y) ** 2),
        x0=w0,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
    )
    return res.x if res.success else w0


def sort_df(df):
    return df.sort_values(
        by=["years", "HLStile", "SiteID", "row", "col", "version"]
    ).reset_index(drop=True)


def load_csv(results_dir, method, split):
    """Load a result CSV for a method and split, raising a clear error if missing."""
    path = os.path.join(results_dir, f"{method}_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {split} CSV: {path}\n"
            f"Run eval_to_dataframe.py first to generate it."
        )
    return sort_df(pd.read_csv(path))


def fit_ensemble_weights(val_dfs, methods, min_rows=20):
    """Learn per-date convex weights on validation predictions.

    Args:
        val_dfs: list of DataFrames (one per method), sorted identically
        methods: list of method name strings
        min_rows: minimum valid rows to fit (else uniform)

    Returns:
        dict: {pred_col: [w0, w1, ...]} with weights per prediction column
    """
    n_models = len(val_dfs)
    uniform = np.full(n_models, 1.0 / n_models)
    weights = {}

    for pred_col, truth_col in zip(PRED_COLS, TRUTH_COLS):
        X = np.column_stack([df[pred_col].values for df in val_dfs])
        y = val_dfs[0][truth_col].values
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)

        if valid.sum() < min_rows:
            weights[pred_col] = uniform.copy()
        else:
            weights[pred_col] = _fit_convex_weights(X[valid], y[valid])

    return weights


def apply_ensemble_weights(dfs, weights):
    """Apply learned weights to produce a blended DataFrame.

    Args:
        dfs: list of DataFrames (one per method), sorted identically
        weights: {pred_col: weights_array}

    Returns:
        DataFrame with blended prediction columns
    """
    df_out = dfs[0].copy()
    for col in PRED_COLS:
        X = np.column_stack([df[col].values for df in dfs])
        df_out[col] = X @ weights[col]
    return df_out


def compute_mae(df):
    """Compute per-date and mean MAE from a results DataFrame."""
    maes = {}
    for pred_col, truth_col in zip(PRED_COLS, TRUTH_COLS):
        date = pred_col.split("_")[0]  # G, M, S, D
        maes[date] = np.mean(np.abs(df[pred_col].values - df[truth_col].values))
    maes["Mean"] = np.mean(list(maes.values()))
    return maes


def main():
    parser = argparse.ArgumentParser(
        description="Fit convex ensemble from result CSVs"
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", required=True,
        help="Method names (matching CSV prefixes, e.g. "
             "prithvi_pretrained_multiscale_crops_conv3d_crop48_1.0)",
    )
    parser.add_argument(
        "--selected_months", type=int, nargs="+", default=[3, 6, 9, 12],
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Ensemble name for output files. "
             "Default: ensemble_{method1}_{method2}_...",
    )
    args = parser.parse_args()

    results_dir = get_results_dir(args.selected_months)
    months_str = months_to_str(args.selected_months)

    # Default ensemble name
    if args.name is None:
        short_names = []
        for m in args.methods:
            short = m.replace("_1.0", "").replace("prithvi_pretrained_", "p_")
            short_names.append(short)
        args.name = "ensemble_" + "_AND_".join(short_names)

    print(f"Ensemble name: {args.name}")
    print(f"Methods: {args.methods}")
    print(f"Results dir: {results_dir}")
    print()

    # Load val and test CSVs
    val_dfs = [load_csv(results_dir, m, "val") for m in args.methods]
    test_dfs = [load_csv(results_dir, m, "test") for m in args.methods]

    # Verify row alignment
    for i, m in enumerate(args.methods):
        for col in ["years", "HLStile", "SiteID", "row", "col"]:
            if not (val_dfs[0][col] == val_dfs[i][col]).all():
                raise ValueError(
                    f"Val CSV row mismatch between {args.methods[0]} and {m} "
                    f"on column '{col}'. CSVs must have identical pixels."
                )
            if not (test_dfs[0][col] == test_dfs[i][col]).all():
                raise ValueError(
                    f"Test CSV row mismatch between {args.methods[0]} and {m} "
                    f"on column '{col}'. CSVs must have identical pixels."
                )

    # Fit weights on validation
    print("Fitting convex weights on validation set...")
    weights = fit_ensemble_weights(val_dfs, args.methods)

    print("\nEnsemble weights:")
    for col, w in weights.items():
        date = col.split("_")[0]
        w_str = "  ".join(
            f"{args.methods[i]}: {w[i]:.3f}" for i in range(len(args.methods))
        )
        print(f"  {date}: {w_str}")

    # Apply to val and test
    val_blended = apply_ensemble_weights(val_dfs, weights)
    test_blended = apply_ensemble_weights(test_dfs, weights)

    # Report MAE
    print("\nTest MAE (individual methods):")
    for m, df in zip(args.methods, test_dfs):
        maes = compute_mae(df)
        mae_str = "  ".join(f"{k}: {v:.1f}" for k, v in maes.items())
        print(f"  {m}: {mae_str}")

    maes = compute_mae(test_blended)
    mae_str = "  ".join(f"{k}: {v:.1f}" for k, v in maes.items())
    print(f"  {args.name}: {mae_str}")

    val_maes = compute_mae(val_blended)
    print(f"\nVal MAE (ensemble): "
          + "  ".join(f"{k}: {v:.1f}" for k, v in val_maes.items()))

    # Save ensemble weights JSON
    ensemble_dir = os.path.join("data", "ensembles", f"m{months_str}")
    os.makedirs(ensemble_dir, exist_ok=True)
    ensemble_path = os.path.join(ensemble_dir, f"{args.name}.json")

    ensemble_info = {
        "name": args.name,
        "methods": args.methods,
        "selected_months": args.selected_months,
        "weights": {col: w.tolist() for col, w in weights.items()},
        "val_mae": val_maes,
        "test_mae": maes,
    }
    with open(ensemble_path, "w") as f:
        json.dump(ensemble_info, f, indent=2)
    print(f"\nSaved ensemble weights: {ensemble_path}")

    # Save blended CSVs
    val_out = os.path.join(results_dir, f"{args.name}_val.csv")
    test_out = os.path.join(results_dir, f"{args.name}_test.csv")
    val_blended.to_csv(val_out, index=False)
    test_blended.to_csv(test_out, index=False)
    print(f"Saved val CSV:  {val_out}")
    print(f"Saved test CSV: {test_out}")


if __name__ == "__main__":
    main()
