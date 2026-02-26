import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import path_config
from lib.utils import get_results_dir

def plot_methods_regions_split(
    results_mae_map,
    methods=("Transformer-LSP", "Prithvi-Pretraining", "Prithvi-Pretraining-Freeze"),
    geo_path="/projectnb/hlsfm/applications/lsp/ancillary/HP_LSP/geotiff_extents.geojson",
    eco_path="useco1/NA_CEC_Eco_Level1.shp",
    region_column="NA_L1NAME",
    date_order=("G", "M", "S", "D", "Mean")
):

    # === Load tile geometries ===
    geo_data = gpd.read_file(geo_path)
    geo_data = geo_data.rename(columns={"Site_ID": "SiteID"})
    geo_data["HLStile"] = "T" + geo_data["name"]
    geo_data = geo_data.set_crs("EPSG:4326").to_crs(epsg=3857)
    geo_data["centroid"] = geo_data.geometry.centroid

    # === Load ecoregion shapefile ===
    eco_gdf = gpd.read_file(eco_path).to_crs(geo_data.crs)

    # === Filter to desired methods and original dates (excluding "Mean") ===
    mae_filtered = results_mae_map[results_mae_map["Method"].isin(methods)].copy()
    original_dates = [d for d in date_order if d != "Mean"]
    mae_filtered = mae_filtered[mae_filtered["Date"].isin(original_dates)]

    # === Prepare GeoDataFrame of result centroids ===
    results_df = mae_filtered[["HLStile", "SiteID", "Method", "Date", "MAE"]]
    results_df = results_df.merge(
        geo_data[["HLStile", "SiteID", "centroid"]],
        on=["HLStile", "SiteID"],
        how="left"
    )
    results_gdf = gpd.GeoDataFrame(results_df, geometry="centroid", crs=geo_data.crs)
    results_gdf = results_gdf.dropna(subset=["centroid"])

    # === Spatial join: assign each result to an ecoregion ===
    results_with_region = gpd.sjoin(results_gdf, eco_gdf, how="inner", predicate="intersects")

    # === Group by region, date, and method ===
    region_method_mae = (
        results_with_region
        .groupby([region_column, "Date", "Method"])["MAE"]
        .mean()
        .reset_index()
    )

    # === Calculate Mean across dates and append ===
    if "Mean" in date_order:
        mean_mae = (
            region_method_mae
            .groupby([region_column, "Method"])["MAE"]
            .mean()
            .reset_index()
        )
        mean_mae["Date"] = "Mean"
        region_method_mae = pd.concat([region_method_mae, mean_mae], ignore_index=True)

    # === Order regions by overall mean MAE (optional, for nicer plotting) ===
    region_order = (
        region_method_mae
        .groupby(region_column)["MAE"]
        .mean()
        .sort_values(ascending=False)  # or True, depending on your preference
        .index
    )

    # Ensure Date is categorical with desired order
    region_method_mae["Date"] = pd.Categorical(
        region_method_mae["Date"], categories=list(date_order), ordered=True
    )

    # === Plot: facet by Date, compare methods within each region ===
    g = sns.catplot(
        data=region_method_mae,
        kind="bar",
        x="MAE",
        y=region_column,
        hue="Method",
        col="Date",
        col_order=date_order,
        order=region_order,
        palette="Set2",
        sharey=True,
        sharex=False,
        height=4,
        aspect=1.1
    )

    g.set_axis_labels("MAE", "Ecoregion")
    g.set_titles("Date: {col_name}")
    g._legend.set_title("Method")
    #set legend to be on top and two columns 
    g._legend.set_bbox_to_anchor((0, 1.02, 0.5, 0.2))
    g._legend.set_ncols(2)
    

    # Increase tick font sizes a bit
    for ax in g.axes.flat:
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_methods_regions(results_mae_map,
                         methods = ["Transformer-LSP", "Prithvi-Pretraining", "Prithvi-Pretraining-Freeze"],
                        geo_path="/projectnb/hlsfm/applications/lsp/ancillary/HP_LSP/geotiff_extents.geojson",
                        eco_path="useco1/NA_CEC_Eco_Level1.shp"):
    # === Load tile geometries ===

    geo_data = gpd.read_file(geo_path)
    geo_data = geo_data.rename(columns={"Site_ID": "SiteID"})
    geo_data["HLStile"] = "T" + geo_data["name"]
    geo_data = geo_data.set_crs("EPSG:4326").to_crs(epsg=3857)
    geo_data["centroid"] = geo_data.geometry.centroid

    # === Load ecoregion shapefile ===
    eco_gdf = gpd.read_file(eco_path).to_crs(geo_data.crs)  # project to match tile centroids

    mae_filtered = results_mae_map[results_mae_map["Method"].isin(methods)]

    # === Prepare GeoDataFrame of result centroids ===
    # Calculate mean MAE across all dates for each HLStile/SiteID/Method combination
    results_df = mae_filtered[["HLStile", "SiteID", "Method", "MAE"]].copy()
    results_df = results_df.groupby(["HLStile", "SiteID", "Method"])["MAE"].mean().reset_index()
    results_df = results_df.merge(
        geo_data[["HLStile", "SiteID", "centroid"]],
        on=["HLStile", "SiteID"],
        how="left"
    )
    results_gdf = gpd.GeoDataFrame(results_df, geometry="centroid", crs=geo_data.crs)
    results_gdf = results_gdf.dropna(subset=["centroid"])  # drop unmatched rows

    # === Spatial join: assign each result to an ecoregion ===
    results_with_region = gpd.sjoin(results_gdf, eco_gdf, how="inner", predicate="intersects")

    # === Choose desired region level ===
    region_column = "NA_L1NAME"  # or US_L1NAME or US_L3NAME for more/fewer regions
    # === Group by region and method ===
    region_method_mae = (
        results_with_region.groupby([region_column, "Method"])["MAE"]
        .mean()
        .reset_index()
    )

    # === Plot grouped bar chart ===
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=region_method_mae,
        x="MAE",
        y=region_column,
        hue="Method",
        palette="Set2"
    )
    plt.title("Average MAE by Ecoregion and Method", fontsize=16)
    plt.ylabel("Average MAE")
    plt.xlabel("Ecoregion")
    plt.xticks(rotation=90)
    plt.legend(title="Method")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_mae_improvement_by_region(
    results_mae_map,
    geo_path,
    eco_path,
    methods_to_compare,
    anchor_method,
    region_column="NA_L1NAME",
    figsize=(12, 8)
):
    import geopandas as gpd
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # === Load tile extent data ===
    geo_data = gpd.read_file(geo_path)
    geo_data = geo_data.rename(columns={"Site_ID": "SiteID"})
    geo_data["HLStile"] = "T" + geo_data["name"]
    geo_data = geo_data.set_crs("EPSG:4326").to_crs(epsg=3857)
    geo_data["centroid"] = geo_data.geometry.centroid

    # === Filter and pivot methods ===
    methods = methods_to_compare + [anchor_method]
    mae_filtered = results_mae_map[results_mae_map["Method"].isin(methods)]

    mae_pivot = (
        mae_filtered.groupby(["HLStile", "SiteID", "Method"])["MAE"]
        .mean()
        .unstack()
        .dropna(subset=[anchor_method])
    )

    # === Calculate improvement relative to anchor ===
    for method in methods_to_compare:
        mae_pivot[f"diff_{method}"] = mae_pivot[anchor_method] - mae_pivot[method] 

    # === Merge centroids ===
    mae_pivot = mae_pivot.reset_index()
    mae_pivot = mae_pivot.merge(
        geo_data[["HLStile", "SiteID", "centroid"]],
        on=["HLStile", "SiteID"],
        how="left"
    )
    gdf = gpd.GeoDataFrame(mae_pivot, geometry="centroid", crs=geo_data.crs)

    # === Load ecoregions ===
    eco_gdf = gpd.read_file(eco_path).to_crs(gdf.crs)
    gdf = gpd.sjoin(gdf, eco_gdf, how="left", predicate="intersects")

    # === Group by region and calculate mean improvements ===
    diff_cols = [f"diff_{m}" for m in methods_to_compare]
    region_summary = gdf.groupby(region_column)[diff_cols].mean().reset_index()

    # === Sort regions by average improvement across all methods ===
    region_summary["mean_diff"] = region_summary[diff_cols].mean(axis=1)
    region_summary = region_summary.sort_values("mean_diff", ascending=False)

    # === Melt for plotting ===
    melted = region_summary.melt(id_vars=[region_column, "mean_diff"],
                                 value_vars=diff_cols,
                                 var_name="Method",
                                 value_name="MAE_diff")
    melted["Method"] = melted["Method"].str.replace("diff_", "")

    # Set region as a categorical variable sorted by mean_diff
    melted[region_column] = pd.Categorical(
        melted[region_column],
        categories=region_summary[region_column],
        ordered=True
    )

    # === Plot ===
    plt.figure(figsize=figsize)
    sns.barplot(
        data=melted,
        x="MAE_diff",
        y=region_column,
        hue="Method",
        palette="Set2"
    )
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel(f"MAE Difference vs '{anchor_method}' (positive = better, negative = worse)")
    plt.ylabel("Ecoregion")
    plt.title(f"Average MAE Difference by Region (vs {anchor_method})", fontsize=14)
    plt.tight_layout()
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.legend(title="Method", loc="lower right")
    # plt.show()

    #increase yticks size 
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    # plt.show()

    #remove title and xaxis label
    plt.title("")
    plt.xlabel("")
    #remove legend
    plt.legend([])
    plt.show()


def print_avg_results(methods, results_mae):
    rows = []
    for method in methods:
        row = {"Method": method}
        for date in ["G", "M", "S", "D"]:
            avg_mae = results_mae[(results_mae["Method"] == method) & (results_mae["Date"] == date)]["MAE"].mean()
            row[date] = avg_mae
        row["Mean"] = results_mae[results_mae["Method"] == method]["MAE"].mean()
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Method")
    print(df.round(1).to_string())



def add_region_to_results(
    results_df: pd.DataFrame,
    geo_path: str,
    eco_path: str,
    region_column: str = "NA_L1NAME",
    tile_name_col: str = "name",   # column in geo_path used to form "HLStile" = "T" + name
    site_id_col_geo: str = "Site_ID",  # will be renamed to "SiteID"
    predicate: str = "within"      # "within" for point-in-polygon (centroids), or "intersects"
) -> pd.DataFrame:

    # --- Load tile geometries and prepare centroids ---
    geo_gdf = gpd.read_file(geo_path).copy()
    if site_id_col_geo in geo_gdf.columns and "SiteID" not in geo_gdf.columns:
        geo_gdf = geo_gdf.rename(columns={site_id_col_geo: "SiteID"})
    if "HLStile" not in geo_gdf.columns:
        if tile_name_col not in geo_gdf.columns:
            raise ValueError(f"'{tile_name_col}' not found in tile file; provide the correct 'tile_name_col'.")
        geo_gdf["HLStile"] = "T" + geo_gdf[tile_name_col].astype(str)

    # ensure geographic CRS then project to a metric CRS for robust centroid calculation
    geo_gdf = geo_gdf.set_crs("EPSG:4326", allow_override=True).to_crs(3857)
    geo_gdf["geometry"] = geo_gdf.geometry  # ensure active geometry
    geo_gdf["centroid"] = geo_gdf.geometry.centroid

    centroids = gpd.GeoDataFrame(
        geo_gdf[["HLStile", "SiteID"]].copy(),
        geometry=geo_gdf["centroid"],
        crs=geo_gdf.crs
    ).drop_duplicates(subset=["HLStile", "SiteID"])

    # --- Load ecoregions and project to match ---
    eco_gdf = gpd.read_file(eco_path).to_crs(centroids.crs)
    if region_column not in eco_gdf.columns:
        raise ValueError(f"'{region_column}' not found in ecoregion data. Available columns: {list(eco_gdf.columns)}")

    # --- Spatial join (point-in-polygon for centroids) ---
    joined = gpd.sjoin(centroids, eco_gdf[[region_column, "geometry"]], how="left", predicate=predicate)
    joined = joined[["HLStile", "SiteID", region_column]].drop_duplicates()

    # --- Merge region back to the original results ---
    out = results_df.copy()
    out = out.merge(joined, on=["HLStile", "SiteID"], how="left")

    return out

################################################################################################################################################
# Convex Ensemble Regression (per region)
# Learns weights w_i >= 0, sum(w_i) = 1 to blend predictions from multiple methods
################################################################################################################################################
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scipy.optimize import minimize


def _fit_convex_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find w >= 0, sum(w) = 1 that minimizes MSE."""
    n = X.shape[1]
    w0 = np.full(n, 1.0 / n)
    res = minimize(
        fun=lambda w: np.mean((X @ w - y) ** 2),
        x0=w0,
        method='SLSQP',
        bounds=[(0, 1)] * n,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    )
    return res.x if res.success else w0


def learn_convex_weights(
    dfs: List[pd.DataFrame],
    region_col: str = 'NA_L1NAME',
    min_rows: int = 20
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Learn convex combination weights per region and prediction column.

    Returns: {region: {pred_col: weights_array}, "__DEFAULT__": {...}}
    """
    n_models = len(dfs)
    uniform = np.full(n_models, 1.0 / n_models)
    pred_cols = [c for c in dfs[0].columns if '_pred_' in c]

    def fit_for_mask(mask):
        weights = {}
        for col in pred_cols:
            X = np.column_stack([df.loc[mask, col].values for df in dfs])
            y = dfs[0].loc[mask, col.replace('_pred_', '_truth_')].values
            valid = np.isfinite(X).all(axis=1) & np.isfinite(y)

            if valid.sum() < min_rows:
                weights[col] = uniform.copy()
            else:
                weights[col] = _fit_convex_weights(X[valid], y[valid])
        return weights

    # Global (default) weights
    global_weights = fit_for_mask(np.ones(len(dfs[0]), dtype=bool))
    models = {"__DEFAULT__": global_weights}

    # Per-region weights
    for region in dfs[0][region_col].dropna().unique():
        mask = dfs[0][region_col] == region
        if mask.sum() >= min_rows:
            models[region] = fit_for_mask(mask)
        else:
            models[region] = global_weights

    return models


def apply_convex_weights(
    dfs: List[pd.DataFrame],
    weights: Dict[str, Dict[str, np.ndarray]],
    region_col: str = 'NA_L1NAME'
) -> pd.DataFrame:
    """Apply learned weights to get blended predictions."""
    df_out = dfs[0].copy()
    pred_cols = [c for c in df_out.columns if '_pred_' in c]
    default_w = weights["__DEFAULT__"]

    for col in pred_cols:
        X = np.column_stack([df[col].values for df in dfs])
        w_col = np.zeros(len(df_out))

        for region in df_out[region_col].dropna().unique():
            mask = df_out[region_col] == region
            w = weights.get(region, default_w).get(col, default_w[col])
            w_col[mask] = X[mask] @ w

        # Handle NaN regions with default weights
        nan_mask = df_out[region_col].isna()
        if nan_mask.any():
            w_col[nan_mask] = X[nan_mask] @ default_w[col]

        df_out[col] = w_col

    return df_out

def convex_ensemble(
    method_names: List[str],
    results_val: Dict[str, pd.DataFrame],
    results_test: Dict[str, pd.DataFrame],
    out_name: str,
    region_col: str = 'NA_L1NAME',
    min_rows: int = 20
) -> Dict[str, Dict[str, np.ndarray]]:
    """Learn weights on val, apply to test, store result in results_test[out_name]."""
    dfs_val = [results_val[f"{m}_val"] for m in method_names]
    dfs_test = [results_test[f"{m}_test"] for m in method_names]

    weights = learn_convex_weights(dfs_val, region_col, min_rows)
    results_test[out_name] = apply_convex_weights(dfs_test, weights, region_col)
    return weights


def simple_ensemble(
    method_names: List[str],
    results_val: Dict[str, pd.DataFrame],
    results_test: Dict[str, pd.DataFrame],
    out_name: str,
    min_rows: int = 20,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Learn a single global set of convex weights on val, apply to test.

    Unlike convex_ensemble, this does NOT split by region — it fits one
    weight vector per prediction column across the entire validation set.

    Returns: {pred_col: weights_array}
    """
    dfs_val = [results_val[f"{m}_val"] for m in method_names]
    dfs_test = [results_test[f"{m}_test"] for m in method_names]

    n_models = len(dfs_val)
    uniform = np.full(n_models, 1.0 / n_models)
    pred_cols = [c for c in dfs_val[0].columns if '_pred_' in c]

    # Fit global weights on val
    weights = {}
    for col in pred_cols:
        X = np.column_stack([df[col].values for df in dfs_val])
        y = dfs_val[0][col.replace('_pred_', '_truth_')].values
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if valid.sum() < min_rows:
            weights[col] = uniform.copy()
        else:
            weights[col] = _fit_convex_weights(X[valid], y[valid])

    if verbose:
        print(f"Simple ensemble weights ({out_name}):")
        for col, w in weights.items():
            w_str = "  ".join(f"{method_names[i]}: {w[i]:.3f}" for i in range(n_models))
            print(f"  {col}: {w_str}")

    # Apply global weights to test
    df_out = dfs_test[0].copy()
    for col in pred_cols:
        X = np.column_stack([df[col].values for df in dfs_test])
        df_out[col] = X @ weights[col]

    results_test[out_name] = df_out
    return weights


################################################################################################################################################
# Oracle Ensemble — upper bound that fits optimal convex weights on the test set itself
################################################################################################################################################

def oracle_ensemble(
    method_names: List[str],
    results_test: Dict[str, pd.DataFrame],
    out_name: str,
    region_col: str = 'NA_L1NAME',
    min_rows: int = 20,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Blending oracle: learn optimal convex weights directly on test ground truth.
    This is the true upper bound for the convex ensemble — same blending approach
    but with no val->test generalization gap.

    Returns the oracle weights dict (same format as learn_convex_weights).
    """
    dfs_test = [results_test[f"{m}_test"] for m in method_names]
    weights = learn_convex_weights(dfs_test, region_col, min_rows)
    results_test[out_name] = apply_convex_weights(dfs_test, weights, region_col)
    return weights


def oracle_ensemble_disagreement(
    method_names: List[str],
    results_test: Dict[str, pd.DataFrame],
    out_name: str,
    n_bins: int = 10,
    min_rows: int = 20,
    verbose: bool = True,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Disagreement-conditioned blending oracle: bins pixels by model disagreement,
    then fits optimal convex weights per bin on test ground truth.

    Disagreement = mean |pred_A - pred_B| across G/M/S/D for each pixel.

    Returns dict mapping bin_index -> {pred_col: weights_array}.
    """
    dfs = [results_test[f"{m}_test"] for m in method_names]
    n_models = len(dfs)
    pred_cols = [c for c in dfs[0].columns if '_pred_' in c]
    uniform = np.full(n_models, 1.0 / n_models)

    # Compute per-pixel disagreement: mean absolute difference across pred columns
    disagreement = np.zeros(len(dfs[0]))
    for col in pred_cols:
        preds = np.column_stack([df[col].values for df in dfs])
        # Max - min across models for this column
        disagreement += np.ptp(preds, axis=1)
    disagreement /= len(pred_cols)  # average across G/M/S/D

    # Assign pixels to quantile bins
    bin_edges = np.percentile(disagreement, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1  # include the max value
    bin_idx = np.digitize(disagreement, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Fit convex weights per bin
    bin_weights = {}
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < min_rows:
            bin_weights[b] = {col: uniform.copy() for col in pred_cols}
            continue
        weights = {}
        for col in pred_cols:
            X = np.column_stack([df.loc[mask, col].values for df in dfs])
            y = dfs[0].loc[mask, col.replace('_pred_', '_truth_')].values
            valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
            if valid.sum() < min_rows:
                weights[col] = uniform.copy()
            else:
                weights[col] = _fit_convex_weights(X[valid], y[valid])
        bin_weights[b] = weights

    # Apply weights per bin
    df_out = dfs[0].copy()
    for col in pred_cols:
        X = np.column_stack([df[col].values for df in dfs])
        blended = np.zeros(len(df_out))
        for b in range(n_bins):
            mask = bin_idx == b
            w = bin_weights[b][col]
            blended[mask] = X[mask] @ w
        df_out[col] = blended

    results_test[out_name] = df_out

    if verbose:
        print(f"Disagreement oracle ({n_bins} bins):")
        print(f"  {'Bin':>4s}  {'Disagree range':>20s}  {'N pixels':>8s}  ", end="")
        print("  ".join(f"w({m[:15]})" for m in method_names))
        for b in range(n_bins):
            mask = bin_idx == b
            lo = disagreement[mask].min() if mask.any() else 0
            hi = disagreement[mask].max() if mask.any() else 0
            n = mask.sum()
            # Average weight across pred columns for display
            avg_w = np.mean([bin_weights[b][col] for col in pred_cols], axis=0)
            w_str = "  ".join(f"{w:.3f}" for w in avg_w)
            print(f"  {b:4d}  {lo:8.1f} - {hi:8.1f}     {n:8d}  {w_str}")

    return bin_weights


def sort_df(df):
	return df.sort_values(by=['years', 'HLStile', 'SiteID', 'row', 'col', 'version']).reset_index(drop=True)

def results_file(split="test", selected_months=[3,6,9,12]):
	dir = get_results_dir(selected_months=selected_months)

	import os 
	results = {}
	for results_file in os.listdir(dir):
		
		if split not in results_file:
			continue
		
		df = pd.read_csv(f"{dir}/{results_file}")
		results[results_file[:-4]] = df

	for key in results.keys(): 
		results[key] = sort_df(results[key])

	results_w_regions = {}
	for key in results.keys(): 
		results_w_regions[key] = add_region_to_results(
			results_df=results[key],
			geo_path=path_config.get_data_geojson(),
			eco_path="useco1/NA_CEC_Eco_Level1.shp",
			region_column="NA_L1NAME"  # or "US_L1NAME"/"US_L3NAME"
		)

	return results, results_w_regions


def get_mae(df, tile, siteid, year, results, method):
    df_tile = df[(df["HLStile"] == tile) & (df["SiteID"] == siteid) & (df["years"] == year)]
    for date in ["G", "M", "S", "D"]:
        mae = np.mean(np.abs(df_tile[f"{date}_pred_DOY"] - df_tile[f"{date}_truth_DOY"]))
        results["HLStile"].append(tile)
        results["Date"].append(date)
        results["MAE"].append(mae)
        results["SiteID"].append(siteid)
        results["years"].append(year)
        results["Method"].append(method)
    return results


def plot_qual(results_w_regions, methods = ["prithvi_pretrained_1.0_test", "Transformer-LSP_test"], methods_plot_names = ["Prithvi", "1D Shallow \n Transformer"], random_seed=120): 
    # ==== CONFIG ====
    title_fontsize = 18
    label_fontsize = 18
    band_fontsize = 18
    model_fontsize = 20
    # ==== LOAD DATA ====
    geo_data = gpd.read_file(path_config.get_data_geojson())
    geo_data["HLStile"] = "T" + geo_data["name"]
    geo_data = geo_data.set_crs("EPSG:4326").to_crs(epsg=3857)
    geo_data = geo_data.rename(columns={"Site_ID": "SiteID"})

    prithvi_pretraining = results_w_regions[methods[0]]
    transformer_lsp = results_w_regions[methods[1]]

    # ==== SAMPLE 4 RANDOM TILE-SITE COMBOS ====
    pairs = prithvi_pretraining[["HLStile", "SiteID"]].drop_duplicates()
    sampled_pairs = pairs.sample(n=4, random_state=random_seed).reset_index(drop=True)

    # ==== HELPER ====
    def reshape_doy(df, hlstile, siteid, band):
        subset = df[(df["HLStile"] == hlstile) & (df["SiteID"] == siteid) & (df["years"] == 2019)]
        image = np.full((33, 33), np.nan)
        for _, row in subset.iterrows():
            r = int(row["row"] // 10)
            c = int(row["col"] // 10)
            image[r, c] = row[f"{band}_pred_DOY"]
        return image

    def reshape_doy_truth(df, hlstile, siteid, band):
        subset = df[(df["HLStile"] == hlstile) & (df["SiteID"] == siteid) & (df["years"] == 2019)]
        image = np.full((33, 33), np.nan)
        for _, row in subset.iterrows():
            r = int(row["row"] // 10)
            c = int(row["col"] // 10)
            image[r, c] = row[f"{band}_truth_DOY"]
        return image

    # Expand band codes
    band_labels = {
        "G": "Growth",
        "M": "Maturity",
        "S": "Senescence",
        "D": "Dormancy"
    }

    # ==== PLOTTING ====
    fig, axs = plt.subplots(2, 2, figsize=(28, 20))

    for idx, (hlstile, siteid) in sampled_pairs.iterrows():
        outer_row = idx // 2
        outer_col = idx % 2
        outer_ax = axs[outer_row, outer_col]

        # print(outer_col)

        # Create 2x4 subgrid inside each 2x2 tile panel
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        gs_sub = GridSpecFromSubplotSpec(4, 4, subplot_spec=outer_ax.get_subplotspec(), 
                                        wspace=0.1, hspace=0.2, height_ratios=[1, 1, 1, 0.05])


        tile_geom = geo_data[(geo_data["HLStile"] == hlstile) & (geo_data["SiteID"] == siteid)]
        if tile_geom.empty:
            print(f"Skipping {hlstile}-{siteid} (no geometry)")
            continue

        # Draw model label once per row (left side)
        outer_ax.axis('off')
        outer_ax.text(0.5, 1.1, f"{hlstile} / {siteid}",
                    transform=outer_ax.transAxes, ha='center', va='bottom',
                    fontsize=title_fontsize, weight='bold')

        # === Loop over bands as before ===
        for i, band in enumerate(["G", "M", "S", "D"]):


            ax_g = fig.add_subplot(gs_sub[0, i])
            img_g = reshape_doy_truth(transformer_lsp, hlstile, siteid, band)
            im = ax_g.imshow(img_g, cmap="viridis", vmin=0, vmax=365)
            ax_g.set_title(band_labels[band], fontsize=band_fontsize)
            ax_g.axis('off')

            # Transformer row
            ax_t = fig.add_subplot(gs_sub[1, i])
            img_t = reshape_doy(transformer_lsp, hlstile, siteid, band)
            im = ax_t.imshow(img_t, cmap="viridis", vmin=0, vmax=365)
            ax_t.set_title(band_labels[band], fontsize=band_fontsize)
            ax_t.axis('off')

            # Prithvi row
            ax_p = fig.add_subplot(gs_sub[2, i])
            img_p = reshape_doy(prithvi_pretraining, hlstile, siteid, band)
            ax_p.imshow(img_p, cmap="viridis", vmin=0, vmax=365)
            ax_p.axis('off')

            if i == 0 and outer_col == 0:
                
                ax_g.text(-2, img_g.shape[0] // 2, "Ground Truth", fontsize=model_fontsize,
                        ha="right", va="center", rotation=90, transform=ax_g.transData, weight='bold')
                ax_t.text(-2, img_t.shape[0] // 2, methods_plot_names[1], fontsize=model_fontsize,
                        ha="right", va="center", rotation=90, transform=ax_t.transData, weight='bold')
                ax_p.text(-2, img_p.shape[0] // 2, methods_plot_names[0], fontsize=model_fontsize,
                        ha="right", va="center", rotation=90, transform=ax_p.transData, weight='bold')

        # === Add per-grid colorbar (bottom row) ===
        cax_sub = fig.add_subplot(gs_sub[3, :])  # Bottom row, span all 4 columns
        cbar = fig.colorbar(im, cax=cax_sub, orientation="horizontal")
        cbar.set_label("Day of Year (DOY)", fontsize=label_fontsize)


    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


def plot_performance_tiles(results_mae_df, methods= ["prithvi_pretrained_1.0_test", "shallow_transformer_pixels_1.0_test"]):

    results_mae_bar = results_mae_df[results_mae_df["Method"].isin(methods)].copy()

    date_to_rename = {
        "G": "Growing",
        "M": "Mature",
        "S": "Senescent",
        "D": "Dormant",
        "Mean": "Mean"
    }


    #create a new column that combines HLStile and SiteID seperated by \n 
    results_mae_bar["Hlstile_SiteID"] = results_mae_bar["HLStile"] + "\n" + results_mae_bar["SiteID"]

    #fig with 4 subplots vertical 
    fig, axs = plt.subplots(4, 1, figsize=(20, 12))
    for idx, date in enumerate(["G", "M", "S", "D"]):
        results_plot_date = results_mae_bar[results_mae_bar["Date"] == date]
        sns.barplot(data=results_plot_date, x="Hlstile_SiteID", y="MAE", hue="Method", palette="Set2", ax=axs[idx])
        # axs[idx].set_title(f"MAE per HLStile for {date} by Method", fontsize=16)
        #make xlabel bold
        axs[idx].set_xlabel(f"Date: {date_to_rename[date]}", fontsize=14, fontweight='bold')
        axs[idx].set_ylabel("MAE (DOY)")
        axs[idx].legend(title="Method", loc="upper right")

    plt.tight_layout()
    plt.show()
