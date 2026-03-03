# Crop Phenology Prediction with Foundation Models

Predicting crop phenology dates (day-of-year) from multi-temporal satellite imagery using [Prithvi EO V2](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) vision transformer backbones with Conv3D temporal fusion heads, and [Presto](https://github.com/nasaharvest/presto) pretrained pixel-level transformer.

## Task

Given monthly satellite composites — either Harmonized Landsat-Sentinel (HLS) at 30m or Sentinel-2 (S2) at 10m (downsampled to 30m) — with 6 spectral bands, predict 4 phenological dates per pixel:

| Output | Description |
|--------|-------------|
| **G** | Greenup / Germination |
| **M** | Maturity |
| **S** | Senescence / Silking |
| **D** | Dormancy / Dough |

Ground truth comes from the High Plains Land Surface Phenology (HP-LSP) dataset. Predictions are evaluated on 330x330 pixel tiles using Mean Absolute Error (MAE) in days.

## Model Architectures

### Prithvi Conv3D MultiScale
Extracts features from 4 intermediate ViT blocks (layers 5, 11, 17, 23) → per-level LayerNorm + linear projection → concatenation → same upscaler + fusion head.

**Script:** `train_prithvi_conv3d_crops_multiscale.py`

### Presto
[Presto](https://github.com/nasaharvest/presto) pretrained pixel-level transformer encoder with a linear regression head. Uses Sentinel-2 composites (6 bands: B2, B3, B4, B8A, B11, B12) with cloud masking (median cloud score < 3000). S2 data is 3x3 block-averaged from 10m to 30m to match GT resolution. Supports frozen or fine-tuned encoder.

**Script:** `train_presto.py`

### Baselines
- **Shallow Temporal Transformer** — lightweight per-pixel transformer on raw spectral time series (`train_lsp_pixels.py`, `train_lsp_patch.py`)
- **Random Baseline** — samples DOY from training set distribution (`generate_random_baseline.py`)

## Setup

### Prerequisites
- Python 3.10+
- PyTorch with CUDA
- Key dependencies: `timm`, `einops`, `rasterio`, `wandb`, `geopandas`, `scipy`

### Configuration
Edit `dirs.txt` to set data and model weight paths:

```
DATA_HLS_COMPOSITES=/path/to/HLS_composites
DATA_S2_COMPOSITES=/path/to/S2_composites
DATA_LSP_ANCILLARY=/path/to/LSP_ancillary
DATA_GEOJSON=/path/to/geotiff_extents.geojson
MODEL_WEIGHTS_300M=/path/to/Prithvi_EO_V2_300M.pt
MODEL_WEIGHTS_600M=/path/to/Prithvi_EO_V2_600M.pt
```

### Presto Submodule
The `presto/` directory is a git submodule. After cloning, initialize it:

```bash
git submodule update --init --recursive
```

Presto requires a separate conda environment (`geo_presto`) with its dependencies (see `presto/` for setup).

## Training

### Single run

```bash
python train_prithvi_conv3d_crops.py \
    --load_checkpoint True \
    --model_size 300m \
    --learning_rate 0.0001 \
    --batch_size 16 \
    --n_epochs 150 \
    --crop_size 48 \
    --epoch_length 5000 \
    --n_layers 4 \
    --loss mse \
    --optimizer adamw \
    --backbone_lr_scale 1.0 \
    --layer_decay 0.75 \
    --warmup_epochs 5 \
    --selected_months 3 6 9 12 \
    --logging True \
    --group_name my_experiment \
    --wandb_name run_1
```

### Multi-scale variant

```bash
python train_prithvi_conv3d_crops_multiscale.py \
    --load_checkpoint True \
    --feature_indices 5 11 17 23 \
    --learning_rate 0.0001 \
    --batch_size 16 \
    --crop_size 48 \
    --n_layers 4
```

### CatHLS variant

```bash
python train_prithvi_conv3d_crops_cathls.py \
    --load_checkpoint True \
    --hls_out_channels 64 \
    --hls_n_layers 4 \
    --progressive_fusion True \
    --optimizer lamb \
    --backbone_lr_scale 1.0
```

### Presto

```bash
python train_presto.py \
    --learning_rate 5e-05 \
    --batch_size 1024 \
    --n_epochs 150 \
    --loss mse \
    --optimizer lamb \
    --freeze_encoder False \
    --warmup_epochs 5 \
    --selected_months 3 6 9 12 \
    --logging True \
    --group_name presto_pretrained \
    --wandb_name run_1
```

### Batch job submission (SGE)

```bash
# Submit hyperparameter sweep
python run_jobs.py

# Submit baseline experiments
python run_prior_work.py
```

## Evaluation

### Select best checkpoints
```bash
python select_best_params.py
```

### Export per-pixel results to CSV
```bash
python eval_to_dataframe.py
```

### Generate random baseline
```bash
python generate_random_baseline.py --selected_months 3 6 9 12
```

### Analysis notebook
Open `results_overview_notebook.ipynb` for method comparison, regional analysis, ensemble methods, and qualitative visualizations.

## Key Training Details

- **LR schedule (Prithvi/Presto):** Linear warmup (5 epochs) → cosine annealing
- **LR schedule (Shallow Transformer):** Cosine annealing (no warmup)
- **Layer-wise LR decay:** Deeper backbone layers get geometrically smaller LR (factor 0.75 per layer) — Prithvi only
- **Loss:** MSE on normalized DOY values (divided by 547)
- **Evaluation:** Sliding-window crops over full 330x330 tiles, averaging overlapping predictions (Prithvi); full-tile pixel-wise (Presto)
- **Cloud masking (S2):** Pixels with median cloud score < 3000 are zeroed out after 3x3 block averaging
- **Logging:** Weights & Biases with per-epoch train/val/test metrics
