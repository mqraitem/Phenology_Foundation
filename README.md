# Crop Phenology Prediction with Prithvi Foundation Models

Predicting crop phenology dates (day-of-year) from multi-temporal satellite imagery using [Prithvi EO V2](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) vision transformer backbones with Conv3D temporal fusion heads.

## Task

Given 4 monthly Harmonized Landsat-Sentinel (HLS) composites (March, June, September, December) with 6 spectral bands (B02–B07) at 30m resolution, predict 4 phenological dates per pixel:

| Output | Description |
|--------|-------------|
| **G** | Greenup / Germination |
| **M** | Maturity |
| **S** | Senescence / Silking |
| **D** | Dormancy / Dough |

Ground truth comes from the High Plains Land Surface Phenology (HP-LSP) dataset. Predictions are evaluated on 330x330 pixel tiles using Mean Absolute Error (MAE) in days.

## Model Architectures

### Prithvi Conv3D (standard)
Prithvi ViT backbone → 3D reshape → 4-stage ConvTranspose3D upscaler (16x spatial) → Conv3D temporal fusion head.

**Script:** `train_prithvi_conv3d_crops.py`

### Prithvi Conv3D MultiScale
Extracts features from 4 intermediate ViT blocks (layers 5, 11, 17, 23) → per-level LayerNorm + linear projection → concatenation → same upscaler + fusion head.

**Script:** `train_prithvi_conv3d_crops_multiscale.py`

### Prithvi Conv3D CatHLS
Two-branch architecture: Prithvi backbone path + parallel HLS temporal branch (temporal-only Conv3D at full resolution). Branches are fused via FiLM (Feature-wise Linear Modulation), optionally at each upscaler stage (progressive fusion).

**Script:** `train_prithvi_conv3d_crops_cathls.py`

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
DATA_LSP_ANCILLARY=/path/to/LSP_ancillary
DATA_GEOJSON=/path/to/geotiff_extents.geojson
MODEL_WEIGHTS_300M=/path/to/Prithvi_EO_V2_300M.pt
MODEL_WEIGHTS_600M=/path/to/Prithvi_EO_V2_600M.pt
```

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

- **LR schedule:** Linear warmup (5 epochs) → cosine annealing
- **Layer-wise LR decay:** Deeper backbone layers get geometrically smaller LR (factor 0.75 per layer)
- **Loss:** MSE on normalized DOY values (divided by 547)
- **Evaluation:** Sliding-window crops over full 330x330 tiles, averaging overlapping predictions
- **Logging:** Weights & Biases with per-epoch train/val/test metrics
