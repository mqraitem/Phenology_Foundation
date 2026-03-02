#!/bin/bash
# Ensemble: shallow transformer + multiscale crop-48 model.
# Outputs: results/m{MONTHS}/ensemble_multiscale_crop48_shallow_transformer_{val,test}.csv
set -e

SELECTED_MONTHS="${SELECTED_MONTHS:-3 6 9 12}"

MULTISCALE48="prithvi_pretrained_multiscale_crops_conv3d_crop48_1.0"
SHALLOW="shallow_transformer_pixels_1.0"

python ensemble_from_csvs.py \
    --methods $MULTISCALE48 $SHALLOW \
    --selected_months $SELECTED_MONTHS \
    --name ensemble_multiscale_crop48_shallow_transformer
