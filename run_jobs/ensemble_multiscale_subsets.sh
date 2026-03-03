#!/bin/bash
# Progressively ensemble multiscale crop-size models: 32+48, then +96, +144, +224.
set -e

SELECTED_MONTHS="3 4 5 6 7 8 9 10"

CROP32="prithvi_pretrained_multiscale_crops_conv3d_crop32_1.0"
CROP48="prithvi_pretrained_multiscale_crops_conv3d_crop48_1.0"
CROP96="prithvi_pretrained_multiscale_crops_conv3d_crop96_1.0"
CROP144="prithvi_pretrained_multiscale_crops_conv3d_crop144_1.0"
CROP224="prithvi_pretrained_multiscale_crops_conv3d_crop224_1.0"

python ensemble_from_csvs.py \
    --methods $CROP32 $CROP48 \
    --selected_months $SELECTED_MONTHS \
    --name ensemble_multiscale_crop32_crop48

python ensemble_from_csvs.py \
    --methods $CROP32 $CROP48 $CROP96 \
    --selected_months $SELECTED_MONTHS \
    --name ensemble_multiscale_crop32_crop48_crop96

python ensemble_from_csvs.py \
    --methods $CROP32 $CROP48 $CROP96 $CROP144 \
    --selected_months $SELECTED_MONTHS \
    --name ensemble_multiscale_crop32_crop48_crop96_crop144

python ensemble_from_csvs.py \
    --methods $CROP32 $CROP48 $CROP96 $CROP144 $CROP224 \
    --selected_months $SELECTED_MONTHS \
    --name ensemble_multiscale_crop32_crop48_crop96_crop144_crop224
