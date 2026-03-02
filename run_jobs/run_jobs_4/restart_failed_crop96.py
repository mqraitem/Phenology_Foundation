"""Restart two failed crop96 multiscale jobs."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_jobs_4.common import setup_records_dir

selected_months = [3, 6, 9, 12]
months_str = "-".join(str(m) for m in selected_months)
months_args = " ".join(str(m) for m in selected_months)
records_dir = setup_records_dir(selected_months)

# ===== Fixed defaults (from prithvi_crops_multiscale_sizes.py) =====
loss = "mse"
n_layers = 4
load_checkpoint = True
grad_accum_steps = 1
data_percentage = 1.0
feature_indices = "5 11 17 23"
wandb_project = f"phenology_crop_{data_percentage}_m{months_str}"

# ===== Crop-96 config =====
crop_size = 96
epoch_length = 1250
epochs = 150
group_name = "prithvi_pretrained_multiscale_crops_conv3d_crop96"
backbone_lr_scale = 1.0

# ===== The two failed jobs =====
failed_jobs = [
    {"learning_rate": 0.001,  "batch_size": 8},
    {"learning_rate": 0.0005, "batch_size": 4},
]

for job in failed_jobs:
    learning_rate = job["learning_rate"]
    batch_size = job["batch_size"]

    name = (f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"
            f"_gradaccum-{grad_accum_steps}_loss-{loss}_n_layers-{n_layers}"
            f"_crop-{crop_size}_epochlen-{epoch_length}"
            f"_feat-5-11-17-23"
            f"_backbone_lr_scale-{backbone_lr_scale}")

    command = (f"qsub -v args='"
               f" --backbone_lr_scale {backbone_lr_scale}"
               f" --crop_size {crop_size}"
               f" --epoch_length {epoch_length}"
               f" --grad_accum_steps {grad_accum_steps}"
               f" --optimizer lamb"
               f" --feature_indices {feature_indices}"
               f" --n_epochs {epochs}"
               f" --selected_months {months_args}"
               f" --n_layers {n_layers}"
               f" --loss {loss}"
               f" --wandb_name {name}"
               f" --wandb_project {wandb_project}"
               f" --feed_timeloc False"
               f" --data_percentage {data_percentage}"
               f" --batch_size {batch_size}"
               f" --group_name {group_name}"
               f" --load_checkpoint {load_checkpoint}"
               f" --logging True"
               f" --learning_rate {learning_rate}'"
               f" -o {records_dir}/{name}"
               f" run_scripts/train_prithvi_conv3d_crops_multiscale.sh")
    print(command)
    os.system(command)
