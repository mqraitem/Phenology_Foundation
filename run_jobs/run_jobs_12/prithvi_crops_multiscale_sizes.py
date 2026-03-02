"""Submit jobs for Prithvi Multi-Scale Conv3d at multiple crop sizes with 12 months.

Best 4-month params per crop size (from best_params.csv):
  crop32:  lr=0.001, batch=72,  epoch_len=11250, epochs=150
  crop96:  lr=0.001, batch=4,   epoch_len=1250,  epochs=150
  crop144: lr=0.001, batch=4,   epoch_len=556,   epochs=168
  crop224: lr=0.0005, batch=1,  epoch_len=230,   epochs=204

12-month adjustment (T: 4->12, ~3x memory):
  crop32:  batch 72->24
  crop96:  batch 4->1, use grad_accum=4 to preserve effective batch
  crop144: batch 4->1, use grad_accum=4 to preserve effective batch
  crop224: batch 1->1, use grad_accum=3 (was already at minimum)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_jobs_12.common import is_done, setup_records_dir

selected_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
months_str = "-".join(str(m) for m in selected_months)
months_args = " ".join(str(m) for m in selected_months)
records_dir = setup_records_dir(selected_months)

# ===== Fixed defaults =====
loss = "mse"
n_layers = 4
load_checkpoint = True
data_percentage = 1.0
feature_indices = "5 11 17 23"

# ===== Crop size configs (best 4-month params, batch adjusted for 3x memory) =====
crop_configs = {
    32:  {"learning_rate": 0.001,  "batch_sizes": [36, 72], "grad_accum_steps": 1, "epoch_length": 11250, "epochs": 150},
    48:  {"learning_rate": 0.001,  "batch_sizes": [8, 16],  "grad_accum_steps": 1, "epoch_length": 5000,  "epochs": 150},
    96:  {"learning_rate": 0.001,  "batch_sizes": [4, 8],   "grad_accum_steps": 1, "epoch_length": 1250,  "epochs": 150},
    144: {"learning_rate": 0.001,  "batch_sizes": [2, 4],   "grad_accum_steps": 1, "epoch_length": 556,   "epochs": 168},
    224: {"learning_rate": 0.0005, "batch_sizes": [1, 2],   "grad_accum_steps": 1, "epoch_length": 230,   "epochs": 204},
}

backbone_lr_scale = 1.0
wandb_project = f"phenology_crop_{data_percentage}_m{months_str}"
base_group = "prithvi_pretrained_multiscale_crops_conv3d"

for crop_size, cfg in crop_configs.items():
    learning_rate = cfg["learning_rate"]
    batch_sizes = cfg["batch_sizes"]
    grad_accum_steps = cfg["grad_accum_steps"]
    epoch_length = cfg["epoch_length"]
    epochs = cfg["epochs"]
    group_name = f"{base_group}_crop{crop_size}"

    for batch_size in batch_sizes:
        name = (f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"
                f"_gradaccum-{grad_accum_steps}_loss-{loss}_n_layers-{n_layers}"
                f"_crop-{crop_size}_epochlen-{epoch_length}"
                f"_feat-5-11-17-23"
                f"_backbone_lr_scale-{backbone_lr_scale}")

        if is_done(f"{records_dir}/{name}"):
            continue

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
                   f" --feed_timeloc False"
                   f" --data_percentage {data_percentage}"
                   f" --batch_size {batch_size}"
                   f" --group_name {group_name}"
                   f" --load_checkpoint {load_checkpoint}"
                   f" --wandb_project {wandb_project}"
                   f" --logging True"
                   f" --learning_rate {learning_rate}'"
                   f" -o {records_dir}/{name}"
                   f" run_scripts/train_prithvi_conv3d_crops_multiscale.sh")
        print(command)
        os.system(command)
