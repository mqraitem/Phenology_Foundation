"""Submit jobs for Prithvi Multi-Scale Conv3d at multiple crop sizes.

Matched data budget: epoch_length scaled so pixels/epoch is constant
across crop sizes (~11.5M pixels/epoch, same as crop=48 baseline).
Epochs scaled to match total gradient updates to the crop=48 baseline.

Baseline (crop=48): epoch_length=5000, batch_sizes=[16, 32], epochs=150
  -> pixels/epoch = 5000 * 48^2 = 11,520,000
  -> grad_updates/epoch = 5000/16=312 or 5000/32=156
  -> total grad_updates = 46,875 or 23,438

crop=96:  epoch_length=1250,  batch=[4, 8],   epochs=150  -> matched pixels & grad updates
crop=144: epoch_length=556,   batch=[2, 4],   epochs=168  -> matched pixels & grad updates
crop=224: epoch_length=230,   batch=[1, 2],   epochs=204  -> matched pixels & grad updates

Each crop size gets its own group folder (e.g. prithvi_..._crop96_1.0)
so select_best_params picks the best hyperparams per crop size independently.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_jobs.common import is_done, setup_records_dir

selected_months = [3, 6, 9, 12]
months_str = "-".join(str(m) for m in selected_months)
months_args = " ".join(str(m) for m in selected_months)
records_dir = setup_records_dir(selected_months)

# ===== Fixed defaults =====
loss = "mse"
n_layers = 4
load_checkpoint = True
grad_accum_steps = 1
data_percentage = 1.0
feature_indices = "5 11 17 23"

# ===== Sweep grid =====
learning_rates = [0.0001, 0.0005, 0.001]
backbone_lr_scales = [1.0]

# ===== Crop size configs =====
# epoch_length: scaled to match ~11.5M pixels/epoch
# batch_sizes: scaled down proportional to crop area increase
# epochs: scaled to match total gradient updates to crop=48 baseline
crop_configs = {
    32:  {"epoch_length": 11250, "batch_sizes": [36, 72], "epochs": 150},
    # 96:  {"epoch_length": 1250, "batch_sizes": [4, 8],  "epochs": 150},
    # 144: {"epoch_length": 556,  "batch_sizes": [2, 4],  "epochs": 168},
    # 224: {"epoch_length": 230,  "batch_sizes": [1, 2],  "epochs": 204},
}

base_group = "prithvi_pretrained_multiscale_crops_conv3d"
for crop_size, cfg in crop_configs.items():
    epoch_length = cfg["epoch_length"]
    batch_sizes = cfg["batch_sizes"]
    epochs = cfg["epochs"]
    group_name = f"{base_group}_crop{crop_size}"

    for learning_rate in learning_rates:
        for backbone_lr_scale in backbone_lr_scales:
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
                           f" --logging True"
                           f" --learning_rate {learning_rate}'"
                           f" -o {records_dir}/{name}"
                           f" run_scripts/train_prithvi_conv3d_crops_multiscale.sh")
                # print(command)
                os.system(command)
