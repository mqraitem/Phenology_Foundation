"""Submit jobs for Prithvi Multi-Scale Conv3d: AdamW vs LAMB optimizer (crop=48)."""
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
epochs = 150
n_layers = 4
load_checkpoint = True
grad_accum_steps = 1
data_percentage = 1.0
crop_size = 48
epoch_length = 5000
feature_indices = "5 11 17 23"

# ===== Sweep grid =====
learning_rates = [0.0001, 0.0005, 0.001]
backbone_lr_scales = [1.0]
batch_sizes = [16, 32]

# ===== Optimizer configs =====
optimizer_configs = {
    "adamw": {"optimizer": "adamw", "group": "prithvi_pretrained_multiscale_crops_conv3d_adamw"},
    # "lamb":  {"optimizer": "lamb",  "group": "prithvi_pretrained_multiscale_crops_conv3d_lamb"},
}

for opt_name, opt_cfg in optimizer_configs.items():
    optimizer = opt_cfg["optimizer"]
    group_name = opt_cfg["group"]

    for learning_rate in learning_rates:
        for backbone_lr_scale in backbone_lr_scales:
            for batch_size in batch_sizes:
                name = (f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"
                        f"_gradaccum-{grad_accum_steps}_loss-{loss}_n_layers-{n_layers}"
                        f"_crop-{crop_size}_epochlen-{epoch_length}"
                        f"_feat-5-11-17-23"
                        f"_backbone_lr_scale-{backbone_lr_scale}")

                if os.path.exists(f"{records_dir}/{name}"):
                    continue

                if is_done(f"{records_dir}/{name}"):
                    continue

                command = (f"qsub -v args='"
                           f" --backbone_lr_scale {backbone_lr_scale}"
                           f" --crop_size {crop_size}"
                           f" --epoch_length {epoch_length}"
                           f" --grad_accum_steps {grad_accum_steps}"
                           f" --optimizer {optimizer}"
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
