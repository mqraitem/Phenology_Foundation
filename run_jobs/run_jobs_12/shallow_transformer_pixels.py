"""Submit jobs for Shallow Transformer (pixels) with 12 months.

Best 4-month params: lr=5e-05, batch_size=512, loss=mse
Per-pixel model, so memory is unaffected by more timesteps.
Keep same hyperparameters; only change selected_months.
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
epochs = 150
data_percentage = 1.0

# ===== Best 4-month params + small sweep =====
learning_rates = [5e-05]
batch_sizes = [512]

wandb_project = f"phenology_crop_{data_percentage}_m{months_str}"
group_name = "shallow_transformer_pixels"
for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        name = (f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"
                f"_loss-{loss}")

        if is_done(f"{records_dir}/{name}"):
            continue

        command = (f"qsub -v args='"
                   f" --n_epochs {epochs}"
                   f" --selected_months {months_args}"
                   f" --loss {loss}"
                   f" --wandb_name {name}"
                   f" --data_percentage {data_percentage}"
                   f" --batch_size {batch_size}"
                   f" --group_name {group_name}"
                   f" --wandb_project {wandb_project}"
                   f" --logging True"
                   f" --learning_rate {learning_rate}'"
                   f" -o {records_dir}/{name}"
                   f" run_scripts/train_lsp_pixels.sh")
        print(command)
        os.system(command)
