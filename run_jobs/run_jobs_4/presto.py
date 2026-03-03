"""Submit jobs for Presto with 4 months.

Sweep grid mirrors shallow_transformer_pixels breadth:
- 3 learning rates × 3 batch sizes × 2 freeze settings × 2 optimizers
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_jobs_4.common import is_done, setup_records_dir

selected_months = [3, 6, 9, 12]
months_str = "-".join(str(m) for m in selected_months)
months_args = " ".join(str(m) for m in selected_months)
records_dir = setup_records_dir(selected_months)

# ===== Fixed defaults =====
loss = "mse"
epochs = 35
data_percentage = 1.0

# ===== Sweep grid =====
learning_rates = [1e-05, 5e-05, 1e-04]
batch_sizes = [512, 1024, 10000]
freeze_encoders = [False]
optimizers = ["lamb"]

group_name = "presto_pretrained"
wandb_project = f"phenology_crop_{data_percentage}_m{months_str}"
for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        for freeze_encoder in freeze_encoders:
            for optimizer in optimizers:
                name = (f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"
                        f"_loss-{loss}_freeze-{freeze_encoder}_opt-{optimizer}")

                if is_done(f"{records_dir}/{name}"):
                    continue

                command = (f"qsub -v args='"
                           f" --n_epochs {epochs}"
                           f" --selected_months {months_args}"
                           f" --loss {loss}"
                           f" --wandb_name {name}"
                           f" --wandb_project {wandb_project}"
                           f" --data_percentage {data_percentage}"
                           f" --batch_size {batch_size}"
                           f" --group_name {group_name}"
                           f" --freeze_encoder {freeze_encoder}"
                           f" --optimizer {optimizer}"
                           f" --logging True"
                           f" --learning_rate {learning_rate}'"
                           f" -o {records_dir}/{name}"
                           f" run_scripts/train_presto.sh")
                print(command)
                os.system(command)
