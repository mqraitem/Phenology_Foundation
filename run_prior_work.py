import os

selected_months = [3, 6, 9, 12]
months_str = "-".join(str(m) for m in selected_months)
months_args = " ".join(str(m) for m in selected_months)

records_dir = f"records/m{months_str}"
os.makedirs(records_dir, exist_ok=True)

# ===== Shared defaults =====
loss = "mse"
epochs = 150
data_percentage = 1.0

# ===== Sweep grid =====
learning_rates = [0.0001, 0.00005, 0.00001]
batch_sizes = [512, 1024]


def is_done(record_path):
    """Check if a job already ran to completion."""
    if not os.path.exists(record_path):
        return False
    file_content = open(record_path, "r", encoding='latin-1').readlines()
    if not file_content:
        return False
    return "wandb: Find logs" in file_content[-1]


###############################################################
# * Shallow Transformer (pixels)
###############################################################
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
                   f" --logging True"
                   f" --learning_rate {learning_rate}'"
                   f" -o {records_dir}/{name}"
                   f" run_scripts/train_lsp_pixels.sh")
        os.system(command)
