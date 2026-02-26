"""Shared utilities for run_jobs scripts."""
import os


def is_done(record_path):
    """Check if a job already ran to completion."""
    if not os.path.exists(record_path):
        return False
    file_content = open(record_path, "r", encoding='latin-1').readlines()
    if not file_content:
        return False
    return "wandb: Find logs" in file_content[-1]


def setup_records_dir(selected_months):
    """Create and return the records directory for the given months."""
    months_str = "-".join(str(m) for m in selected_months)
    records_dir = f"records/m{months_str}"
    os.makedirs(records_dir, exist_ok=True)
    return records_dir
