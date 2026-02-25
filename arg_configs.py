"""
Core argument parsing configuration for phenology training scripts.

This module provides core arguments shared across all training scripts.
Each training script should add its own model-specific arguments.
"""

import argparse
from lib.utils import str2bool


def get_core_parser():
    """
    Create an argument parser with core arguments shared by all training scripts.

    Returns:
        argparse.ArgumentParser: Parser with core arguments added

    Example:
        >>> parser = get_core_parser()
        >>> # Add model-specific arguments
        >>> parser.add_argument("--freeze", type=str2bool, default=False)
        >>> args = parser.parse_args()
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for the model")
    parser.add_argument("--logging", type=str2bool, default=False,
                       help="Whether to log the results to wandb")
    parser.add_argument("--group_name", type=str, default="default",
                       help="Group name for wandb logging")
    parser.add_argument("--wandb_name", type=str, default="default",
                       help="Group name for wandb logging")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--data_percentage", type=float, default=1.0,
                       help="Fraction of data to use (0.0-1.0)")
    parser.add_argument("--use_config_normalization", type=str2bool, default=False,
                       help="Use mean/std from config file instead of computing from dataset")
    parser.add_argument("--n_epochs", type=int, default=120,
                       help="Number of training epochs")
    parser.add_argument("--selected_months", type=int, nargs='+',
                       default=[3,6,9,12],
                       help="Which months to include (e.g., --selected_months 3 6 9 12)")

    return parser
