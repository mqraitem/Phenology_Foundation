
import os

# Limit threading libraries BEFORE importing torch/numpy
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["GDAL_NUM_THREADS"] = "4"
os.environ["GDAL_CACHEMAX"] = "512"

import torch
from torch.utils.data import DataLoader
import numpy as np

import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from lib.models.presto_phenology import PrestoPhenologyModel
from lib.utils import (
	segmentation_loss_pixels, segmentation_loss_pixels_mae,
	segmentation_loss, segmentation_loss_mae,
	eval_data_loader_presto, get_masks_paper,
	print_trainable_parameters, save_checkpoint, str2bool,
	months_to_str, get_checkpoint_dir, get_data_paths_s2,
)
from lib.dataloaders.dataloaders_pixels_s2 import CycleDatasetPixelsS2
from lib.dataloaders.dataloaders_s2 import CycleDatasetS2
from arg_configs import get_core_parser

#######################################################################################

def main():

	parser = get_core_parser()
	parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae"],
	                   help="Loss function: mse or mae")
	parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "lamb"],
	                   help="Optimizer: adamw or lamb")
	parser.add_argument("--freeze_encoder", type=str2bool, default=False,
	                   help="Freeze Presto encoder (train head only)")
	parser.add_argument("--warmup_epochs", type=int, default=5,
	                   help="Number of linear warmup epochs")
	parser.add_argument("--min_lr", type=float, default=1e-7,
	                   help="Minimum learning rate for cosine annealing")
	args = parser.parse_args()

	months_str = months_to_str(args.selected_months)
	file_suffix = f"_m{months_str}"
	n_timesteps = len(args.selected_months)

	# 0-indexed months for Presto (expects 0-11)
	selected_months_0idx = [m - 1 for m in args.selected_months]
	month_tensor = torch.tensor(selected_months_0idx, dtype=torch.long)

	wandb_config = {
		"learningrate": args.learning_rate,
		"batch_size": args.batch_size,
		"data_percentage": args.data_percentage,
		"loss": args.loss,
		"optimizer": args.optimizer,
		"selected_months": args.selected_months,
		"freeze_encoder": args.freeze_encoder,
		"warmup_epochs": args.warmup_epochs,
		"min_lr": args.min_lr,
	}

	wandb_name = args.wandb_name
	group_name = args.group_name

	if args.logging:
		wandb.init(
			project=args.wandb_project or f"phenology_crop_{args.data_percentage}",
			group=group_name,
			config=wandb_config,
			name=wandb_name,
		)
		wandb.run.log_code(".")

	path_train = get_data_paths_s2("training", args.data_percentage, args.selected_months)
	path_val = get_data_paths_s2("validation", args.data_percentage, args.selected_months)
	path_test = get_data_paths_s2("testing", args.data_percentage, args.selected_months)

	# Build tile datasets first to get all_locations for pixel dataloader
	cycle_dataset_val = CycleDatasetS2(path_val, split="validation", data_percentage=args.data_percentage, n_timesteps=n_timesteps, file_suffix=file_suffix)
	cycle_dataset_test = CycleDatasetS2(path_test, split="testing", data_percentage=args.data_percentage, n_timesteps=n_timesteps, file_suffix=file_suffix)

	# Merge all_locations from val+test+train tile datasets
	all_locations = {}
	all_locations.update(cycle_dataset_val.all_locations)
	all_locations.update(cycle_dataset_test.all_locations)

	# Also build a temporary dataset for train tiles to get their locations
	temp_train_dataset = CycleDatasetS2(path_train, split="training", data_percentage=args.data_percentage, n_timesteps=n_timesteps, file_suffix=file_suffix)
	all_locations.update(temp_train_dataset.all_locations)
	del temp_train_dataset

	cycle_dataset_train = CycleDatasetPixelsS2(
		path_train, split="training", all_locations=all_locations,
		data_percentage=args.data_percentage, n_timesteps=n_timesteps,
		file_suffix=file_suffix,
	)

	train_dataloader = DataLoader(cycle_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
	val_dataloader = DataLoader(cycle_dataset_val, batch_size=1, shuffle=False, num_workers=2)
	test_dataloader = DataLoader(cycle_dataset_test, batch_size=1, shuffle=False, num_workers=2)

	device = "cuda"

	model = PrestoPhenologyModel(num_classes=4, freeze_encoder=args.freeze_encoder)
	print_trainable_parameters(model)
	model = model.to(device)

	checkpoint_dir = get_checkpoint_dir(group_name, args.data_percentage, args.selected_months)
	checkpoint = f"{checkpoint_dir}/{wandb_name}.pth"

	trainable_params = filter(lambda p: p.requires_grad, model.parameters())

	if args.optimizer == "lamb":
		from torch_optimizer import Lamb
		optimizer = Lamb(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
	else:
		optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
	print(f"Using optimizer: {args.optimizer}")

	# Cosine annealing with linear warmup (matching Prithvi training)
	warmup_epochs = min(args.warmup_epochs, args.n_epochs)
	cosine_epochs = max(1, args.n_epochs - warmup_epochs)
	warmup_scheduler = LinearLR(
		optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
	)
	cosine_scheduler = CosineAnnealingLR(
		optimizer, T_max=cosine_epochs, eta_min=args.min_lr,
	)
	scheduler = SequentialLR(
		optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs],
	)

	train_loss_fn = segmentation_loss_pixels_mae if args.loss == "mae" else segmentation_loss_pixels
	eval_loss_fn = segmentation_loss_mae if args.loss == "mae" else segmentation_loss
	print(f"Using loss function: {args.loss}")

	best_acc_val = 100
	for epoch in range(args.n_epochs):

		loss_i = 0.0

		print("iteration started")
		model.train()

		for j, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

			input = batch_data["image"]
			mask = batch_data["gt_mask"]
			latlons = batch_data["latlons"]

			mask = mask.to(device)

			optimizer.zero_grad()
			out = model(input, processing_images=False, latlons=latlons.to(device), month=month_tensor)

			loss = train_loss_fn(mask, out, device=device)
			loss_i += loss.item() * input.size(0)

			loss.backward()
			optimizer.step()

			if j % 500 == 0:
				to_print = f"Epoch: {epoch}, iteration: {j}, loss: {loss.item()} \n "
				print(to_print)

		epoch_loss_train = loss_i / len(train_dataloader.dataset)

		# Validation Phase
		acc_dataset_val, _, epoch_loss_val = eval_data_loader_presto(
			val_dataloader, model, device, get_masks_paper("train"),
			month=month_tensor, loss_fn=eval_loss_fn,
		)
		acc_dataset_test, _, epoch_loss_test = eval_data_loader_presto(
			test_dataloader, model, device, get_masks_paper("test"),
			month=month_tensor, loss_fn=eval_loss_fn,
		)

		if args.logging:
			to_log = {}
			to_log["epoch"] = epoch + 1
			to_log["val_loss"] = epoch_loss_val
			to_log["test_loss"] = epoch_loss_test
			to_log["train_loss"] = epoch_loss_train
			to_log["learning_rate"] = optimizer.param_groups[0]['lr']
			for idx in range(4):
				to_log[f"acc_val_{idx}"] = acc_dataset_val[idx]
				to_log[f"acc_test_{idx}"] = acc_dataset_test[idx]
			wandb.log(to_log)

		print("=" * 100)
		to_print = f"Epoch: {epoch}, val_loss: {epoch_loss_val} \n "
		for idx in range(4):
			to_print += f"acc_val_{idx}: {acc_dataset_val[idx]} \n "
		for idx in range(4):
			to_print += f"acc_test_{idx}: {acc_dataset_test[idx]} \n "
		print(to_print)
		print("=" * 100)

		scheduler.step()
		acc_dataset_val_mean = np.mean(list(acc_dataset_val.values()))

		if acc_dataset_val_mean < best_acc_val:
			save_checkpoint(model, optimizer, epoch, epoch_loss_train, epoch_loss_val, checkpoint, selected_months=args.selected_months)
			best_acc_val = acc_dataset_val_mean

	model.load_state_dict(torch.load(checkpoint)["model_state_dict"])

	acc_dataset_val, _, epoch_loss_val = eval_data_loader_presto(
		val_dataloader, model, device, get_masks_paper("train"),
		month=month_tensor, loss_fn=eval_loss_fn,
	)
	acc_dataset_test, _, _ = eval_data_loader_presto(
		test_dataloader, model, device, get_masks_paper("test"),
		month=month_tensor, loss_fn=eval_loss_fn,
	)

	if args.logging:
		for idx in range(4):
			wandb.run.summary[f"best_acc_val_{idx}"] = acc_dataset_val[idx]
			wandb.run.summary[f"best_acc_test_{idx}"] = acc_dataset_test[idx]
		wandb.run.summary[f"best_avg_acc_val"] = np.mean(list(acc_dataset_val.values()))
		wandb.run.summary[f"best_avg_acc_test"] = np.mean(list(acc_dataset_test.values()))

	if args.logging:
		wandb.finish()


if __name__ == "__main__":
	main()
