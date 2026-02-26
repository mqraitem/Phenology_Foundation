
import os

# Limit threading libraries BEFORE importing torch/numpy
os.environ["OMP_NUM_THREADS"] = "4"  # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "4"  # Intel MKL threads
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # OpenBLAS threads
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # vecLib threads
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # NumExpr threads
os.environ["GDAL_NUM_THREADS"] = "4"
os.environ["GDAL_CACHEMAX"] = "512"  # Limit cache to 512MB

import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml

import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import path_config

from lib.models.prithvi_conv3d_cathls import PrithviSegConv3DCatHLS
from lib.utils import (
	segmentation_loss, segmentation_loss_mae,
	eval_data_loader_crops, get_masks_paper, save_checkpoint,
	str2bool, months_to_str, get_checkpoint_dir,
	get_data_paths, print_trainable_parameters, get_layer_lr_groups,
)
from lib.dataloaders.dataloaders_crops import CycleDatasetCrops
from lib.dataloaders.dataloaders import CycleDataset
from arg_configs import get_core_parser

#######################################################################################

def main():

	parser = get_core_parser()

	# Prithvi-specific arguments
	parser.add_argument("--model_size", type=str, default="300m",
	                   help="Model size to use (300m or 600m)")
	parser.add_argument("--load_checkpoint", type=str2bool, default=False,
	                   help="Whether to load pretrained checkpoint")
	parser.add_argument("--feed_timeloc", type=str2bool, default=False,
	                   help="Whether to feed time/loc coords")
	parser.add_argument("--n_layers", type=int, default=4,
	                   help="Number of Conv3d layers in temporal fusion head")
	parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae"],
	                   help="Loss function: mse or mae")
	parser.add_argument("--backbone_lr_scale", type=float, default=1.0,
	                   help="Backbone peak LR as a fraction of head LR")
	parser.add_argument("--load_finetuned", type=str2bool, default=False,
	                   help="Load domain-adapted MAE checkpoint")

	# Crop-specific arguments
	parser.add_argument("--crop_size", type=int, default=48,
	                   help="Spatial size of random crops (must be multiple of 16)")
	parser.add_argument("--epoch_length", type=int, default=5000,
	                   help="Number of crops per epoch")
	parser.add_argument("--optimizer", type=str, default="lamb", choices=["adamw", "lamb"],
	                   help="Optimizer: adamw or lamb")

	# Gradient accumulation
	parser.add_argument("--grad_accum_steps", type=int, default=1,
	                   help="Gradient accumulation steps. Effective batch = batch_size * this")

	# Layer-wise LR decay and cosine schedule
	parser.add_argument("--layer_decay", type=float, default=0.75,
	                   help="Layer-wise LR decay factor (0-1)")
	parser.add_argument("--warmup_epochs", type=int, default=5,
	                   help="Number of linear warmup epochs")
	parser.add_argument("--min_lr", type=float, default=1e-7,
	                   help="Minimum LR for cosine schedule")

	# CatHLS-specific arguments
	parser.add_argument("--hls_out_channels", type=int, default=128,
	                   help="Output channels of the HLS branch")
	parser.add_argument("--hls_n_layers", type=int, default=4,
	                   help="Number of Conv3d blocks in HLS branch")
	parser.add_argument("--progressive_fusion", type=str2bool, default=False,
	                   help="FiLM at each upscaler stage (True) or end only (False)")

	args = parser.parse_args()

	months_str = months_to_str(args.selected_months)
	file_suffix = f"_m{months_str}"
	n_timesteps = len(args.selected_months)

	# --- Config ---
	with open(f'configs/prithvi_{args.model_size}.yaml', 'r') as f:
		config = yaml.safe_load(f)

	# img_size = crop_size ALWAYS (train and eval both use crop_size x crop_size)
	config["pretrained_cfg"]["img_size"] = args.crop_size
	config["pretrained_cfg"]["num_frames"] = n_timesteps
	config["training"]["n_iteration"] = args.n_epochs
	config["training"]["batch_size"] = args.batch_size
	config["validation"]["batch_size"] = 1
	config["test"]["batch_size"] = 1

	group_name = args.group_name
	wandb_name = args.wandb_name

	wandb_config = {
		"learningrate": args.learning_rate,
		"model_size": args.model_size,
		"load_checkpoint": args.load_checkpoint,
		"batch_size": args.batch_size,
		"data_percentage": args.data_percentage,
		"n_layers": args.n_layers,
		"loss": args.loss,
		"backbone_lr_scale": args.backbone_lr_scale,
		"selected_months": args.selected_months,
		"crop_size": args.crop_size,
		"epoch_length": args.epoch_length,
		"optimizer": args.optimizer,
		"grad_accum_steps": args.grad_accum_steps,
		"effective_batch_size": args.batch_size * args.grad_accum_steps,
		"layer_decay": args.layer_decay,
		"warmup_epochs": args.warmup_epochs,
		"min_lr": args.min_lr,
		"hls_out_channels": args.hls_out_channels,
		"hls_n_layers": args.hls_n_layers,
		"progressive_fusion": args.progressive_fusion,
	}

	if args.logging:
		wandb.init(
				project=f"phenology_crop_{args.data_percentage}",
				group=group_name,
				config=wandb_config,
				name=wandb_name,
				)
		wandb.run.log_code(".")

	# --- Data ---
	path_train = get_data_paths("training", args.data_percentage, args.selected_months)
	path_val = get_data_paths("validation", args.data_percentage, args.selected_months)
	path_test = get_data_paths("testing", args.data_percentage, args.selected_months)

	print(f"Train: {len(path_train)}, Val: {len(path_val)}, Test: {len(path_test)}")

	# Crop dataset for training
	cycle_dataset_train = CycleDatasetCrops(
		path_train, split="training",
		crop_size=args.crop_size,
		data_percentage=args.data_percentage,
		n_timesteps=n_timesteps,
		file_suffix=file_suffix,
		epoch_length=args.epoch_length,
	)

	# Standard tile datasets for val/test (sliding window eval uses 48x48 crops)
	cycle_dataset_val = CycleDataset(path_val, split="validation", data_percentage=args.data_percentage,
	                                  feed_timeloc=False, n_timesteps=n_timesteps, file_suffix=file_suffix)
	cycle_dataset_test = CycleDataset(path_test, split="testing", data_percentage=args.data_percentage,
	                                   feed_timeloc=False, n_timesteps=n_timesteps, file_suffix=file_suffix)

	train_dataloader = DataLoader(cycle_dataset_train, batch_size=config["training"]["batch_size"],
	                              shuffle=True, num_workers=4)
	val_dataloader = DataLoader(cycle_dataset_val, batch_size=1, shuffle=False, num_workers=2)
	test_dataloader = DataLoader(cycle_dataset_test, batch_size=1, shuffle=False, num_workers=2)

	effective_batch = args.batch_size * args.grad_accum_steps
	print(f"Crop size: {args.crop_size}x{args.crop_size}")
	print(f"Batch size: {args.batch_size}, Grad accum: {args.grad_accum_steps}, Effective batch: {effective_batch}")
	print(f"Training iterations per epoch: {len(train_dataloader)}")

	# --- Model ---
	device = "cuda"
	if args.load_finetuned:
		weights_path = "data/checkpoints/pretrained_prithvi_1.0/default.pth"
		args.load_checkpoint = True
	elif args.load_checkpoint:
		weights_path = path_config.get_model_weights(args.model_size)
	else:
		weights_path = None

	model = PrithviSegConv3DCatHLS(
		config["pretrained_cfg"], weights_path,
		n_classes=4, model_size=args.model_size,
		feed_timeloc=args.feed_timeloc, n_layers=args.n_layers,
		hls_out_channels=args.hls_out_channels,
		hls_n_layers=args.hls_n_layers,
		progressive_fusion=args.progressive_fusion,
	)
	model = model.to(device)

	n_epochs = config["training"]["n_iteration"]

	print_trainable_parameters(model, detailed=True)

	checkpoint_dir = get_checkpoint_dir(group_name, args.data_percentage, args.selected_months)
	checkpoint = f"{checkpoint_dir}/{wandb_name}.pth"

	# --- Optimizer with layer-wise LR decay ---
	head_lr = args.learning_rate
	backbone_lr = args.learning_rate * args.backbone_lr_scale

	if args.load_checkpoint:
		param_groups = get_layer_lr_groups(
			model,
			head_lr=head_lr,
			backbone_lr=backbone_lr,
			layer_decay=args.layer_decay,
		)
	else:
		param_groups = [{'params': list(model.parameters()), 'lr': head_lr, 'name': 'all'}]

	# Print LR schedule summary
	print(f"\nLayer-wise LR schedule (decay={args.layer_decay}):")
	for pg in param_groups:
		n_params = sum(p.numel() for p in pg['params'])
		print(f"  {pg['name']:25s}  lr={pg['lr']:.2e}  params={n_params:,}")
	print()

	if args.optimizer == "lamb":
		from torch_optimizer import Lamb
		optimizer = Lamb(param_groups, weight_decay=1e-4)
	else:
		optimizer = AdamW(param_groups, weight_decay=1e-4)
	print(f"Using optimizer: {args.optimizer}")

	# --- Cosine schedule with linear warmup ---
	warmup_epochs = min(args.warmup_epochs, n_epochs)
	cosine_epochs = n_epochs - warmup_epochs

	warmup_scheduler = LinearLR(
		optimizer,
		start_factor=0.01,
		end_factor=1.0,
		total_iters=warmup_epochs,
	)
	cosine_scheduler = CosineAnnealingLR(
		optimizer,
		T_max=cosine_epochs,
		eta_min=args.min_lr,
	)
	scheduler = SequentialLR(
		optimizer,
		schedulers=[warmup_scheduler, cosine_scheduler],
		milestones=[warmup_epochs],
	)
	print(f"LR schedule: {warmup_epochs} warmup epochs + {cosine_epochs} cosine epochs (min_lr={args.min_lr})")

	loss_fn = segmentation_loss_mae if args.loss == "mae" else segmentation_loss
	print(f"Using loss function: {args.loss}")
	print(f"Progressive fusion: {args.progressive_fusion}, HLS channels: {args.hls_out_channels}, HLS layers: {args.hls_n_layers}")

	# --- Training Loop with Gradient Accumulation ---
	best_acc_val = 100
	accum_steps = args.grad_accum_steps

	for epoch in range(n_epochs):

		loss_i = 0.0

		current_lr = optimizer.param_groups[0]['lr']
		print(f"Epoch {epoch} started (head_lr={current_lr:.2e})")
		model.train()
		optimizer.zero_grad()

		for j, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

			input = batch_data["image"]
			mask = batch_data["gt_mask"]

			mask = mask.to(device)

			out = model(input)
			cs = args.crop_size
			out = out[:, :, :cs, :cs]
			mask = mask[:, :, :cs, :cs]

			loss = loss_fn(mask=mask, pred=out, device=device)
			loss = loss / accum_steps  # normalize for accumulation
			loss.backward()

			loss_i += loss.item() * accum_steps * mask.size(0)

			# Step optimizer every accum_steps iterations
			if (j + 1) % accum_steps == 0 or (j + 1) == len(train_dataloader):
				optimizer.step()
				optimizer.zero_grad()

			if j % 50 == 0:
				to_print = f"Epoch: {epoch}, iteration: {j}, loss: {loss.item() * accum_steps} \n "
				print(to_print)

		epoch_loss_train = loss_i / len(train_dataloader.dataset)

		# Step LR schedule (per epoch, after training)
		scheduler.step()

		# Validation Phase (sliding window, 48x48 crops over full tiles)
		acc_dataset_val, _, epoch_loss_val = eval_data_loader_crops(
			val_dataloader, model, device, get_masks_paper("train"),
			crop_size=args.crop_size, loss_fn=loss_fn)
		acc_dataset_test, _, epoch_loss_test = eval_data_loader_crops(
			test_dataloader, model, device, get_masks_paper("test"),
			crop_size=args.crop_size, loss_fn=loss_fn)

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

		acc_dataset_val_mean = np.mean(list(acc_dataset_val.values()))

		if acc_dataset_val_mean < best_acc_val:
			save_checkpoint(model, optimizer, epoch, epoch_loss_train, epoch_loss_val, checkpoint, selected_months=args.selected_months)
			best_acc_val = acc_dataset_val_mean

	model.load_state_dict(torch.load(checkpoint)["model_state_dict"])

	acc_dataset_val, _, epoch_loss_val = eval_data_loader_crops(
		val_dataloader, model, device, get_masks_paper("train"),
		crop_size=args.crop_size, loss_fn=loss_fn)
	acc_dataset_test, _, _ = eval_data_loader_crops(
		test_dataloader, model, device, get_masks_paper("test"),
		crop_size=args.crop_size, loss_fn=loss_fn)

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
