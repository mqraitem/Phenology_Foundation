
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import path_config

from lib.models.lsp_transformer_pixels import TemporalTransformer
from lib.utils import segmentation_loss_pixels, segmentation_loss_pixels_mae, segmentation_loss, segmentation_loss_mae, eval_data_loader, get_masks_paper, print_trainable_parameters, save_checkpoint,str2bool, months_to_str, get_checkpoint_dir

from lib.utils import get_data_paths
from lib.dataloaders.dataloaders_pixels import CycleDatasetPixels
from lib.dataloaders.dataloaders import CycleDataset
from arg_configs import get_core_parser

#######################################################################################

def main():

	# Parse the arguments - only core args needed for this model
	parser = get_core_parser()
	parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae"],
	                   help="Loss function: mse (mean squared error) or mae (mean absolute error)")
	parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "lamb"],
	                   help="Optimizer: adamw or lamb")
	args = parser.parse_args()

	months_str = months_to_str(args.selected_months)
	file_suffix = f"_m{months_str}"
	n_timesteps = len(args.selected_months)

	wandb_config = {
		"learningrate": args.learning_rate,
		"batch_size": args.batch_size,
		"data_percentage": args.data_percentage,
		"loss": args.loss,
		"optimizer": args.optimizer,
		"selected_months": args.selected_months,
	}

	args.model_size = "300m"
	wandb_name = args.wandb_name

	with open(f'configs/prithvi_{args.model_size}.yaml', 'r') as f:
		config = yaml.safe_load(f)

	config["training"]["n_iteration"] = args.n_epochs
	config["pretrained_cfg"]["img_size"] = 336

	group_name = args.group_name

	if args.logging:
		wandb.init(
				project=args.wandb_project or f"phenology_crop_{args.data_percentage}",
				group=group_name,
				config = wandb_config,
				name=wandb_name,
				)
		wandb.run.log_code(".")

	path_train=get_data_paths("training", args.data_percentage, args.selected_months)
	path_val=get_data_paths("validation", args.data_percentage, args.selected_months)
	path_test=get_data_paths("testing", args.data_percentage, args.selected_months)

	cycle_dataset_train=CycleDatasetPixels(path_train,split="training", data_percentage=args.data_percentage, n_timesteps=n_timesteps, file_suffix=file_suffix)
	cycle_dataset_val=CycleDataset(path_val,split="validation", data_percentage=args.data_percentage, n_timesteps=n_timesteps, file_suffix=file_suffix)
	cycle_dataset_test=CycleDataset(path_test,split="testing", data_percentage=args.data_percentage, n_timesteps=n_timesteps, file_suffix=file_suffix)


	config["training"]["batch_size"] = args.batch_size
	config["validation"]["batch_size"] = 1
	config["test"]["batch_size"] = 1

	train_dataloader=DataLoader(cycle_dataset_train,batch_size=config["training"]["batch_size"],shuffle=config["training"]["shuffle"],num_workers=4)
	val_dataloader=DataLoader(cycle_dataset_val,batch_size=config["validation"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=2)
	test_dataloader=DataLoader(cycle_dataset_test,batch_size=config["test"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=2)

	device = "cuda"

	model = TemporalTransformer(
		input_channels=6,
		seq_len=n_timesteps,
		num_classes=4,
		d_model=128,
		nhead=4,
		num_layers=3,
		dropout=0.1
	)

	print_trainable_parameters(model)
	model=model.to(device)

	checkpoint_dir = get_checkpoint_dir(group_name, args.data_percentage, args.selected_months)
	checkpoint = f"{checkpoint_dir}/{wandb_name}.pth"
	
	if args.optimizer == "lamb":
		from torch_optimizer import Lamb
		optimizer = Lamb(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
	else:
		optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
	print(f"Using optimizer: {args.optimizer}")
	scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)

	train_loss_fn = segmentation_loss_pixels_mae if args.loss == "mae" else segmentation_loss_pixels
	eval_loss_fn = segmentation_loss_mae if args.loss == "mae" else segmentation_loss
	print(f"Using loss function: {args.loss}")

	best_acc_val=100
	for epoch in range(config["training"]["n_iteration"]):

		loss_i=0.0

		print("iteration started")
		model.train()

		for j,batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

			input = batch_data["image"]
			mask = batch_data["gt_mask"]	

			# input=input.to(device)
			mask=mask.to(device)

			optimizer.zero_grad()
			out=model(input, processing_images=False)

			loss=train_loss_fn(mask,out,device=device)
			loss_i += loss.item() * input.size(0)  # Multiply by batch size

			loss.backward()
			optimizer.step()

			if j%500==0:
				to_print = f"Epoch: {epoch}, iteration: {j}, loss: {loss.item()} \n "
				print(to_print)


		epoch_loss_train = loss_i / len(train_dataloader.dataset)

		# Validation Phase
		acc_dataset_val, _, epoch_loss_val = eval_data_loader(val_dataloader, model, device, get_masks_paper("train"), loss_fn=eval_loss_fn)
		acc_dataset_test, _, epoch_loss_test = eval_data_loader(test_dataloader, model, device, get_masks_paper("test"), loss_fn=eval_loss_fn)
		
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

		print("="*100)
		to_print = f"Epoch: {epoch}, val_loss: {epoch_loss_val} \n "
		for idx in range(4):
			to_print += f"acc_val_{idx}: {acc_dataset_val[idx]} \n "

		for idx in range(4):
			to_print += f"acc_test_{idx}: {acc_dataset_test[idx]} \n "

		print(to_print)
		print("="*100)


		scheduler.step(epoch_loss_val)
		acc_dataset_val_mean = np.mean(list(acc_dataset_val.values()))

		if acc_dataset_val_mean<best_acc_val:
			save_checkpoint(model, optimizer, epoch, epoch_loss_train, epoch_loss_val, checkpoint, selected_months=args.selected_months)
			best_acc_val=acc_dataset_val_mean

	model.load_state_dict(torch.load(checkpoint)["model_state_dict"])

	acc_dataset_val, _, epoch_loss_val = eval_data_loader(val_dataloader, model, device, get_masks_paper("train"), loss_fn=eval_loss_fn)
	acc_dataset_test, _, _ = eval_data_loader(test_dataloader, model, device, get_masks_paper("test"), loss_fn=eval_loss_fn)

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
