import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)
import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
import time
from SLPF import SLPF
from model.BasicTrainer import Trainer
from model.BasicInferencer import Inferencer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from os.path import join


# Set environment variables for CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# Parse dataset argument early to load the correct config file
parser = argparse.ArgumentParser(description='arguments', add_help=False)
parser.add_argument('--dataset', default='PEMS08', type=str)
args_temp, _ = parser.parse_known_args()
DATASET = args_temp.dataset
# Load config file based on dataset
config_file = f'./configs/{DATASET}.conf'
config = configparser.ConfigParser()
config.read(config_file)
# Define all arguments
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default='train', type=str)
args.add_argument('--cuda', default=True, type=bool)
args.add_argument('--comment', default='', type=str)

# Data-related arguments
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_locs', default=config['data']['num_locs'], type=int)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--num_unsensed_locs', default=config['data']['num_unsensed_locs'], type=int)

# Model-related arguments
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--num_layer', default=config['model']['num_layer'], type=int)

# Training-related arguments
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--device', default=config['train']['device'], type=int, help='indices of GPUs')
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--weight_decay', default=config['train']['weight_decay'], type=eval)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help='use real value for loss calculation')

# Testing-related arguments
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
args.add_argument('--adp_model_path', default='', type=str)
args.add_argument('--forecast_model_path', default='', type=str)
args.add_argument('--agg_model_path', default='', type=str)

# Logging-related arguments
args.add_argument('--log_dir', default='../runs', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args.add_argument('--tensorboard', action='store_true', help='tensorboard')
args = args.parse_args()

# Set random seed for reproducibility
init_seed(args.seed)

# Configure log directory
save_name = (
    time.strftime("%m-%d-%Hh%Mm%Ss") + args.comment + "_" + args.dataset +
    f"_embed{args.embed_dim}_lyr{args.num_layer}_lr{args.lr_init}_wd{args.weight_decay}"
)
base_log_dir = '../runs'
log_dir = join(base_log_dir, args.dataset, save_name)
args.log_dir = f"{log_dir}_s_{args.seed}_m_{args.num_unsensed_locs}"

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
else:
    print('Model save path already exists')


# Load data and graph structure
train_loader, val_loader, test_loader, scaler, sensed_locations, unsensed_locations, graph = get_dataloader(
    args, normalizer=args.normalizer
)

# Initialize models for three stages
adp_model = SLPF(sensed_locations, unsensed_locations, graph, args).to(args.device)
adp_model.stage = 'adp'

forecast_model = SLPF(sensed_locations, unsensed_locations, graph, args).to(args.device)
forecast_model.stage = 'forecast'

agg_model = SLPF(sensed_locations, unsensed_locations, graph, args).to(args.device)
agg_model.stage = 'agg'

if args.mode == 'train':
    # Set up optimizers for each model
    adp_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, adp_model.parameters()),
        lr=args.lr_init, weight_decay=args.weight_decay
    )
    forecast_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, forecast_model.parameters()),
        lr=args.lr_init, weight_decay=args.weight_decay
    )
    agg_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, agg_model.parameters()),
        lr=args.lr_init, weight_decay=args.weight_decay
    )

    # Select loss function
    if args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    elif args.loss_func == 'huber_loss':
        loss = torch.nn.HuberLoss(delta=1.0).to(args.device)
    else:
        raise ValueError("Unsupported loss function")

    # Custom masked MAE loss for LA dataset
    class MaskedMAELoss:
        def _get_name(self):
            return self.__class__.__name__

        def __call__(self, preds, labels, null_val=0.0):
            if np.isnan(null_val):
                mask = ~torch.isnan(labels)
            else:
                mask = labels != null_val
            mask = mask.float()
            mask /= torch.mean(mask)
            mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
            loss = torch.abs(preds - labels)
            loss = loss * mask
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
            return torch.mean(loss)

    if DATASET == 'LA':
        loss = MaskedMAELoss()

    # Start training
    trainer = Trainer(
        sensed_locations, unsensed_locations,
        adp_model, forecast_model, agg_model,
        adp_optimizer, forecast_optimizer, agg_optimizer,
        train_loader, val_loader, test_loader, scaler, args, args.device, loss
    )
    trainer.train()
else:
    # Inference mode: load best models and run inference
    adp_best_model = torch.load(args.adp_model_path)
    adp_model.load_state_dict(adp_best_model)

    forecast_best_model = torch.load(args.forecast_model_path)
    forecast_model.load_state_dict(forecast_best_model)

    agg_best_model = torch.load(args.agg_model_path)
    agg_model.load_state_dict(agg_best_model)

    inferencer = Inferencer(
        sensed_locations, unsensed_locations,
        adp_model, forecast_model, agg_model,
        test_loader, scaler, args, args.device
    )
    inferencer.inference()


