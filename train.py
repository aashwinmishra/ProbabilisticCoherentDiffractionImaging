"""
Takes parameters from user; trains, evaluates and saves models on 
Coherent Diffraction Imaging Data.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from data_setup import get_dataloaders
from models import PtychoNNBase
from engine import train
from utils import get_devices, set_seeds, save_model


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./gdrive/MyDrive/PtychoNNData/")
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--model", type=str, default="PtychoNNBase")
args = parser.parse_args()

set_seeds(42)
device = get_devices()
d = get_dataloaders(args.data_path)
train_dl, val_dl, test_dl = d["train_dl"], d["val_dl"], d["test_dl"]
model = PtychoNNBase().to(device)
loss_fn = torch.nn.L1Loss()
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
results = train(model, train_dl, val_dl, loss_fn, opt, device, args.num_epochs)
model_name = args.model + str(args.num_epochs)
save_model("./Models", "model_" + args.model, model)
