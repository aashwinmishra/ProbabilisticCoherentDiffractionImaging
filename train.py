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
from models import PtychoNN, PtychoPNN, DeepEnsemble, PtychoMCDropout
from engine import train
from utils import get_devices, set_seeds, save_model


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./gdrive/MyDrive/PtychoNNData/")
parser.add_argument("--num_epochs", type=int, default=125)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--model", type=str, default="PtychoModel")
parser.add_argument("--model_type", type=str, default="PtychoNN")
args = parser.parse_args()

set_seeds(42)
device = get_devices()
d = get_dataloaders(args.data_path)
train_dl, val_dl, test_dl = d["train_dl"], d["val_dl"], d["test_dl"]
if args.model_type == "PtychoNN":
  model = PtychoNN().to(device)
elif args.model_type == "PtychoPNN":
  model = PtychoPNN().to(device)
elif args.model_type == "PtychoMCDropout":
  model = PtychoMCDropout().to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
results = train(model, train_dl, val_dl, opt, device, args.num_epochs)
model_name = args.model + str(args.num_epochs)
save_model("./Models", model_name, model)

