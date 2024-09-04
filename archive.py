"""
Functions to train and evaluate deterministic PtychoNN model.
"""
import torch
import torch.nn as nn

def train_step(model: torch.nn.Module,
               train_dl: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               opt: torch.optim.Optimizer,
               device: torch.device
               ) -> dict:
  """
  Performs 1 epoch of training of model on train dataloader,
  returning model loss on the amplitude and phase reconstruction.
  Args:
    model: model too be trained
    train_dl: Dataloader with training data
    loss_fn: Differentiable loss function to be used for gradients
    opt: Optimizer to train model.
    device: Device on which model and data will reside.
  Returns:
      Dict with keys "total_loss", "amp_loss" and "phase_loss".
  """
  model.train()
  total_loss, amplitude_loss, phase_loss = 0.0, 0.0, 0.0
  for ft_images, amps, phis in train_dl:
    ft_images, amps, phis = ft_images.to(device), amps.to(device), phis.to(device)
    pred_amps, pred_phis = model(ft_images)
    amp_loss = loss_fn(pred_amps, amps)
    phi_loss = loss_fn(pred_phis, phis)
    loss = amp_loss + phi_loss
    opt.zero_grad()
    loss.backward()
    opt.step()

    total_loss += loss.detach().item()
    amplitude_loss += amp_loss.detach().item()
    phase_loss += phi_loss.detach().item()

  model.eval()
  return {"total_loss": total_loss/len(train_dl),
          "amp_loss": amplitude_loss/len(train_dl),
          "phase_loss": phase_loss/len(train_dl)}

def val_step(model: torch.nn.Module,
            val_dl: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            device: torch.device
            ) -> dict:
  """
  Performs 1 epoch of evaluation of model on validation dataloader,
  returning model loss on the amplitude and phase reconstruction.
  Args:
    model: model too be trained
    train_dl: Dataloader with training data
    loss_fn: Loss function to be used for gradients
    device: Device on which model and data will reside.
  Returns:
      Dict with keys "total_loss", "amp_loss" and "phase_loss".
  """
  model.eval()
  total_loss, amplitude_loss, phase_loss = 0.0, 0.0, 0.0
  with torch.inference_mode():
    for ft_images, amps, phis in val_dl:
      ft_images, amps, phis = ft_images.to(device), amps.to(device), phis.to(device)
      pred_amps, pred_phis = model(ft_images)
      amp_loss = loss_fn(pred_amps, amps)
      phi_loss = loss_fn(pred_phis, phis)
      loss = amp_loss + phi_loss

      total_loss += loss.detach().item()
      amplitude_loss += amp_loss.detach().item()
      phase_loss += phi_loss.detach().item()

  return {"total_loss": total_loss/len(val_dl),
          "amp_loss": amplitude_loss/len(val_dl),
          "phase_loss": phase_loss/len(val_dl)}


def train(model: torch.nn.Module,
          train_dl: torch.utils.data.DataLoader,
          val_dl: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          opt: torch.optim.Optimizer,
          device: torch.device,
          num_epochs: int) -> dict:
  """
  Performs defined number of epochs of training and evaluation for the model on
  the data loaders, returning the loss history on amplitude and phase reconstruction.
  Args:
    model: model to be trained and evaluated.
    train_dl: Dataloader with training data.
    val_dl: Dataloader with testing data.
    loss_fn: Differentiable loss function to use for gradients.
    opt: Optimizer to tune model params.
    device: Device on which model and eventually data shall reside
    num_epochs: Number of epochs of training
  Returns:
    Dict with history of "total_loss", "amp_loss" and "phase_loss".
  """
  amp_loss_train, phi_loss_train, amp_loss_val, phi_loss_val = [], [], [], []
  for epoch in range(num_epochs):
    train_results = train_step(model, train_dl, loss_fn, opt, device)
    val_results = val_step(model, val_dl, loss_fn, device)
    amp_loss_train.append(train_results["amp_loss"])
    phi_loss_train.append(train_results["phase_loss"])
    amp_loss_val.append(val_results["amp_loss"])
    phi_loss_val.append(val_results["phase_loss"])
    print(f"Epoch: {epoch+1} Train Amp: {amp_loss_train[-1]} Train Phi: {phi_loss_train[-1]} Val Amp: {amp_loss_val[-1]} Val Phi: {phi_loss_val[-1]}")

  return {"amp_loss_train": amp_loss_train, "phi_loss_train": phi_loss_train,
          "amp_loss_val": amp_loss_val, "phi_loss_val": phi_loss_val}


# set_seeds(42)
# device = get_devices()
# d = get_dataloaders(args.data_path)
# train_dl, val_dl, test_dl = d["train_dl"], d["val_dl"], d["test_dl"]
# model = PtychoNNBase().to(device)
# loss_fn = torch.nn.L1Loss()
# opt = torch.optim.Adam(model.parameters(), lr=args.lr)
# results = train(model, train_dl, val_dl, loss_fn, opt, device, args.num_epochs)
# model_name = args.model + str(args.num_epochs)
# save_model("./Models", "model_" + args.model, model)
