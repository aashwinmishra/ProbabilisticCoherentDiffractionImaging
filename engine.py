"""
Functions to train and evaluate model on the image dataset
"""
import torch
import torch.nn as nn


def train_step(model: torch.nn.Module,
               train_dl: torch.utils.data.DataLoader,
               opt: torch.optim.Optimizer,
               device: torch.device
               ) -> dict:
  """
  Performs 1 epoch of training of model on train dataloader,
  returning model loss on the amplitude and phase reconstruction.
  Args:
    model: model too be trained
    train_dl: Dataloader with training data
    opt: Optimizer to train model.
    device: Device on which model and data will reside.
  Returns:
      Dict with keys "amp_loss", "phase_loss", "amp_metric", "phase_metric".
  """
  # iterations_per_epoch = len(train_dl)
  # step_size = 6*iterations_per_epoch
  # scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=0.0001, 
  #                                               max_lr=0.001, 
  #                                               step_size_up=step_size, 
  #                                               cycle_momentum=False, 
  #                                               mode='triangular2')               
  model.train()
  amplitude_loss, phase_loss, amplitude_metric, phase_metric = 0.0, 0.0, 0.0, 0.0
  for ft_images, amps, phis in train_dl:
    ft_images, amps, phis = ft_images.to(device), amps.to(device), phis.to(device)
    amp_loss, phi_loss, amp_metric, phi_metric = model.train_step(ft_images, amps, phis)
    loss = amp_loss + phi_loss
    opt.zero_grad()
    loss.backward()
    opt.step()

    amplitude_loss += amp_loss.detach().item()
    phase_loss += phi_loss.detach().item()
    amplitude_metric += amp_metric.detach().item()
    phase_metric += phi_metric.detach().item()

    #scheduler.step()

  model.eval()
  return {"amp_loss": amplitude_loss/len(train_dl),
          "phase_loss": phase_loss/len(train_dl),
          "amp_metric": amplitude_metric/len(train_dl),
          "phase_metric": phase_metric/len(train_dl)}


def val_step(model: torch.nn.Module,
            val_dl: torch.utils.data.DataLoader,
            device: torch.device
            ) -> dict:
  """
  Performs 1 epoch of evaluation of model on validation dataloader,
  returning model loss on the amplitude and phase reconstruction.
  Args:
    model: model too be trained
    train_dl: Dataloader with training data
    device: Device on which model and data will reside.
  Returns:
      Dict with keys "total_loss", "amp_loss" and "phase_loss".
  """
  model.eval()
  amplitude_loss, phase_loss, amplitude_metric, phase_metric = 0.0, 0.0, 0.0, 0.0
  with torch.inference_mode():
    for ft_images, amps, phis in val_dl:
      ft_images, amps, phis = ft_images.to(device), amps.to(device), phis.to(device)
      amp_loss, phi_loss, amp_metric, phi_metric = model.eval_step(ft_images, amps, phis)
      loss = amp_loss + phi_loss

      amplitude_loss += amp_loss.detach().item()
      phase_loss += phi_loss.detach().item()
      amplitude_metric += amp_metric.detach().item()
      phase_metric += phi_metric.detach().item()

  return {"amp_loss": amplitude_loss/len(val_dl),
          "phase_loss": phase_loss/len(val_dl),
          "amp_metric": amplitude_metric/len(val_dl),
          "phase_metric": phase_metric/len(val_dl)}


def train(model: torch.nn.Module,
          train_dl: torch.utils.data.DataLoader,
          val_dl: torch.utils.data.DataLoader,
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
    opt: Optimizer to tune model params.
    device: Device on which model and eventually data shall reside
    num_epochs: Number of epochs of training
  Returns:
    Dict with history of "total_loss", "amp_loss" and "phase_loss".
  """
  amp_loss_train, phi_loss_train, amp_loss_val, phi_loss_val = [], [], [], []
  amp_metric_train, phi_metric_train, amp_metric_val, phi_metric_val = [], [], [], []
  for epoch in range(num_epochs):
    train_results = train_step(model, train_dl, opt, device)
    val_results = val_step(model, val_dl, device)
    amp_loss_train.append(train_results["amp_loss"])
    phi_loss_train.append(train_results["phase_loss"])
    amp_loss_val.append(val_results["amp_loss"])
    phi_loss_val.append(val_results["phase_loss"])
    amp_metric_train.append(train_results["amp_metric"])
    phi_metric_train.append(train_results["phase_metric"])
    amp_metric_val.append(val_results["amp_metric"])
    phi_metric_val.append(val_results["phase_metric"])
    print(f"Epoch: {epoch+1} Train Amp Loss: {amp_loss_train[-1]:.5f} Train Phi Loss: {phi_loss_train[-1]:.5f} Val Amp Loss: {amp_loss_val[-1]:.5f} Val Phi Loss: {phi_loss_val[-1]:.5f}")
    print(f"Epoch: {epoch+1} Train Amp Metric: {amp_metric_train[-1]:.5f} Train Phi Metric: {phi_metric_train[-1]:.5f} Val Amp Metric: {amp_metric_val[-1]:.5f} Val Phi Metric: {phi_metric_val[-1]:.5f}")

  return {"amp_loss_train": amp_loss_train, "phi_loss_train": phi_loss_train,
          "amp_loss_val": amp_loss_val, "phi_loss_val": phi_loss_val,
          "amp_metric_train": amp_metric_train, "phi_metric_train": phi_metric_train,
          "amp_metric_val": amp_metric_val, "phi_metric_val": phi_metric_val}

