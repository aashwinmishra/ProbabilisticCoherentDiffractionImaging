"""
Utility functions for model training, evaluation and visualization.
"""
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


def get_devices() -> torch.device:
  """
  Returns gpu device if available, else cpu
  """
  return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seeds(seed: int = 42):
  """
  Sets torch seeds to ensure reproducability.
  """
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)


def save_model(model_dir: str, model_name: str, model: torch.nn.Module):
  """
  Saves pytorch model in model_dir with model_name.
  Args:
    model_dir: Directory to save model in.
    model_name: name of file to store model.
    model: model to be saved.
  Returns:
    None
  """
  os.makedirs(model_dir, exist_ok=True)
  if not model_name.endswith("pt"):
    model_name += ".pt"
  torch.save(model.state_dict(), os.path.join(model_dir, model_name))


def create_summary_writer(experiment_name: str, model_name: str, extras: str = None):
        # -> torch.utils.tensorboard.SummaryWriter:
  """
  Instantiates and returns a Summary writer for the experiment, that writers to
  runs/experiment_name/model_name/extras
  Args:
    experiment_name: Name of experiment (say, dataset)
    model_name: Name of model used
    extras: Additional details
  Returns:
    SummaryWriter instance for the experiment
  """
  if extras:
    log_dir = os.path.join("runs/", experiment_name, model_name, extras)
  else:
    log_dir = os.path.join("runs/", experiment_name, model_name)
  writer = torch.utils.tensorboard.SummaryWriter(log_dir)
  return writer


def predict_and_plot(model: torch.nn.Module, 
                     test_dl: torch.utils.data.DataLoader, 
                     device: torch.device):
  """
  Takes the test data loader and a probabilistic model, predicts for entire data.
  For a subsample, plots the predictions.
  Args:
    model: Trained probabilistic model (PNN, Deep Ensemble, etc)
    test_dl: Data Loader for the test data
    device: Device of the model
  Returns:
    tuple with amps_mean, amps_var, phis_mean, phis_var
  """
  model.eval() #just to be safe
  amps_mean, amps_var, phis_mean, phis_var = [], [], [], []
  amps_true, phis_true = [], []
  for i, ft_images in enumerate(test_dl):
    amps, phis = ft_images[1], ft_images[2]
    ft_images = ft_images[0].to(device)
    amp_mean, amp_var, phi_mean, phi_var = model(ft_images)
    for j in range(ft_images.shape[0]):
      amps_mean.append(amp_mean[j].detach().to("cpu").numpy())
      amps_var.append(amp_var[j].detach().to("cpu").numpy())
      phis_mean.append(phi_mean[j].detach().to("cpu").numpy())
      phis_var.append(phi_var[j].detach().to("cpu").numpy())
      amps_true.append(amps[j].detach().to("cpu").numpy())
      phis_true.append(phis[j].detach().to("cpu").numpy())

  amps_mean = np.array(amps_mean).squeeze()
  amps_var = np.array(amps_var).squeeze()
  phis_mean = np.array(phis_mean).squeeze()
  phis_var = np.array(phis_var).squeeze()
  amps_true = np.array(amps_true).squeeze()
  phis_true = np.array(phis_true).squeeze()

  n = 6
  h, w = 64, 64
  len_test = len(test_dl)
  fig, ax = plt.subplots(8, n, figsize=(24, 16))
  plt.gcf().text(0.02, 0.85, "True Amp", fontsize=18)
  plt.gcf().text(0.02, 0.75, "Pred Amp", fontsize=18)
  plt.gcf().text(0.02, 0.65, "Diff Amp", fontsize=18)
  plt.gcf().text(0.02, 0.55, "Sigma Amp", fontsize=18)
  plt.gcf().text(0.02, 0.45, "True Phase", fontsize=18)
  plt.gcf().text(0.02, 0.35, "Pred Phase", fontsize=18)
  plt.gcf().text(0.02, 0.25, "Diff Phase", fontsize=18)
  plt.gcf().text(0.02, 0.15, "Sigma Phase", fontsize=18)

  for i in range(0, n):
    j = int(round(np.random.rand()*len_test))

    im = ax[0,i].imshow(amps_true[j].reshape(h, w))
    plt.colorbar(im, ax=ax[0,i], format='%.2f')

    im = ax[1,i].imshow(amps_mean[j].reshape(h, w))
    plt.colorbar(im, ax=ax[1,i], format='%.2f')

    im = ax[2,i].imshow(amps_true[j].reshape(h, w) - amps_mean[j].reshape(h, w))
    plt.colorbar(im, ax=ax[2,i], format='%.2f')

    im = ax[3,i].imshow(amps_var[j].reshape(h, w))
    plt.colorbar(im, ax=ax[3,i], format='%.2f')

    im = ax[4,i].imshow(phis_true[j].reshape(h, w))
    plt.colorbar(im, ax=ax[4,i], format='%.2f')

    im = ax[5,i].imshow(phis_mean[j].reshape(h, w))
    plt.colorbar(im, ax=ax[5,i], format='%.2f')

    im = ax[6,i].imshow(phis_true[j].reshape(h, w) - phis_mean[j].reshape(h, w))
    plt.colorbar(im, ax=ax[6,i], format='%.2f')

    im = ax[7,i].imshow(phis_var[j].reshape(h, w))
    plt.colorbar(im, ax=ax[7,i], format='%.2f')

  plt.savefig("Samples.pdf", bbox_inches='tight')

  return amps_true, phis_true, amps_mean, amps_var, phis_mean, phis_var


def evaluate(model: torch.nn.Module, 
             test_dl: torch.utils.data.DataLoader, 
             type: str,
             device: torch.device):
    """
  Takes the test data loader and a model, predicts for entire data.
  Returns the MAE and MSE metrics over the dataste.
  Args:
    model: Trained model 
    test_dl: Data Loader for the test data
    type: Type of model (NN, PNN, Ensemble)
    device: Device of the model
  Returns:
    tuple with MAE_amps, MSE_amps, MAE_phase, MSE_Phase
  """
    model.eval()
    amps_true, phis_true, amps_mean, amps_var, phis_mean, phis_var = predict_and_plot(model, test_dl, device)
    amps_mean = torch.from_numpy(amps_mean)
    phis_mean = torch.from_numpy(phis_mean)
    amps_true = torch.from_numpy(amps_true)
    phis_true = torch.from_numpy(phis_true)
    with torch.inference_mode():
      amps_mse = torch.nn.functional.mse_loss(amps_mean, amps_true)
      phis_mse = torch.nn.functional.mse_loss(phis_mean, phis_true)
      amps_mae = torch.nn.functional.l1_loss(amps_mean, amps_true)
      phis_mae = torch.nn.functional.l1_loss(phis_mean, phis_true)
    print("Amp MAE:", amps_mae.detach().item())
    print("Phi MAE:", phis_mae.detach().item())
    print("Amp MSE:", amps_mse.detach().item())
    print("Phi MSE:", phis_mse.detach().item())
    return amps_mae.detach().item(), amps_mse.detach().item(), phis_mae.detach().item(), phis_mse.detach().item()


def load_ptycho_data(file_path: str) -> PtychoDataContainer:
    """
    Load the npz-serialized ptycho data.

    Args:
        file_path (str): Path to the npz file.

    Returns:
        PtychoDataContainer: Loaded ptycho data.
    """
    data = np.load(file_path, allow_pickle=True)
    return PtychoDataContainer(
        X=data['X'],
        Y_I=data['Y_I'],
        Y_phi=data['Y_phi'],
        norm_Y_I=data['norm_Y_I'],
        YY_full=data['YY_full'],
        coords_nominal=data['coords_nominal'],
        coords_true=data['coords_true'],
        nn_indices=data['nn_indices'],
        global_offsets=data['global_offsets'],
        local_offsets=data['local_offsets'],
        probeGuess=data['probe']
    )


def inspect_ptycho_frames(data: PtychoDataContainer, 
                          num_frames: int = 2):
    """
    Visually inspect a couple of frames from X, Y_I, and Y_phi.
    Args:
        data (PtychoDataContainer): Loaded ptycho data.
        num_frames (int): Number of frames to display. Defaults to 2.
    """
    fig, axes = plt.subplots(3, num_frames, figsize=(5*num_frames, 15))
    for i in range(num_frames):
        axes[0, i].imshow(data.X[i, ..., 0], cmap='viridis')
        axes[0, i].set_title(f'X - Frame {i}')
        axes[0, i].axis('off')
        axes[1, i].imshow(data.Y_I[i, ..., 0], cmap='viridis')
        axes[1, i].set_title(f'Y_I - Frame {i}')
        axes[1, i].axis('off')
        axes[2, i].imshow(data.Y_phi[i, ..., 0], cmap='viridis')
        axes[2, i].set_title(f'Y_phi - Frame {i}')
        axes[2, i].axis('off')
    plt.tight_layout()
    plt.show()
