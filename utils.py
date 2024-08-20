"""
Utility functions for model training and evaluation.
"""
import torch
import os


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
  for i, ft_images in enumerate(test_dl):
    ft_images = ft_images[0].to(device)
    amp_mean, amp_var, phi_mean, phi_var = model(ft_images)
    for j in range(ft_images.shape[0]):
      amps_mean.append(amp_mean[j].detach().to("cpu").numpy())
      amps_var.append(amp_var[j].detach().to("cpu").numpy())
      phis_mean.append(phi_mean[j].detach().to("cpu").numpy())
      phis_var.append(phi_var[j].detach().to("cpu").numpy())

  amps_mean = np.array(amps_mean).squeeze()
  amps_var = np.array(amps_var).squeeze()
  phis_mean = np.array(phis_mean).squeeze()
  phis_var = np.array(phis_var).squeeze()

  return amps_mean, amps_var, phis_mean, phis_var
