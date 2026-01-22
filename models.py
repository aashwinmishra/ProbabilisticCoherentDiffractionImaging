"""
Classes definitions and constructor functions for the models. This includes:
a) A Basic PtychoNN Deterministic model,
b) A PtychoPNN Probabilistic model,
c) A Deep Ensemble model (Lakshminarayanan et al, "Simple and scalable predictive uncertainty estimation using deep ensembles", Neurips (2017)"
d) A MC Dropout variant of PtychoNN (Based off of the Bayesian SegNet work by Kendall et al (2016))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def contraction_block(in_channels: int,
                      mid_channels: int,
                      out_channels: int,
                      kernel_size: int=3,
                      stride: int=1,
                      padding: int=1,
                      pool_factor:int=2,
                      use_dropout: bool=False,
                      dropout_rate: float=0.5) -> nn.Module:
  """
  Creates a constituent Conv block for the encoder section of the ptychoNN model.
  Consists of Conv-Relu-Conv-Relu-Maxpool layers.
  Optionally integrates Dropout layers for regularization/Bayesian approximation.

  Args:
    in_channels: Input channels to the block
    mid_channels: intermediate channels (ie, output of first conv layer and input channels to the second)
    out_channels: Final number of output channels from the conv block
    kernel_size: Uniform kernel size across both conv layers in the block
    stride: Uniform stride across both conv layers in the block
    padding: Uniform padding across both conv layers in the block
    pool_factor: Kernel size of the square max pool
    use_droput: If True, inserts a dropout layer after the activations. Default False.
    dropout_rate: Dropout probability. Defaults to 0.5 based on Gal & Gharmani.

  Returns:
    nn.Sequential container of modules.
  """
  layers = [
      nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(mid_channels),
      nn.ReLU()
  ]
  if use_dropout:
    layers.append(nn.Dropout2d(p=dropout_rate))
  layers.extend([
      nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
  ])
  if use_dropout:
    layers.append(nn.Dropout2d(p=dropout_rate))
  layers.append(nn.MaxPool2d(pool_factor))
  return nn.Sequential(*layers)


def expansion_block(in_channels: int,
                    mid_channels: int,
                    out_channels: int,
                    kernel_size: int=3,
                    stride: int=1,
                    padding: int=1,
                    upsamling_factor:int=2,
                    use_dropout: bool=False,
                    dropout_rate: float=0.5) -> nn.Module:
  """
  Creates a constituent Conv block for the decoder sections of the ptychoNN model.
  Consists of Conv-Relu-Conv-Relu-Upsample layers.
  Optionally, adds dropout layers after the activation for regularization/Bayesian approximation.

  Args:
    in_channels: Input channels to the block
    mid_channels: intermediate channels (ie, output of first conv layer and input channels to the second)
    out_channels: Final number of output channels from the conv block
    kernel_size: Uniform kernel size across both conv layers in the block
    stride: Uniform stride across both conv layers in the block
    padding: Uniform padding across both conv layers in the block
    upsampling_factor: Scale factor for the upsampling layer
    use_droput: If True, inserts a dropout layer after the each activation. Default False.
    dropout_rate: Dropout probability. Defaults to 0.5 based on Gal & Gharmani. 

  Returns:
    nn.Sequential container of modules.
  """
  layers = [
      nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(mid_channels),
      nn.ReLU()
  ]
  if use_dropout:
    layers.append(nn.Dropout2d(p=dropout_rate))
  layers.extend([
      nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
  ])
  if use_dropout:
    layers.append(nn.Dropout2d(p=dropout_rate))
  layers.append(nn.Upsample(scale_factor=upsamling_factor, mode='bilinear'))
  return nn.Sequential(*layers)


class PtychoMCDropout(nn.Module):
  """
  Defines the base PtychoNN model, with a PNN output head for aleatoric uncertainty
  and dropout layers for regularization/epistemic uncertainty.
  Attributes:
    nconv: number of feature maps from the first conv layer.
    dropout_rate: dropout probability for the dropout layers.
  """
  def __init__(self, nconv: int=32, dropout_rate: float=0.25):
    super().__init__()
    self.encoder = nn.Sequential(
        contraction_block(in_channels=1, mid_channels=nconv, out_channels=nconv),
        contraction_block(in_channels=nconv, mid_channels=2*nconv, out_channels=2*nconv),
        contraction_block(in_channels=2*nconv, mid_channels=4*nconv, out_channels=4*nconv)
    )
    self.amplitude_decoder = nn.Sequential(
        expansion_block(in_channels=4*nconv, mid_channels=4*nconv, out_channels=4*nconv, use_dropout=True, dropout_rate=0.25),
        expansion_block(in_channels=4*nconv, mid_channels=2*nconv, out_channels=2*nconv),
        expansion_block(in_channels=2*nconv, mid_channels=2*nconv, out_channels=2*nconv),
    )
    self.amplitude_mean_end = nn.Sequential(
        nn.Conv2d(in_channels=2*nconv, out_channels=1, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid()
    )
    self.amplitude_log_sigma = nn.Sequential(
        nn.Conv2d(in_channels=2*nconv, out_channels=1, kernel_size=3, stride=1, padding=1)
    )
    self.phase_decoder = nn.Sequential(
        expansion_block(in_channels=4*nconv, mid_channels=4*nconv, out_channels=4*nconv, use_dropout=True),
        expansion_block(in_channels=4*nconv, mid_channels=2*nconv, out_channels=2*nconv),
        expansion_block(in_channels=2*nconv, mid_channels=2*nconv, out_channels=2*nconv),
    )
    self.phase_mean_end = nn.Sequential(
        nn.Conv2d(in_channels=2*nconv, out_channels=1, kernel_size=3, stride=1, padding=1),
        nn.Tanh()
    )
    self.phase_log_sigma = nn.Sequential(
        nn.Conv2d(in_channels=2*nconv, out_channels=1, kernel_size=3, stride=1, padding=1)
    )

  def forward(self, 
              x: torch.tensor)->tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """
    Takes input diffraction pattern image, passes through encoder 
    and both decoder branches to return the mean and aleatoric std 
    for the amplitude and phase.
    Args:
      x: input diffraction pattern patch.
    Returns:
      tuple of tensors corresponding to amplitude mean and log sigma, 
      phase mean and log sigma predictions.
    """
    encoded = self.encoder(x)
    amps_decoded = self.amplitude_decoder(encoded)
    amps_mean = self.amplitude_mean_end(amps_decoded)
    amps_logsigma = torch.clamp(self.amplitude_log_sigma(amps_decoded), min=-8.0, max=1.0)
    phis_decoded = self.phase_decoder(encoded)
    phis_mean = self.phase_mean_end(phis_decoded)
    phis_logsigma = torch.clamp(self.phase_log_sigma(phis_decoded), min=-8.0, max=1.0)
    phis_mean = phis_mean * torch.pi
    return amps_mean, amps_logsigma, phis_mean, phis_logsigma

  def train_step(self, 
                 ft_images: torch.tensor, 
                 amps: torch.tensor, 
                 phis: torch.tensor)->tuple:
    """
    Takes a batch of diffraction pattern images, along with the target amplitude
    and phase fields. Returns the NLL loss (the differentiable loss for the PNN) 
    and the L1 loss (a metric) for both the amplitude and phase fields.
    Args:
      ft_images: batch of diffraction patterns.
      amps: target values of amplitude.
      phis: target values of phase.
    Returns:
      A tuple of NLL losses for the amplitude and phase, 
      L1 metric for amplitude and phase. 
    """
    amps_mean, amps_logsigma, phis_mean, phis_logsigma = self(ft_images)
    amp_loss = F.gaussian_nll_loss(amps_mean, amps, amps_logsigma.exp().square()) #input, target, var
    phi_loss = F.gaussian_nll_loss(phis_mean, phis, phis_logsigma.exp().square())
    amp_metric = F.l1_loss(amps_mean, amps)
    phi_metric = F.l1_loss(phis_mean, phis)
    return amp_loss, phi_loss, amp_metric, phi_metric

  def eval_step(self, 
                ft_images: torch.tensor, 
                amps: torch.tensor, 
                phis: torch.tensor)->tuple:
    """
    Takes a batch of diffraction pattern images, along with the target amplitude
    and phase fields. Returns the NLL loss (the differentiable loss for the PNN) 
    and the L1 loss (a metric) for both the amplitude and phase fields.
    Args:
      ft_images: batch of diffraction patterns.
      amps: target values of amplitude.
      phis: target values of phase.
    Returns:
      A tuple of NLL losses for the amplitude and phase, 
      L1 metric for amplitude and phase. 
    """
    amps_mean, amps_logsigma, phis_mean, phis_logsigma = self(ft_images)
    amp_loss = F.gaussian_nll_loss(amps_mean, amps, amps_logsigma.exp().square()) #input, target, var
    phi_loss = F.gaussian_nll_loss(phis_mean, phis, phis_logsigma.exp().square())
    amp_metric = F.l1_loss(amps_mean, amps)
    phi_metric = F.l1_loss(phis_mean, phis)
    return amp_loss, phi_loss, amp_metric, phi_metric


class PtychoNNBase(nn.Module):
  """
  Defines the deterministic version of the PtychoNN model.
  Carries out Ptychographic reconstruction, by mapping Fourier transformed images 
  to the corresponding intensity and phase maps.
  
  Attributes:
    nconv: number of feature maps from the first conv layer.
  """
  def __init__(self, nconv: int=32, **kwargs):
    super().__init__(**kwargs)
    self.encoder = nn.Sequential(
        contraction_block(in_channels=1, mid_channels=nconv, out_channels=nconv),
        contraction_block(in_channels=nconv, mid_channels=2*nconv, out_channels=2*nconv),
        contraction_block(in_channels=2*nconv, mid_channels=4*nconv, out_channels=4*nconv)
    )
    self.amplitude_decoder = nn.Sequential(
        expansion_block(in_channels=4*nconv, mid_channels=4*nconv, out_channels=4*nconv),
        expansion_block(in_channels=4*nconv, mid_channels=2*nconv, out_channels=2*nconv),
        expansion_block(in_channels=2*nconv, mid_channels=2*nconv, out_channels=2*nconv),
        nn.Conv2d(in_channels=2*nconv, out_channels=1, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid()
    )
    self.phase_decoder = nn.Sequential(
        expansion_block(in_channels=4*nconv, mid_channels=4*nconv, out_channels=4*nconv),
        expansion_block(in_channels=4*nconv, mid_channels=2*nconv, out_channels=2*nconv),
        expansion_block(in_channels=2*nconv, mid_channels=2*nconv, out_channels=2*nconv),
        nn.Conv2d(in_channels=2*nconv, out_channels=1, kernel_size=3, stride=1, padding=1),
        nn.Tanh()
    )

  def forward(self, x):
    encoded = self.encoder(x)
    amps = self.amplitude_decoder(encoded)
    phis = self.phase_decoder(encoded)
    phis = phis * torch.pi
    return amps, phis




class PtychoNN(nn.Module):
  """
  Defines the deterministic version of the PtychoNN model
  Attributes:
    nconv: number of feature maps from the first conv layer.
  """
  def __init__(self, nconv: int=32, **kwargs):
    super().__init__(**kwargs)
    self.encoder = nn.Sequential(
        contraction_block(in_channels=1, mid_channels=nconv, out_channels=nconv),
        contraction_block(in_channels=nconv, mid_channels=2*nconv, out_channels=2*nconv),
        contraction_block(in_channels=2*nconv, mid_channels=4*nconv, out_channels=4*nconv)
    )
    self.amplitude_decoder = nn.Sequential(
        expansion_block(in_channels=4*nconv, mid_channels=4*nconv, out_channels=4*nconv),
        expansion_block(in_channels=4*nconv, mid_channels=2*nconv, out_channels=2*nconv),
        expansion_block(in_channels=2*nconv, mid_channels=2*nconv, out_channels=2*nconv),
        nn.Conv2d(in_channels=2*nconv, out_channels=1, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid()
    )
    self.phase_decoder = nn.Sequential(
        expansion_block(in_channels=4*nconv, mid_channels=4*nconv, out_channels=4*nconv),
        expansion_block(in_channels=4*nconv, mid_channels=2*nconv, out_channels=2*nconv),
        expansion_block(in_channels=2*nconv, mid_channels=2*nconv, out_channels=2*nconv),
        nn.Conv2d(in_channels=2*nconv, out_channels=1, kernel_size=3, stride=1, padding=1),
        nn.Tanh()
    )

  def forward(self, x):
    encoded = self.encoder(x)
    amps = self.amplitude_decoder(encoded)
    phis = self.phase_decoder(encoded)
    phis = phis * torch.pi
    return amps, phis

  def train_step(self, ft_images, amps, phis):
    pred_amps, pred_phis = self(ft_images)
    amp_loss = F.mse_loss(pred_amps, amps)
    phi_loss = F.mse_loss(pred_phis, phis)
    amp_metric = F.l1_loss(pred_amps, amps)
    phi_metric = F.l1_loss(pred_phis, phis)

    return amp_loss, phi_loss, amp_metric, phi_metric

  def eval_step(self, ft_images, amps, phis):
    pred_amps, pred_phis = self(ft_images)
    amp_loss = F.mse_loss(pred_amps, amps)
    phi_loss = F.mse_loss(pred_phis, phis)
    amp_metric = F.l1_loss(pred_amps, amps)
    phi_metric = F.l1_loss(pred_phis, phis)

    return amp_loss, phi_loss, amp_metric, phi_metric


class PtychoPNN(nn.Module):
  """
  Defines the Probabilistic Neural Network avatar of the PtychoNN model,
  accounting for aleatoric uncertainty in predictions.

  The network still maps images to the intensity and phase maps, But the returns the 
  log sigma of the predictions as well.
  The loss function is a NLL loss and the metric is an MSE.
  
  Attributes:
    nconv: number of feature maps from the first conv layer.
  """
  def __init__(self, nconv: int=32, **kwargs):
    """
    Initializes the Ptycho PNN.

    Args:
     nconv: the base number of convolutional filters to define the expansion and contraction sections.
    """
    super().__init__(**kwargs)
    self.encoder = nn.Sequential(
        contraction_block(in_channels=1, mid_channels=nconv, out_channels=nconv),
        contraction_block(in_channels=nconv, mid_channels=2*nconv, out_channels=2*nconv),
        contraction_block(in_channels=2*nconv, mid_channels=4*nconv, out_channels=4*nconv)
    )
    self.amplitude_decoder = nn.Sequential(
        expansion_block(in_channels=4*nconv, mid_channels=4*nconv, out_channels=4*nconv),
        expansion_block(in_channels=4*nconv, mid_channels=2*nconv, out_channels=2*nconv),
        expansion_block(in_channels=2*nconv, mid_channels=2*nconv, out_channels=2*nconv),
    )
    self.amplitude_mean_end = nn.Sequential(
        nn.Conv2d(in_channels=2*nconv, out_channels=1, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid()
    )
    self.amplitude_log_sigma = nn.Sequential(
        nn.Conv2d(in_channels=2*nconv, out_channels=1, kernel_size=3, stride=1, padding=1)
    )

    self.phase_decoder = nn.Sequential(
        expansion_block(in_channels=4*nconv, mid_channels=4*nconv, out_channels=4*nconv),
        expansion_block(in_channels=4*nconv, mid_channels=2*nconv, out_channels=2*nconv),
        expansion_block(in_channels=2*nconv, mid_channels=2*nconv, out_channels=2*nconv),
    )
    self.phase_mean_end = nn.Sequential(
        nn.Conv2d(in_channels=2*nconv, out_channels=1, kernel_size=3, stride=1, padding=1),
        nn.Tanh()
    )
    self.phase_log_sigma = nn.Sequential(
        nn.Conv2d(in_channels=2*nconv, out_channels=1, kernel_size=3, stride=1, padding=1)
    )

  def forward(self, x):
    encoded = self.encoder(x)
    amps_decoded = self.amplitude_decoder(encoded)
    amps_mean = self.amplitude_mean_end(amps_decoded)
    amps_logsigma = torch.clamp(self.amplitude_log_sigma(amps_decoded), min=-8.0, max=1.0)
    phis_decoded = self.phase_decoder(encoded)
    phis_mean = self.phase_mean_end(phis_decoded)
    phis_logsigma = torch.clamp(self.phase_log_sigma(phis_decoded), min=-8.0, max=1.0)
    phis_mean = phis_mean * np.pi
    return amps_mean, amps_logsigma, phis_mean, phis_logsigma

  def train_step(self, ft_images, amps, phis):
    amps_mean, amps_logsigma, phis_mean, phis_logsigma = self(ft_images)

    amp_loss = F.gaussian_nll_loss(amps_mean, amps, amps_logsigma.exp().square()) #input, target, var
    phi_loss = F.gaussian_nll_loss(phis_mean, phis, phis_logsigma.exp().square())
    amp_metric = F.l1_loss(amps_mean, amps)
    phi_metric = F.l1_loss(phis_mean, phis)

    return amp_loss, phi_loss, amp_metric, phi_metric

  def eval_step(self, ft_images, amps, phis):
    amps_mean, amps_logsigma, phis_mean, phis_logsigma = self(ft_images)
    amp_loss = F.gaussian_nll_loss(amps_mean, amps, amps_logsigma.exp().square()) #input, target, var
    phi_loss = F.gaussian_nll_loss(phis_mean, phis, phis_logsigma.exp().square())
    amp_metric = F.l1_loss(amps_mean, amps)
    phi_metric = F.l1_loss(phis_mean, phis)

    return amp_loss, phi_loss, amp_metric, phi_metric


class DeepEnsemble(torch.nn.Module):
  """
  Defines a Deep Ensemble wrapper for PtychoPNN models,
  accounting for epistemic and aleatoric uncertainty in predictions.
  The constituent models are pre-trained and the wrapper just defines a 
  forward function for the prediction step from Lakshminarayanan et al (2017).
  
  Attributes:
    models: a list of trained PtychoPNN models to form the ensemble.
  """
  def __init__(self, models: list):
    """
    Initializes the deep ensemble based on the constituent adversirially trained PNNs.

    Args:
      models: List of trained PNNs.
    """
    super().__init__()
    self.models = torch.nn.ModuleList(models)

  def forward(self, x):
    """
    Treats the ensemble predictions as a Mixture of Gaussians. Computes the 
    resultant means and variances accordingly
    """
    amps_mean_preds, amps_var_preds, phi_mean_preds, phi_var_preds = [], [], [], [] 
    for model in self.models:
      amps_mean, amps_logsigma, phis_mean, phis_logsigma = model(x)
      amps_var, phis_var = amps_logsigma.exp().square(), phis_logsigma.exp().square()
      amps_mean_preds.append(amps_mean)
      amps_var_preds.append(amps_var)
      phi_mean_preds.append(phis_mean)
      phi_var_preds.append(phis_var)

    amps_mean_preds = torch.stack(amps_mean_preds) #Shape: [M, 64, 1, 64,, 64]
    amps_var_preds = torch.stack(amps_var_preds)
    phi_mean_preds = torch.stack(phi_mean_preds)
    phi_var_preds = torch.stack(phi_var_preds)
    amps_ens_mean = torch.mean(amps_mean_preds, dim=0) #Shape: [64, 1, 64, 64]
    phi_ens_mean = torch.mean(phi_mean_preds, dim=0)
    amps_ens_var = torch.mean(amps_var_preds + amps_mean_preds.square(), dim=0) - amps_ens_mean.square()
    phi_ens_var = torch.mean(phi_var_preds + phi_mean_preds.square(), dim=0) - phi_ens_mean.square()
    return amps_ens_mean, amps_ens_var, phi_ens_mean, phi_ens_var

