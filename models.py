"""
Classes definitions and constructor functions for the models. This includes:
a) A Basic PtychoNN Deterministic model,
b) A PtychoPNN Probabilistic model,
c) A Deep Ensemble model (Lakshminarayanan et al, "Simple and scalable predictive uncertainty estimation using deep ensembles", Neurips (2017)"
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
                      pool_factor:int=2) -> nn.Module:
  """
  Creates a constituent Conv block for the encoder section of the ptychoNN model.
  Consists of Conv-Relu-Conv-Relu-Maxpool layers.
  Args:
    in_channels: Input channels to the block
    mid_channels: intermediate channels (ie, output of first conv layer and input channels to the second)
    out_channels: Final number of output channels from the conv block
    kernel_size: Uniform kernel size across both conv layers in the block
    stride: Uniform stride across both conv layers in the block
    padding: Uniform padding across both conv layers in the block
    pool_factor: Kernel size of the square max pool
  Returns:
    nn.Sequential container of modules.
  """
  return nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(mid_channels),
      nn.ReLU(),
      nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.MaxPool2d(pool_factor)
  )


def expansion_block(in_channels: int,
                    mid_channels: int,
                    out_channels: int,
                    kernel_size: int=3,
                    stride: int=1,
                    padding: int=1,
                    upsamling_factor:int=2) -> nn.Module:
    """
  Creates a constituent Conv block for the decoder sections of the ptychoNN model.
  Consists of Conv-Relu-Conv-Relu-Upsample layers.
  Args:
    in_channels: Input channels to the block
    mid_channels: intermediate channels (ie, output of first conv layer and input channels to the second)
    out_channels: Final number of output channels from the conv block
    kernel_size: Uniform kernel size across both conv layers in the block
    stride: Uniform stride across both conv layers in the block
    padding: Uniform padding across both conv layers in the block
    upsampling_factor: Scale factor for the upsampling layer
  Returns:
    nn.Sequential container of modules.
  """
    return nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(mid_channels),
      nn.ReLU(),
      nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Upsample(scale_factor=upsamling_factor, mode='bilinear')
      )


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
    phis = phis * np.pi
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
    phis = phis * np.pi
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

  The network still amps images to the intensity and phase maps, But the returns the 
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
    amps_logsigma = self.amplitude_log_sigma(amps_decoded)
    phis_decoded = self.phase_decoder(encoded)
    phis_mean = self.phase_mean_end(phis_decoded)
    phis_logsigma = self.phase_log_sigma(phis_decoded)
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

