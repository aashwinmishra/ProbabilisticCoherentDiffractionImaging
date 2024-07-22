"""
Classes for models to be trained.
"""
import torch
import torch.nn as nn
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
      nn.ReLU(),
      nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
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
      nn.ReLU(),
      nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.ReLU(),
      nn.Upsample(scale_factor=upsamling_factor, mode='bilinear')
      )
  

class PtychoNNBase(nn.Module):
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

