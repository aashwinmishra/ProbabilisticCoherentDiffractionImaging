import torch
import torch.nn as nn


class Gaussian:
  def __init__(self, mu: float=0.0, rho: float=1.0):
    self.mu = mu 
    self.rho = rho
    self.normal = torch.distributions.Normal(0, 1.0)

  @property
  def sigma(self):
    return torch.log1p(torch.exp(self.rho))

  def sample(self):
    epsilon = self.normal.sample(self.rho.size())
    return self.mu + self.sigma * epsilon 

  def log_prob(self, x: torch.tensor):
    return ( -0.5 * torch.log(2 * torch.pi * self.sigma**2) - 0.5 * ((x - self.mu) / self.sigma)**2).sum()


class BBBLayer(nn.Module):
  def __init__(self,
               in_dim: int,
               out_dim: int):
    super().__init__()
    self.weight_mu = nn.Parameter(torch.Tensor(in_dim, out_dim).uniform_(-0.2, 0.2))
    self.weight_rho = nn.Parameter(torch.Tensor(in_dim, out_dim).uniform_(-5, -4))
    self.weight = Gaussian(self.weight_mu, self.weight_rho)

    self.bias_mu = nn.Parameter(torch.Tensor(out_dim,).uniform_(-0.2, 0.2))
    self.bias_rho = nn.Parameter(torch.Tensor(out_dim,).uniform_(-5, -4))
    self.bias = Gaussian(self.bias_mu, self.bias_rho)

    self.weight_prior = Gaussian(torch.zeros_like(self.weight_mu), torch.ones_like(self.weight_rho)*0.542)
    self.bias_prior = Gaussian(torch.zeros_like(self.bias_mu), torch.ones_like(self.bias_rho)*0.542)

    self.log_prior = 0.0
    self.log_variational_posterior = 0.0

  def forward(self, x, sample=False):
    if self.training or sample:
      weight = self.weight.sample()
      bias = self.bias.sample()
    else:
      weight = self.weight.mu
      bias = self.bias.mu
    if self.training:
      self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
      self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
    else:
      self.log_prior, self.log_variational_posterior = 0.0, 0.0
    return x @ weight + bias


class BBBFinalLayer(nn.Module):
  def __init__(self,
               in_dim: int,
               out_dim: int):
    super().__init__()
    self.weight_mu = nn.Parameter(torch.Tensor(in_dim, out_dim * 2).uniform_(-0.2, 0.2))
    self.weight_rho = nn.Parameter(torch.Tensor(in_dim, out_dim * 2).uniform_(-5, -4))
    self.weight = Gaussian(self.weight_mu, self.weight_rho)

    self.bias_mu = nn.Parameter(torch.Tensor(out_dim * 2,).uniform_(-0.2, 0.2))
    self.bias_rho = nn.Parameter(torch.Tensor(out_dim * 2,).uniform_(-5, -4))
    self.bias = Gaussian(self.bias_mu, self.bias_rho)

    self.weight_prior = Gaussian(torch.zeros_like(self.weight_mu), torch.ones_like(self.weight_rho)*0.542)
    self.bias_prior = Gaussian(torch.zeros_like(self.bias_mu), torch.ones_like(self.bias_rho)*0.542)

    self.log_prior = 0.0
    self.log_variational_posterior = 0.0

  def forward(self, x, sample=False):
    if self.training or sample:
      weight = self.weight.sample()
      bias = self.bias.sample()
    else:
      weight = self.weight.mu
      bias = self.bias.mu
    if self.training:
      self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
      self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
    else:
      self.log_prior, self.log_variational_posterior = 0.0, 0.0
    out = x @ weight + bias
    return out[:, :out_dim], out[:, out_dim:]


class BBB_FC(nn.Module):
  def __init__(self, 
               dims: list, 
               kl_weight: float):
    super().__init__()
    self.depth = len(dims)
    self.kl_weight = kl_weight
    layers = [BBBLayer(dims[i], dims[i+1]) for i in range(len(dims) -2)]
    layers.append(BBBFinalLayer(dims[-2], dims[-1]))
    self.layers = nn.ModuleList(layers)


  def forward(self, x, sample=False):
    for i in range(self.depth - 1):
      x = F.relu(self.layers[i](x, sample))
    return self.layers[-1](x, sample)

  def log_prior(self):
    log_prior = 0.0
    for layer in self.layers:
      log_prior += layer.log_prior
    return log_prior

  def log_variational_posterior(self):
    log_posterior = 0.0
    for layer in self.layers:
      log_posterior += layer.log_variational_posterior
    return log_posterior 

  def elbo(self, x, target, samples, kl_weight):
    batch_size = x.shape[0]
    outputs = torch.zeros((samples, batch_size, 1))
    log_sigmas = torch.zeros((samples, batch_size, 1))
    log_priors = torch.zeros(samples)
    log_variational_posteriors = torch.zeros(samples)
    for i in range(samples):
      outputs[i], log_sigmas[i] = self(x, sample=True)
      log_priors[i] = self.log_prior()
      log_variational_posteriors[i] = self.log_variational_posterior()
    log_prior = log_priors.mean()
    log_variational_posteriror = log_variational_posteriors.mean()
    vars = log_sigmas.exp().square()
    NLL = F.gaussian_nll_loss(outputs, target.unsqueeze_(0).expand(outputs.shape[0], -1, -1), vars, reduction='mean')
    loss = (log_variational_posteriror - log_prior) * self.kl_weight + NLL
    return loss


class BBB_neuron(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight_mu = nn.Parameter(torch.Tensor(1,).uniform_(-0.2, 0.2))
    self.weight_rho = nn.Parameter(torch.Tensor(1,).uniform_(-5, -4))
    self.weight = Gaussian(self.weight_mu, self.weight_rho)

    self.bias_mu = nn.Parameter(torch.Tensor(1,).uniform_(-0.2, 0.2))
    self.bias_rho = nn.Parameter(torch.Tensor(1,).uniform_(-5, -4))
    self.bias = Gaussian(self.bias_mu, self.bias_rho)

    self.weight_aleatoric = nn.Parameter(torch.Tensor(1,).uniform_(-0.1, 0.1))
    self.bias_aleatoric = nn.Parameter(torch.Tensor(1,).uniform_(-0.1, 0.1))

    self.weight_prior = Gaussian(torch.zeros_like(self.weight_mu), torch.ones_like(self.weight_rho)*0.542)
    self.bias_prior = Gaussian(torch.zeros_like(self.bias_mu), torch.ones_like(self.bias_rho)*0.542)

    self.log_prior = 0.0
    self.log_variational_posterior = 0.0

  def forward(self, x, sample=False):
    if self.training or sample:
      weight = self.weight.sample()
      bias = self.bias.sample()
    else:
      weight = self.weight_mu 
      bias = self.bias_mu
    if self.training:
      self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias) 
      self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)   
    else:
      self.log_variational_posterior, self.log_prior = 0.0, 0.0
    return x * weight + bias, x * self.weight_aleatoric + self.bias_aleatoric

  def elbo(self, x, target, samples, kl_weight):
    outputs = torch.zeros((samples, batch_size, 1))
    log_sigmas = torch.zeros((samples, batch_size, 1))
    log_priors = torch.zeros(samples)
    log_variational_posteriors = torch.zeros(samples)
    for i in range(samples):
      outputs[i], log_sigmas[i] = self(x, sample=True)
      log_priors[i] = self.log_prior 
      log_variational_posteriors[i] = self.log_variational_posterior
    log_prior = log_priors.mean()
    log_variational_posteriror = log_variational_posteriors.mean()
    vars = log_sigmas.exp().square()
    NLL = F.gaussian_nll_loss(outputs, target.unsqueeze_(0).expand(outputs.shape[0], -1, -1), vars, reduction='mean')
    loss = (log_variational_posteriror - log_prior) * kl_weight + NLL
    return loss, log_variational_posteriror, log_prior, NLL 

