import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def EFIM(model, data_loader):
  # Initialize EFIM dictionary with zeros for each learnable parameter
  efim = {name: torch.zeros_like(param, device=param.device)
          for name, param in model.named_parameters() if param.requires_grad}

  model.to(DEVICE)
  model.train()

  # Accumulate the squared gradients over all batches
  for inputs, targets in data_loader:
      inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

      model.zero_grad()
      outputs = model(inputs)
      loss = cross_entropy(outputs, targets)
      loss.backward()


      for name, param in model.named_parameters():
          if param.grad is not None:
             efim[name] += param.grad.data ** 2
             #efim[name] += param.grad

  # Average over the number of data points
  num_data_points = len(data_loader.dataset)
  for name in efim:
      efim[name] /= num_data_points


  return efim

def get_parameters_with_small_norm(efim, threshold):
    """Get parameters with a norm smaller than the threshold."""
    small_norm_params = []
    large_norm_params = []
    for name, value_tensor in efim.items():
        if value_tensor.norm() < threshold:
            small_norm_params.append(name)
        else:
          large_norm_params.append(name)
    return small_norm_params, large_norm_params

def freeze_parameters(model, list_parameters):
  for name, param in model.named_parameters():
    if name in list_parameters:
      param.requires_grad = False

def count_frozen_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)

def params_below_threshold(aggregatedEFIM, threshold):
    params = []
    for param_name, value in aggregatedEFIM:
       if value < threshold:
          params.append(param_name)
    return params
          

def aggregatedEFIM(efim):
   aggregated_efim = {name: efim[name].sum().item() for name in efim}
   aggregated_efim = sorted(aggregated_efim.items(), key=lambda x: x[1], reverse=True)
   return aggregated_efim



