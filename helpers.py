import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
import numpy as np
import torch.nn.functional as F

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

#this is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
#Specific for mix-max problems

def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
  beta = 0.1
  return (1 - beta) * averaged_model_parameter + beta * model_parameter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
    

def adjust_learning_rate(epoch, lr_decay_rate, sgda_learning_rate , lr_decay_epochs , optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    new_lr = sgda_learning_rate
    if steps > 0:
        new_lr = sgda_learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr


def param_dist(model, swa_model, p):
    #This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist

def train_distill(epoch, train_loader, module_list, swa_model, criterion_list, optimizer, distill, gamma, alpha, beta, smoothing, split, quiet=True, bt=False, rf_t=False):
    """One epoch distillation"""

    #print('train_distill')

    # set modules as train()
    for module in module_list:
        module.train()

    if rf_t:
      module_list[1].eval()


    # set teacher as eval()
    module_list[-1].eval()



    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    if rf_t:
      print('getting rft')
      model_rf_t= module_list[1]



    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()

    for idx, data in enumerate(train_loader):
        if distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target = data

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if distill in ['crd']:
                contrast_idx = contrast_idx.cuda()
                index = index.cuda()

        # ===================forward=====================

        logit_s = model_s(input)



        with torch.no_grad():
            logit_t = model_t(input)

        if rf_t and split == "maximize":
          logit_t = model_rf_t(input)




        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)


        # other kd beyond KL divergence
        if distill == 'kd':
            loss_kd = 0

        else:
            raise NotImplementedError(distill)

        # Here I eliminated the other losses

        if split == "minimize":
            loss = gamma * loss_cls + alpha * loss_div + beta * loss_kd

        elif split == "maximize":
            loss = -loss_div

        loss = loss + param_dist(model_s, swa_model, smoothing)

        if split == "minimize":

            losses.update(loss.item(), input.size(0))


        elif split == "maximize":
            kd_losses.update(loss.item(), input.size(0))


        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


    if split == "minimize":
        return losses.avg
    else:
        return kd_losses.avg




