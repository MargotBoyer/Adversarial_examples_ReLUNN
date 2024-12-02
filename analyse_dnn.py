import torch.nn as nn
import torch
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

def count_parameters(model: torch.nn.Module) -> int:
  """ Counts the number of trainable parameters of a module
  :param model: model that contains the parameters to count
  :returns: the number of parameters in the model
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualize_layers(model: torch.nn.Module):
    """ Visualize layers and components and
    number of parameters for each """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

def summary_model(model: torch.nn.Module):
    print('Number of parameters:', count_parameters(model))
    print('Layers:')
    visualize_layers(model)


