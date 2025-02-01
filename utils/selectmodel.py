import torch

from models.ResNet import ResNet18
from models.BasicCNN import ChineseTonesCNN
from models.MultiTask import MTLNetwork

def SelectModel(model, y_train):
  if model == "mtl":
    num_initials = len(torch.unique(y_train[:, 0]))
    num_finals = len(torch.unique(y_train[:, 1]))
    num_tones = len(torch.unique(y_train[:, 2]))

    return MTLNetwork(num_initials=num_initials, num_finals=num_finals, num_tones=num_tones)
  elif model == "resnet":
    num_classes = len(torch.unique(y_train)) 
    return ResNet18(num_classes=num_classes)
  elif model == "basic":
    num_classes = len(torch.unique(y_train)) 
    return ChineseTonesCNN(num_classes=num_classes)
  else:
    raise ValueError(f"Unknown model name: {model}")

