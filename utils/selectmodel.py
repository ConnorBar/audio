from models.ResNet import ResNet18
from models.BasicCNN import ChineseTonesCNN

def SelectModel(model):
  if model == "resnet":
    return ResNet18
  elif model == "basic":
    return ChineseTonesCNN
  else:
    raise ValueError(f"Unknown model name: {model}")

