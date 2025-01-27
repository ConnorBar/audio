import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np

class AugmentedTonesDataset(Dataset):
  def __init__(self, features, labels, augment_prob=0.5, time_mask_param=30, freq_mask_param=2):
    self.features = features
    self.labels = labels
    self.augment_prob = augment_prob
    
    # inits time & freq masking
    self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param)
    self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param)
    
  def __len__(self):
    return len(self.labels)
  
  def specAugment(self, x):
    # relies on (batch, channel, frequency, time) or (channel, frequency, time) - (1, 13, 237)
    x = self.time_masking(x)
    x = self.freq_masking(x)
    
    return x

  def __getitem__(self, idx):
    feats = self.features[idx]
    
    # if torch.rand(1).item() < self.augment_prob:
    #   feats = self.specAugment(feats)

    return feats, self.labels[idx]

    
def GetDevice():
  if torch.backends.mps.is_available():
    device = torch.device("mps")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  return device


