import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

class ChineseTonesDataset(Dataset):
  def __init__(self, features, labels):
    self.features = features
    self.labels = labels
    
  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]

# consider adding batch normalization
class ChineseTonesCNN(nn.Module):
  def __init__(self, num_classes):
    super(ChineseTonesCNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1)  # Input channels = 1 (MFCCs treated as images)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
    self.fc1 = nn.Linear(32 * (13 // 4) * (237 // 4), 128)  # Adjust input size based on pooling
    self.fc2 = nn.Linear(128, num_classes)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)
  
  def forward(self, x):
    x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, 13, 237]
    x = self.relu(self.conv1(x))
    x = self.pool(x)
    x = self.relu(self.conv2(x))
    x = self.pool(x)
    x = x.view(x.size(0), -1)  # Flatten
    x = self.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x
