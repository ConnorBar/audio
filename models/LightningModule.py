import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L


class LitResNet(L.LightningModule):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    outputs = self.model(x)
    loss = F.cross_entropy(outputs, y)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer


class LitDataModule(L.LightningDataModule):
  def __init__(self, train_dataset, val_dataset, batch_size=32):
    super().__init__()
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.batch_size = batch_size

  def train_dataloader(self):
    return DataLoader(self.train_dataset, 
                      batch_size=self.batch_size, 
                      shuffle=True, 
                      num_workers=4, 
                      persistent_workers=True)
  
  def val_dataloader(self):
    return DataLoader(self.val_dataset, 
                      batch_size=self.batch_size, 
                      shuffle=True, 
                      num_workers=4, 
                      persistent_workers=True)
    
    
