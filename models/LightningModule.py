import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from torchmetrics import Accuracy

class MyLightningModule(L.LightningModule):
  def __init__(self, model, learning_rate=1e-3, batch_size=32, num_classes=4):
    super().__init__()
    self.save_hyperparameters(ignore=[model]) # auto saves all hyper params into hparams
    self.model = model
    self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    outputs = self.model(x)
    train_loss = F.cross_entropy(outputs, y)
    acc = self.train_accuracy(outputs, y)
    
    self.log("train_loss", train_loss, prog_bar=True)
    self.log("train_acc", acc, prog_bar=True)
    return train_loss
  
  def test_step(self, batch, batch_idx):
    x, y = batch
    outputs = self.model(x)
    test_loss = F.cross_entropy(outputs, y)
    acc = self.train_accuracy(outputs, y)

    self.log("test_loss", test_loss, prog_bar=True)
    self.log("test_acc", acc, prog_bar=True)

  def validation_step(self, batch, batch_idx):
    x, y = batch
    outputs = self.model(x)
    val_loss = F.cross_entropy(outputs, y)
    acc = self.train_accuracy(outputs, y)

    self.log("val_loss", val_loss, prog_bar=True)
    self.log("val_acc", acc, prog_bar=True)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    return optimizer
  
  def predict_step(self, batch):
    x, _ = batch
    return self(x)


class LitDataModule(L.LightningDataModule):
  def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32):
    super().__init__()
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.test_dataset = test_dataset
    self.batch_size = batch_size

  def train_dataloader(self):
    return DataLoader(self.train_dataset, 
                      batch_size=self.batch_size, 
                      shuffle=True, 
                      num_workers=4, 
                      persistent_workers=True)
  
  def test_dataloader(self):
    return DataLoader(self.test_dataset, 
                      batch_size=self.batch_size, 
                      shuffle=False, 
                      num_workers=4, 
                      persistent_workers=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, 
                      batch_size=self.batch_size, 
                      shuffle=False, 
                      num_workers=4, 
                      persistent_workers=True)
    
    
