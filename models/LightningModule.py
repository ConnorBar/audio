import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L

class MyLightningModule(L.LightningModule):
  def __init__(self, model, learning_rate=1e-3, batch_size=32, num_classes=4):
    super().__init__()
    self.save_hyperparameters(ignore=[model]) # auto saves all hyper params into hparams
    self.model = model

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    predictions = self.model(x)

    train_loss = self.model.compute_loss(predictions, y)
    acc = self.model.compute_metrics(predictions, y, metric='acc')
    f1 = self.model.compute_metrics(predictions, y, metric='f1')

    self.log("train_loss", train_loss, prog_bar=True)
    self.log("train_acc", acc, prog_bar=True)
    self.log("train_f1", f1, prog_bar=True)

    init_acc, final_acc, tone_acc, sanity_acc = self.model.compute_metrics(predictions, y, metric='acc', average=False)

    self.log("train_init_acc", init_acc)
    self.log("train_final_acc", final_acc)
    self.log("train_tone_acc", tone_acc)
    self.log("train_sanity_acc", sanity_acc)

    return train_loss
  
  def test_step(self, batch, batch_idx):
    x, y = batch
    predictions = self.model(x)

    test_loss = self.model.compute_loss(predictions, y)
    acc = self.model.compute_metrics(predictions, y, metric='acc')
    f1 = self.model.compute_metrics(predictions, y, metric='f1')

    self.log("test_loss", test_loss, prog_bar=True)
    self.log("test_acc", acc, prog_bar=True)
    self.log("test_f1", f1, prog_bar=True)

    init_acc, final_acc, tone_acc, sanity_acc = self.model.compute_metrics(predictions, y, metric='acc', average=False)

    self.log("test_init_acc", init_acc)
    self.log("test_final_acc", final_acc)
    self.log("test_tone_acc", tone_acc)
    self.log("test_sanity_acc", sanity_acc)

  def validation_step(self, batch, batch_idx):
    x, y = batch
    predictions = self.model(x)

    val_loss = self.model.compute_loss(predictions, y)
    acc = self.model.compute_metrics(predictions, y, metric='acc')
    f1 = self.model.compute_metrics(predictions, y, metric='f1')

    self.log("val_loss", val_loss, prog_bar=True)
    self.log("val_acc", acc, prog_bar=True)
    self.log("val_f1", f1, prog_bar=True)

    init_acc, final_acc, tone_acc, sanity_acc = self.model.compute_metrics(predictions, y, metric='acc', average=False)

    self.log("val_init_acc", init_acc)
    self.log("val_final_acc", final_acc)
    self.log("val_tone_acc", tone_acc)
    self.log("val_sanity_acc", sanity_acc)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    return optimizer
  
  # not using this yet but i think that i will need to change how i handle the output from here
  # since i am doing mtl
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
    
    
