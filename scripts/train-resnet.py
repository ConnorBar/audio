import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from lightning import Trainer

from utils.constants import *
from utils.dataset import AugmentedTonesDataset
from models.ResNet import ResNet18
from models.LightningModule import LitResNet, LitDataModule

def main(hparams):
  # Read in data
  X_train = torch.tensor(np.load(DATA_DIR / 'train_features.npy')).unsqueeze(1)
  X_test = torch.tensor(np.load(DATA_DIR / 'test_features.npy')).unsqueeze(1)
  y_train = torch.tensor(np.load(DATA_DIR / 'train_labels.npy'))
  y_test = torch.tensor(np.load(DATA_DIR / 'test_labels.npy'))

  # Data loaders 
  train_dataset = AugmentedTonesDataset(X_train, y_train, augment_prob=0.5)
  test_dataset = AugmentedTonesDataset(X_test, y_test, augment_prob=0.5)

  lit_dataloader = LitDataModule(train_dataset=train_dataset, val_dataset=test_dataset)

  num_classes = len(torch.unique(y_train))
  model = ResNet18(num_classes=num_classes)

  # lightning model & trainer
  lit_model = LitResNet(model=model)

  trainer = Trainer(max_epochs=10, devices=hparams.devices, accelerator=hparams.accelerator)
  trainer.fit(lit_model, datamodule=lit_dataloader)

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--accelerator", default=None)
  parser.add_argument("--devices", default=None)
  args = parser.parse_args()

  main(args)
