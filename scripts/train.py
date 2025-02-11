from tkinter.tix import Select
import numpy as np
from pyparsing import Word
import torch
import torch.nn as nn
from torch.utils.data import random_split, TensorDataset
from argparse import ArgumentParser
from lightning import Trainer
from lightning.fabric import fabric
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from utils.constants import *
from utils.dataset import AugmentedTonesDataset
from utils.selectmodel import SelectModel
from models.LightningModule import MyLightningModule, LitDataModule

def main(hparams):
  devices = hparams.devices
  accelerator = hparams.accelerator
  model = hparams.model
  predictor = hparams.predictor
  mode = hparams.mode # for train, test, pred, etc

  # Read in data
  X_train = torch.tensor(np.load(CUR_FEATS_DIR / 'train_features.npy')).unsqueeze(1) # [batch, channels, freq, time]
  X_test = torch.tensor(np.load(CUR_FEATS_DIR / 'test_features.npy')).unsqueeze(1)
  X_val = torch.tensor(np.load(CUR_FEATS_DIR / 'val_features.npy')).unsqueeze(1)

  y_train = torch.tensor(np.load(CUR_FEATS_DIR / 'train_labels.npy'))
  y_test = torch.tensor(np.load(CUR_FEATS_DIR / 'test_labels.npy'))
  y_val = torch.tensor(np.load(CUR_FEATS_DIR / 'val_labels.npy'))

  # Datasets -> lightning loader
  train_dataset = AugmentedTonesDataset(X_train, y_train, augment_prob=0.5)
  test_dataset = AugmentedTonesDataset(X_test, y_test, augment_prob=0.5)
  val_dataset = AugmentedTonesDataset(X_val, y_val, augment_prob=0.5)

  lit_dataloader = LitDataModule(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)

  training_model = SelectModel(model, y_train)

  # lightning model & trainer
  lit_model = MyLightningModule(model=training_model)
  
  earlystop_callback = EarlyStopping(monitor="val_acc", mode="min", patience=3)
  checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="min",
    save_top_k=1,
    filename="best-checkpoint",
  )

  trainer = Trainer(max_epochs=10, 
                    devices=devices, 
                    accelerator=accelerator,
                    callbacks=[earlystop_callback, checkpoint_callback],
                    fast_dev_run=7, # for testing
                    # profiler="simple", # also for testing/debugging
  )
  trainer.fit(lit_model, datamodule=lit_dataloader)
  
  # auto uses best weights since called right after .fit
  # trainer.test(datamodule=lit_dataloader, ckpt_path='best') 

  # predict
  # predictions = trainer.predict(lit_model, datamodule=lit_dataloader)

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--accelerator", default='gpu')
  parser.add_argument("--devices", default='1')
  parser.add_argument("--model", default='mtl', choices=["mtl", "resnet"])
  parser.add_argument("--predictor", default='tone', choices=["tone", "word"])
  parser.add_argument("--mode", default="train", choices=["train", "test", "predict"])
  args = parser.parse_args()

  main(args)
