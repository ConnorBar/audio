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
  model = SelectModel(hparams.model)
  predictor = hparams.predictor
  mode = hparams.mode # for train, test, pred, etc

  # Read in data - TODO add fxn to make this look nicer
  X_train = torch.tensor(np.load(DATA_DIR / 'train_features.npy')).unsqueeze(1)
  X_test = torch.tensor(np.load(DATA_DIR / 'test_features.npy')).unsqueeze(1)
  y_train_tone = torch.tensor(np.load(DATA_DIR / 'train_tone_labels.npy'))
  y_train_word = torch.tensor(np.load(DATA_DIR / 'train_word_labels.npy'))
  y_test_tone = torch.tensor(np.load(DATA_DIR / 'test_tone_labels.npy'))
  y_test_word = torch.tensor(np.load(DATA_DIR / 'test_word_labels.npy'))

  # make val set
  train_dataset = TensorDataset(X_train, y_train_tone)

  train_set_size = int(len(train_dataset) * 0.8)
  valid_set_size = len(train_dataset) - train_set_size
  
  seed = torch.Generator().manual_seed(RANDOM_SEED)
  train_set, valid_set = random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)

  # Datasets -> lightning loader
  train_dataset = AugmentedTonesDataset(train_set.dataset.tensors[0], train_set.dataset.tensors[1], augment_prob=0.5)
  val_dataset = AugmentedTonesDataset(valid_set.dataset.tensors[0], valid_set.dataset.tensors[1], augment_prob=0.5)
  test_dataset_tone = AugmentedTonesDataset(X_test, y_test_tone, augment_prob=0.5)
  test_dataset_word = AugmentedTonesDataset(X_test, y_test_word, augment_prob=0.5)

  lit_dataloader = LitDataModule(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset_tone)

  num_tones = len(torch.unique(y_train_tone))
  tone_model = model(num_classes=num_tones)

  num_words = len(torch.unique(y_train_word))
  word_model = model(num_classes=num_words)

  # will do tone, consonant, vowel
  if predictor == "tone":
    num_classes = num_tones
    training_model = tone_model
  elif predictor == "word":
    num_classes = num_words 
    training_model = word_model

  # lightning model & trainer
  lit_model = MyLightningModule(model=training_model, num_classes=num_classes)
  
  earlystop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=3)
  checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
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
  parser.add_argument("--model", default='resnet', choices=["resnet"])
  parser.add_argument("--predictor", default='tone', choices=["tone", "word"])
  parser.add_argument("--mode", default="train", choices=["train", "test", "predict"])
  args = parser.parse_args()

  main(args)
