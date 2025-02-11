from random import shuffle
from typing import List, Dict, Tuple
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import joblib
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F


from utils.constants import *
from utils.selectmodel import GetDevice

def generate_labels(df: pd.DataFrame) -> List[Tuple[int, int, int, int]]:
  """encodes the labels & saves encoders to decode at prediction time

  Args:
      df (pd.DataFrame): pd df w labels

  Returns:
      Tuple[List[int], List[int], List[int], List[int]]: tuple of initial, final, tone, sanity
  """
  initials = df['initial'].to_list()
  finals = df['final'].to_list()
  tones = df['tone'].to_list()
  sanity = df['sanity'].to_list() # dont need an encoder, just binary classification

  initial_encoder = LabelEncoder()
  final_encoder = LabelEncoder()
  tone_encoder = LabelEncoder()

  initials_encoded = initial_encoder.fit_transform(initials)
  finals_encoded = final_encoder.fit_transform(finals)
  tones_encoded = tone_encoder.fit_transform(tones)

  # used to decode at inference time
  joblib.dump(initial_encoder, PKL_DATA_DIR / "initial_encoder.pkl") 
  joblib.dump(final_encoder, PKL_DATA_DIR / "final_encoder.pkl")
  joblib.dump(tone_encoder, PKL_DATA_DIR / "tone_encoder.pkl")

  # labels = list(zip(initials_encoded, finals_encoded, tones_encoded))
  labels = list(zip(initials_encoded, finals_encoded, tones_encoded, sanity))

  return labels
  
def assign_augmentations(X_train: List[str], y_train: Tuple[List[int], List[List[int]]], augments_needed: Dict[int, int]) -> List[Tuple[str, List[Tuple[int, int, int]], bool]]:  
  """ 
  resamples the req number to get target dist from each class and labels the resampled for augmentation

  Args:
      X_train (List): train data
      y_train (Tuple[]): tuple of tones for augment purposes, and tuple of labels
      augments_needed (Dict): mapping of class labels to the number of augmentations needed

  Returns:
      Tuple[List, List, List]: modifed X_train, y_train & indicator if sample will be augmented
  """
  y_tones, y_mtl = zip(*y_train)
  grouped_examples = {clss: [] for clss in augments_needed.keys()}
  for x, tone, y in zip(X_train, y_tones, y_mtl):
    grouped_examples[tone].append((x, y))
    
  X_train_new = []
  y_mtl_new = []
  will_augment = []

  # resampling each class, adding og & resampled to new X & new y & marking resampled ones for augmentation
  for clss, n_samples in augments_needed.items():
    original_samples = grouped_examples[clss]
    augment_samples = resample(original_samples, replace=True, n_samples=n_samples, random_state=RANDOM_SEED)

    X_train_new.extend([x for x, _ in original_samples] + [x for x, _ in augment_samples])
    y_mtl_new.extend([y_one_hot for _, y_one_hot in original_samples] + [y_one_hot for _, y_one_hot in augment_samples])
    will_augment.extend([False] * len(original_samples) + [True] * len(augment_samples))

  # shuffle to avoid any issues training later - sorted data is iffy
  combined_data = list(zip(X_train_new, y_mtl_new, will_augment))
  shuffle(combined_data)
    
  return combined_data

def preemphasis(waveform: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    """PyTorch implementation of preemphasis filter"""
    return torch.cat((waveform[:1], waveform[1:] - coeff * waveform[:-1]))


def augment_raw_audio(audio, sr=16000, aug_threshold = 0.5):
  """randomly applies atleast one of the augmentations 

  Args:
      audio (librosa wav obj): raw wav
      sr (number): sampling rate, 
      aug_threshold (float, optional): decides if augmentation is applied. Defaults to 0.5.

  Returns:
      wav: augmented wav form
  """
  augmented_sample = audio.clone()
 
  augmentations = [
    lambda x: torchaudio.functional.pitch_shift(x, sr, np.random.choice([-4, -2, 0, 2, 4])), 
    lambda x: x + torch.randn_like(x) * 0.02, # noise
    lambda x: preemphasis(x),
    ]

  # augmentations = [
  #   lambda x: librosa.effects.pitch_shift(y=x, sr=sr, n_steps=np.random.randint(-4, 5)),
  #   lambda x: x + (torch.randn_like(torch.from_numpy(x)) * 0.01).numpy(), #0.01 is noise level
  #   lambda x: librosa.effects.time_stretch(x, rate=np.random.uniform(0.8, 1.2)),
  #   lambda x: librosa.effects.percussive(x),
  #   lambda x: librosa.effects.preemphasis(x),
  # ]

  # ensures atleast one is chosen
  chosen = [np.random.choice(augmentations)]
  for aug in augmentations:
    if np.random.random() < aug_threshold and aug not in chosen:
      chosen.append(aug)
      
  for aug in chosen:
    augmented_sample = aug(augmented_sample)
  
  return augmented_sample