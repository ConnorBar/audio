from random import shuffle
from typing import List, Dict, Tuple
import librosa
import numpy as np
from sklearn.utils import resample
import torch

from utils.constants import RANDOM_SEED

def augment_raw_audio(audio, sr, aug_threshold = 0.5):
  """randomly applies atleast one of the augmentations 

  Args:
      audio (librosa wav obj): raw wav
      sr (number): sampling rate, 
      aug_threshold (float, optional): decides if augmentation is applied. Defaults to 0.5.

  Returns:
      wav: augmented wav form
  """
  augmented_sample = audio.copy()

  augmentations = [
    lambda x: librosa.effects.pitch_shift(y=x, sr=sr, n_steps=np.random.randint(-4, 5)),
    lambda x: x + (torch.randn_like(torch.from_numpy(x)) * 0.01).numpy(), #0.01 is noise level
    lambda x: librosa.effects.time_stretch(x, rate=np.random.uniform(0.8, 1.2)),
    lambda x: librosa.effects.percussive(x),
    lambda x: librosa.effects.preemphasis(x),
  ]

  # ensures atleast one is chosen
  chosen_augmentations = [np.random.choice(augmentations)]
  
  for augment in augmentations:
    if np.random.random() < aug_threshold and augment not in chosen_augmentations:
      chosen_augmentations.append(augment)
      
  for augment in chosen_augmentations:
    augmented_sample = augment(augmented_sample)
  
  return augmented_sample

  
def assign_augmentations(X_train: List[str], y_train: List[int], augments_needed: Dict[int, int]) -> List[Tuple[str, int, bool]]:  
  """ 
  resamples the req number to get target dist from each class and labels the resampled for augmentation

  Args:
      X_train (List): train data
      y_train (List): train labels
      augments_needed (Dict): mapping of class labels to the number of augmentations needed

  Returns:
      Tuple[List, List, List]: modifed X_train, y_train & indicator if sample will be augmented
  """
  grouped_examples = {clss: [] for clss in augments_needed.keys()}
  for x, y in zip(X_train, y_train):
    grouped_examples[y].append(x)
    
  X_train_new = []
  y_train_new = []
  will_augment = []

  # resampling each class, adding og & resampled to new X & new y & marking resampled ones for augmentation
  for clss, n_samples in augments_needed.items():
    original_samples = grouped_examples[clss]
    augment_samples = resample(original_samples, replace=True, n_samples=n_samples, random_state=RANDOM_SEED)

    X_train_new.extend(original_samples + augment_samples)
    y_train_new.extend([clss] * (len(original_samples) + len(augment_samples)))
    will_augment.extend([False] * len(original_samples) + [True] * len(augment_samples))

  # shuffle to avoid any issues training later - sorted data iffy
  combined_data = list(zip(X_train_new, y_train_new, will_augment))
  shuffle(combined_data)
    
  return combined_data
