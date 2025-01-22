from random import shuffle
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import librosa
import torch
import torchaudio
import os
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from data.constants import RESULTS_PATH, LARGE_WAV_DIR, MAX_FRAMES, DATA_DIR, POOL_NUM

np.random.seed(42)

def assign_augmentations(X_train: List, y_train: List, augments_needed: Dict[int, int]) -> Tuple[List, List, List]:
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
    augment_samples = resample(original_samples, replace=True, n_samples=n_samples)

    X_train_new.extend(original_samples + augment_samples)
    y_train_new.extend([clss] * (len(original_samples) + len(augment_samples)))
    will_augment.extend([False] * len(original_samples) + [True] * len(augment_samples))

  # shuffle to avoid any issues training later - sorted data iffy
  combined_data = list(zip(X_train_new, y_train_new, will_augment))
  shuffle(combined_data)
  X_train_shuffled, y_train_shuffled, will_augment_shuffled = zip(*combined_data)
  
  return list(X_train_shuffled), list(y_train_shuffled), list(will_augment_shuffled)

def augment_audio(audio, sr, aug_threshold = 0.5):
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

def feature_extraction(args) -> Tuple[List, List]:
  wav_file, label, augment = args
   # if classifyTone: # this is for when adding sentence
  label -= 1 # this is to keep the labels in bounds of BCE, can just add 1 later if you want
  try:
    wav_path = os.path.join(LARGE_WAV_DIR, wav_file)
    
    y, sr = librosa.load(wav_path, sr=None)
    
    # RAW AUDIO AUGMENTATION:
    if augment:
      y = augment_audio(y, sr)
    
    # mfcc normalization 
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    scaler = StandardScaler(with_mean=True, with_std=True)
    mfcc_scaled = scaler.fit_transform(mfccs.T).T
    
    # ensure consistent shape - truncate or pad w 0s
    if mfccs.shape[1] > MAX_FRAMES:
      mfcc_scaled = mfcc_scaled[:, :MAX_FRAMES]
    else:
      pad_width = MAX_FRAMES - mfccs.shape[1]
      mfcc_scaled = np.pad(mfcc_scaled, ((0, 0), (0, pad_width)), mode='constant')
    
    # MFCC AUGMENTATION: time & freq masking - gonna try just augments on raw wav for now
    # if augment:
    #   if np.random.random() < 0.5:
    #     time_mask = np.random.randint(0, mfcc_scaled.shape[1] // 2)
    #     start = np.random.randint(0, mfcc_scaled.shape[1] - time_mask)
    #     mfcc_scaled[:, start + time_mask] = 0
    #   if np.random.random() < 0.5:
    #     freq_mask = np.random.randint(0, mfcc_scaled.shape[0] // 2)
    #     start = np.random.randint(0, mfcc_scaled.shape[0] - freq_mask)
    #     mfcc_scaled[start + freq_mask, :] = 0


    return mfcc_scaled, label
    
  except Exception as e:
    print(f"Error processing {wav_file}: {e}")
    return None, None  
  
def main():
  result = pd.read_csv(RESULTS_PATH)

  X = result['wav_path']
  y = result['tone'] 
  
  # test train split
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)
 
  """
  only doing this for train:
  instead of augmenting every example, can find the proportions of class distributions
  pick an augment rate for the biggest one, sample & augment all those
  sample and augment the other ones until they reach the same number/proportion ish
  """   
  print("Splitting data and assigning augmentations...")
  augment_rate = 1.2 # sample 20% of biggest class & augment them
  unique_classes, class_counts = torch.unique(torch.tensor(y_train.to_numpy()), return_counts=True)
  target = class_counts.max() * augment_rate
  
  augments_needed = {clss.item(): max(0, int(target - count)) for clss, count in zip(unique_classes, class_counts)}

  X_train_new, y_train_new, will_augment = assign_augmentations(X_train=X_train, y_train=y_train, augments_needed=augments_needed)
  
  # TESTING:
  # X_TRAIN_TESTING, Y_TRAIN_TESTING, WILL_AUGMENT_TESTING= zip(*resample(list(zip(X_train_new, y_train_new, will_augment)), n_samples=100))
  # X_TEST_TESTING, Y_TEST_TESTING = zip(*resample(list(zip(X_train_new, y_train_new)), n_samples=100))
  
  # AUGMENTS - TRAIN feature extraction & pairing labels
  with Pool(POOL_NUM) as p:
    total = len(X_train_new)
    train_results = list(tqdm(p.imap(feature_extraction, zip(X_train_new,  y_train_new, will_augment)), total=total, desc="Train Feature Extraction"))
    # total = len(X_TRAIN_TESTING)
    # train_results = list(tqdm(p.imap(feature_extraction, zip(X_TRAIN_TESTING,  Y_TRAIN_TESTING, WILL_AUGMENT_TESTING)), total=total, desc="Train Feature Extraction"))

  train_results = [(feat, tone) for feat, tone in train_results if feat is not None]
  train_features, train_tones = zip(*train_results)

  X_train_augmented = np.array(train_features)
  y_train_augmented = np.array(train_tones)
  
  # exporting now as a checkpoint
  np.save(os.path.join(DATA_DIR, 'train_features.npy'), X_train_augmented)
  np.save(os.path.join(DATA_DIR, 'train_labels.npy'), y_train_augmented)
  print("Train features and labels saved in \'data\' directory...")

  # NO AUGMENTS - TEST feature extraction & pairing labels
  with Pool(POOL_NUM) as p:
    total = len(X_test)
    test_results = list(tqdm(p.imap(feature_extraction, zip(X_test,  y_test, [False]*total)), total=total, desc="Test Feature Extraction"))
    # total = len(X_TEST_TESTING)
    # test_results = list(tqdm(p.imap(feature_extraction, zip(X_TEST_TESTING,  Y_TEST_TESTING, [False]*total)), total=total, desc="Test Feature Extraction"))

  test_results = [(feat, tone) for feat, tone in test_results if feat is not None]
  test_features, test_tones = zip(*test_results)

  X_test = np.array(test_features)
  y_test = np.array(test_tones)
  
  # exporting all features & labels
  np.save(os.path.join(DATA_DIR, 'test_features.npy'), X_test)
  np.save(os.path.join(DATA_DIR, 'test_labels.npy'), y_test)
  print("Test features and labels saved in \'data\' directory...")

if __name__ == '__main__':
  main()


