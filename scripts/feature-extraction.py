from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import librosa
from sklearn.utils import resample
import torch
import torchaudio
from pypinyin import lazy_pinyin, Style
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.constants import *
from utils.augmentations import augment_raw_audio, assign_augmentations, generate_labels

np.random.seed(RANDOM_SEED)

def feature_extraction(args) -> Tuple[List, List]:
  wav_file, label, augment = args
  try:
    wav_path = LARGE_WAV_DIR / wav_file
    
    y, sr = librosa.load(wav_path, sr=None)
    
    # RAW AUDIO AUGMENTATION:
    if augment:
      y = augment_raw_audio(y, sr)
    
    # mfcc normalization 
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    scaler = StandardScaler(with_mean=True, with_std=True)
    mfcc_scaled = scaler.fit_transform(mfccs.T).T
    
    # ensure consistent shape - truncate or pad w 0s
    mfcc_scaled = librosa.util.fix_length(mfcc_scaled, size=MAX_FRAMES, axis=1)
    
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
  y = generate_labels(result)

  y_tone = result['tone'] # used for augmentation, may want to check the other labsels
  
  TESTING = True
  
  # test train split
  X_train, X_test, y_train, y_test = train_test_split(X, list(zip(y_tone, y)), train_size=0.7, shuffle=True, random_state=RANDOM_SEED)

  # TODO REFACTOR HOW THE ONE-HOT IS BEING PUSHED THRU
 
  """
  only doing this for train:
  instead of augmenting every example, can find the proportions of class distributions
  pick an augment rate for the biggest one, sample & augment all those
  sample and augment the other ones until they reach the same number/proportion ish
  """   
  print("Splitting data and assigning augmentations...")
  # can do multiple augments depending on the class, might just it based on the tone
  y_train_tone, y_train_one_hot = zip(*y_train)
  y_test_tone, y_test_one_hot = zip(*y_test)

  augment_rate = 1.2 # sample 20% of biggest tone class & augment them
  unique_classes, class_counts = torch.unique(torch.tensor(list(y_train_tone)), return_counts=True)
  target = class_counts.max() * augment_rate
  augments_needed = {clss.item(): max(0, int(target - count)) for clss, count in zip(unique_classes, class_counts)}

  train_params = assign_augmentations(X_train=X_train, y_train=zip(y_train_tone, y_train_one_hot), augments_needed=augments_needed)
  test_params =  list(zip(X_test, y_test_one_hot, [False] * len(X_test)))
 
  local_data_dir = CUR_FEATS_DIR
  if TESTING:
    train_params = resample(list(train_params), n_samples=100)
    test_params = resample(list(zip(X_test, y_test_one_hot, [False] * len(X_test))), n_samples=100)
    local_data_dir = TEST_DATA_DIR

  # AUGMENTS - TRAIN feature extraction & pairing labels
  total = len(train_params)
  chunksize = max(1, total // (POOL_NUM * 10))  
  with Pool(POOL_NUM) as p:
    train_results = list(
      tqdm(
        p.imap(feature_extraction, train_params, chunksize=chunksize), 
           total=total, 
           desc="Train Feature Extraction"
           )
      )

  train_results = [(feat, label) for feat, label in train_results if feat is not None]
  train_features, train_label = zip(*train_results)

  X_train_augmented = np.array(train_features)
  y_train_augmented = np.array(train_label)
  
  # exporting now as a checkpoint
  np.save(local_data_dir / 'train_features.npy', X_train_augmented)
  np.save(local_data_dir / 'train_labels.npy', y_train_augmented)
  print("Train features and labels saved in \'data\' directory...")

  # NO AUGMENTS - TEST feature extraction & pairing labels
  total = len(test_params)
  chunksize = max(1, total // (POOL_NUM * 10))  
  with Pool(POOL_NUM) as p:
    test_results = list(
      tqdm(
        p.imap(feature_extraction, test_params, chunksize=chunksize), 
           total=total, 
           desc="Test Feature Extraction"
           )
      )

  test_results = [(feat, label) for feat, label in test_results if feat is not None]
  test_features, test_labels = zip(*test_results)

  X_test = np.array(test_features)
  y_test = np.array(test_labels)
  
  # exporting all features & labels
  np.save(local_data_dir / 'test_features.npy', X_test)
  np.save(local_data_dir / 'test_labels.npy', y_test)
  print("Test features and labels saved in \'data\' directory...")

if __name__ == '__main__':
  main()


