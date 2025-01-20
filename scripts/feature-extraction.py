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

from data.constants import RESULTS_PATH, LARGE_WAV_DIR, MAX_FRAMES, DATA_DIR, POOL_NUM

def augment_audio(audio, sr):
  augmented = audio.copy()
  
  # Randomly apply pitch shift
  if np.random.random() < 0.5:
    n_steps = np.random.randint(-4, 5)
    augmented = librosa.effects.pitch_shift(augmented, sr, n_steps)
  
  # Randomly add noise
  if np.random.random() < 0.5:
    noise_level=0.01
    noise = torch.randn_like(torch.from_numpy(augmented)) * noise_level
    augmented += noise.numpy()
  
  # Randomly time stretch
  if np.random.random() < 0.5:
    rate = np.random.uniform(0.8, 1.2)
    augmented = librosa.effects.time_stretch(augmented, rate)
  
  return audio, augmented

def feature_extraction(args):
  wav_file, label, augment = args
   # if classifyTone: # this is for when adding sentence
  label -= 1 # this is to keep the labels in bounds of BCE, can just add 1 later if you want
  try:
    wav_path = os.path.join(LARGE_WAV_DIR, wav_file)
    
    y, sr = librosa.load(wav_path, sr=None)
    
    # RAW AUDIO AUGMENTATION:
    # if train aug, can make new samples and return multiple features - more training data yippee
    if augment:
      y = augment_audio(y, sr)
      if label == 1: # do additional augmenting on second tone since i have so few of them
        ...
    # need to handle multiple augments coming back, esp for when im generating more examples for second tone
    
    # mfcc normalization & standardization
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    scaler = StandardScaler(with_mean=True, with_std=True)
    mfcc_scaled = scaler.fit_transform(mfccs.T).T
    
    # ensure consistent shape - truncate or pad w 0s
    if mfccs.shape[1] > MAX_FRAMES:
      mfcc_scaled = mfcc_scaled[:, :MAX_FRAMES]
    else:
      pad_width = MAX_FRAMES - mfccs.shape[1]
      mfcc_scaled = np.pad(mfcc_scaled, ((0, 0), (0, pad_width)), mode='constant')

    # MFCC AUGMENTATION:
    # if train aug, can make new samples and return multiple features - more training data yippee
    if augment:
      # can do some masking? dont want to do too much and lose in 
      ...

   

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
  
  # AUGMENTS - TRAIN feature extraction & pairing labels
  with Pool(POOL_NUM) as p:
    total = len(result['path'])
    train_results = list(tqdm(p.imap(feature_extraction, zip(X_train,  y_train, augment=True)), total=total, desc="Train Feature Extraction"))

  train_results = [(feat, tone_label) for feat, tone_label in train_results if feat is not None]
  train_features, train_tones = zip(*train_results)

  X_train = np.array(train_features)
  y_train = np.array(train_tones)

  # NO AUGMENTS - TEST feature extraction & pairing labels
  with Pool(POOL_NUM) as p:
    total = len(result['path'])
    test_results = list(tqdm(p.imap(feature_extraction, zip(X_test,  y_test, augment=False)), total=total, desc="Test Feature Extraction"))

  test_results = [(feat, tone_label) for feat, tone_label in test_results if feat is not None]
  test_features, test_tones = zip(*test_results)

  X_test = np.array(test_features)
  y_test = np.array(test_tones)
  
  # exporting all features & labels
  np.save(os.path.join(DATA_DIR, 'train_features.npy'), X_train)
  np.save(os.path.join(DATA_DIR, 'train_labels.npy'), y_train)
  np.save(os.path.join(DATA_DIR, 'test_features.npy'), X_test)
  np.save(os.path.join(DATA_DIR, 'test_labels.npy'), y_test)

  print('\nFeatures and labels created and saved in \'data\' directory.')

if __name__ == '__main__':
  main()
  
      


