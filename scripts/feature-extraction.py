import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
import os

from data.constants import RESULTS_PATH, LARGE_WAV_DIR, MAX_FRAMES, DATA_DIR, POOL_NUM


def feature_extraction(args):
  wav_file, sentence, tone = args
  try:
    wav_path = os.path.join(LARGE_WAV_DIR, wav_file)
    
    y, sr = librosa.load(wav_path, sr=None)
    # can do any raw audio processing here, will add after baseline testing
    
    # extract features, can change, maybe something with pitch contour would be good?
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # removed spectral centroids for now, on github if want to bring back

    # mfcc standardization
    scaler = StandardScaler(with_mean=True, with_std=True)
    mfcc_scaled = scaler.fit_transform(mfccs.T).T
    # can augment some and make new samples and return multiple features - more training data yippee
    
    # ensure consistent shape
    if mfccs.shape[1] > MAX_FRAMES:
      # If longer than MAX_FRAMES, truncate
      mfcc_scaled = mfcc_scaled[:, :MAX_FRAMES]
    else:
      # If shorter, pad with zeros
      pad_width = MAX_FRAMES - mfccs.shape[1]
      mfcc_scaled = np.pad(mfcc_scaled, ((0, 0), (0, pad_width)), mode='constant')

    tone -= 1 # this is to keep the labels in bounds of BCE, can just add 1 later if you want
    return mfcc_scaled, sentence, tone
    
  except Exception as e:
    print(f"Error processing {wav_file}: {e}")
    return None, None  
  
def main():
  result = pd.read_csv(RESULTS_PATH)

  # feature extraction & pairing labels
  with Pool(POOL_NUM) as p:
    total = len(result['path'])
    results = list(tqdm(p.imap(feature_extraction, zip(result['wav_path'], result['sentence'], result['tone'])), total=total, desc="Feature Extraction"))
  
  results = [(feat, tone_label, sentence_label) for feat, tone_label, sentence_label in results if feat is not None]
  all_features, all_tones, all_sentences = zip(*results)

  # exporting all features & labels
  X = np.array(all_features)
  y_tones = np.array(all_tones)
  y_sentences = np.array(all_sentences)
  
  np.save(os.path.join(DATA_DIR, 'features.npy'), X)
  np.save(os.path.join(DATA_DIR, 'tone_labels.npy'), y_tones)
  np.save(os.path.join(DATA_DIR, 'sentence_labels.npy'), y_sentences)
  print('\nFeatures and labels created and saved in \'data\' directory.')

if __name__ == '__main__':
  main()
  
      


