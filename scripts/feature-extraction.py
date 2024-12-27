import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from multiprocessing import Pool
import os

from data.constants import RESULTS_PATH, LARGE_WAV_DIR, MAX_FRAMES, DATA_DIR, POOL_NUM

result = pd.read_csv(RESULTS_PATH)

def feature_calculation(args):
  wav_file, tone = args
  try:
    wav_path = os.path.join(LARGE_WAV_DIR, wav_file)
    
    y, sr = librosa.load(wav_path, sr=None)
    
    # extract features, can change, maybe something with pitch contour would be good?
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    # Ensure consistent shape - this kinda sketch tbh
    if mfccs.shape[1] > MAX_FRAMES:
      mfccs = mfccs[:, :MAX_FRAMES]
      spectral_centroids = spectral_centroids[:MAX_FRAMES]
    else:
      pad_width = MAX_FRAMES - mfccs.shape[1]
      mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
      spectral_centroids = np.pad(spectral_centroids, (0, pad_width), mode='constant')
    
    combined_features = np.concatenate((mfccs, spectral_centroids.reshape(1, -1)), axis=0)

    return combined_features, tone
    
  except Exception as e:
    print(f"Error processing {wav_file}: {e}")
    return None, None  
  
def main():
  with Pool(POOL_NUM) as p:
    total = len(result['path'])
    results = list(tqdm(p.imap(feature_calculation, zip(result['wav_path'], result['tone'])), total=total, desc="Feature Extraction"))
  
  results = [(feat, lbl) for feat, lbl in results if feat is not None]
  all_features, all_labels = zip(*results)

  X = np.array(all_features)
  y = np.array(all_labels)
  
  np.save(os.path.join(DATA_DIR, 'features.npy'), X)
  np.save(os.path.join(DATA_DIR, 'labels.npy'), y)

if __name__ == '__main__':
  main()
  
      


