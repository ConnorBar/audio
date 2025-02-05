"""
Proposed Pipeline:
  take dataset
  
  explode it into the 8mil examples based on pinyin
    - need to add sentence segmentation later to
  
  filter out all bad info - Esmond, SatanI etc
  
  get some sort of idea of how balanced the classes are
  
  mark some percent for augmentation to balance

  sample some percent of that for the actual data to use 
  
  split into train test validation
    - look into torches feature extraction - does that run on mps or cpu?
    - it runs on cuda so might want to switch i guess??
  
  train - generate like 20-40% extra exmaples that are labeled as BAD with a BAD combination
    - could distinguish between severely bad and a little bad but that sounds like cancer right now
    - I have all valid pinyin combos, somehow generate invalid ones
      - make an invalid pinyin mapping, randomly pick one from the INVALID list
    
  test - do NOT augmnet ANYTHING

  validate - only a few extra bad exmaples but also can just use regular non-augmented ones
    
"""
from multiprocessing import Pool
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import re
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from utils.data_processing import breakdown_pinyin, is_valid_pinyin, proportional_sample, mark_augments
from utils.augmentations import *
from utils.constants import *

np.random.seed(RANDOM_SEED)


def feature_extraction(args) -> Tuple[List, List[Tuple[int, int, int, int]]]:
  """is used in multiprocessing pipeline, takes in wav and returns extracted feature and its labels

  Args:
      args (_type_): _description_

  Returns:
      Tuple[List, List[Tuple[int, int, int, int]]]: ( mfcc, (initial, final, tone, sanity) )
  """
  wav_file, labels, augment = args
  try:
    wav_path = LARGE_WAV_DIR / wav_file
    
    y, sr = librosa.load(wav_path, sr=None)
    
    # RAW AUDIO AUGMENTATION:
    if augment:
      y = augment_raw_audio(y, sr)
    
    # mfcc normalization 
    # librosa.feature.melspectrogram(y=y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    scaler = StandardScaler(with_mean=True, with_std=True)
    mfcc_scaled = scaler.fit_transform(mfccs.T).T
    
    # ensure consistent shape - truncate or pad w 0s
    mfcc_scaled = librosa.util.fix_length(mfcc_scaled, size=MAX_FRAMES, axis=1)
    
    return mfcc_scaled, labels
    
  except Exception as e:
    print(f"Error processing {wav_file}: {e}")
    return None, None  

def main():
  df = pd.read_csv('./large-corpus/other.tsv', sep='\t')

  df = df[['path', 'sentence', 'age', 'gender', 'accents']]

  print('WIP - Segmenting sentences...')

  # TODO SENTENCE SEGMENTATION - should only change how i handle the 'path' attribute

  print('Breaking down pinyin...')
  df['pinyin_breakdown'] = df['sentence'].apply(breakdown_pinyin)
  df['character'] = df['sentence'].apply(lambda row: [char for char in re.sub(r'[^\w]', '', row)])
  
  print('Cleaning up...')
  mask = df.apply(lambda row: len(row['character']) == len(row['pinyin_breakdown']), axis=1)
  filtered_df = df[mask]

  exploded = filtered_df.explode(['character', 'pinyin_breakdown'], ignore_index=True)
  exploded[['initial', 'final', 'tone']] = pd.DataFrame(exploded['pinyin_breakdown'].tolist(), index=exploded.index)
  exploded.drop(columns=['sentence', 'pinyin_breakdown'], inplace=True)
  mask = exploded.apply(lambda x: is_valid_pinyin(x['initial'], x['final'], x['tone']), axis=1)

  clean_df = exploded[mask]

  print('Sampling data...')
  sampled_data = proportional_sample(clean_df)
  
  # test train split
  train_df, temp_df = train_test_split(sampled_data, test_size=0.2, stratify=sampled_data[['final']], random_state=42)
  val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[['final']], random_state=42)

  train_df = mark_augments(train_df)

  X_train, y_train, train_augs = train_df['path'].to_list(), generate_labels(train_df), train_df['augment']
  X_val, y_val = val_df['path'].tolist(), generate_labels(val_df)
  X_test, y_test = test_df['path'].tolist(), generate_labels(test_df)
  
  TESTING = True
  
  train_params = list(zip(X_train, y_train, train_augs))
  test_params =  list(zip(X_test, y_test, [False] * len(X_test)))
  val_params = list(zip(X_val, y_val, [False] * len(X_val))) 
 
  output_data_dir = CUR_FEATS_DIR
  if TESTING:
    train_params = resample(list(train_params), n_samples=1000)
    test_params = resample(test_params, n_samples=100)
    test_params = resample(val_params, n_samples=100)
    output_data_dir = TEST_DATA_DIR

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
  np.save(output_data_dir / 'train_features.npy', X_train_augmented)
  np.save(output_data_dir / 'train_labels.npy', y_train_augmented)
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
  np.save(output_data_dir / 'test_features.npy', X_test)
  np.save(output_data_dir / 'test_labels.npy', y_test)

  print("Test features and labels saved in \'data\' directory...") 

  # NO AUGMENTS - VALIDATION feature extraction & pairing labels
  total = len(val_params)
  chunksize = max(1, total // (POOL_NUM * 10))  
  with Pool(POOL_NUM) as p:
    val_results = list(
      tqdm(
        p.imap(feature_extraction, val_params, chunksize=chunksize), 
           total=total, 
           desc="Test Feature Extraction"
           )
      )

  val_results = [(feat, label) for feat, label in val_results if feat is not None]
  val_features, val_labels = zip(*val_results)

  X_val = np.array(val_features)
  y_val = np.array(val_labels)
  
  # exporting all features & labels
  np.save(output_data_dir / 'val_features.npy', X_val)
  np.save(output_data_dir / 'val_labels.npy', y_val)
  print("Validation features and labels saved in \'data\' directory...")
  
  
  return
  
  
if __name__ == '__main__':
  main()
