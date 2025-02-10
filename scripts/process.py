"""
Proposed Pipeline:
  given segmented sentences, clean anything up
  
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
import ast
from multiprocessing import Pool
import pandas as pd
from pypinyin import lazy_pinyin
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import re
import json
import numpy as np
from tqdm import tqdm

import utils.data_processing as dp
from utils.data_processing import breakdown_pinyin, is_valid_pinyin, proportional_sample, mark_augments, feature_extraction, split_chinese_non_chinese
from utils.augmentations import *
from utils.constants import *


np.random.seed(RANDOM_SEED)

def main():

  # ------------Data Cleaning----------------------- #

  print('Reading in data and cleaning...')
  df = pd.read_csv(DATA_DIR / 'metadata.csv')

  tqdm.pandas(desc="Converting to paths to list")
  df['word_files'] = df['word_files'].progress_apply(ast.literal_eval)
  tqdm.pandas(desc="Breaking down pinyin")
  df['pinyin_breakdown'] = df['sentence'].progress_apply(breakdown_pinyin)
  tqdm.pandas(desc="Cleaning Pinyin")
  df['pinyin'] = df['sentence'].progress_apply(lambda row: dp.clean_pinyin(lazy_pinyin(row)))
  tqdm.pandas(desc="Removing non-chinese characters")
  df['character'] = df['sentence'].progress_apply(split_chinese_non_chinese)

  tqdm.pandas(desc="Checking even lengths")
  valid_lengths = df.progress_apply(lambda row: len({
    len(row['word_files']), 
    len(row['pinyin_breakdown']), 
    len(row['pinyin']), 
    len(row['character'])
    }) == 1, axis=1)
  df = df[valid_lengths]
  
  exploded = df.explode(['word_files', 'pinyin_breakdown', 'pinyin', 'character'], ignore_index=True)
  exploded[['initial', 'final', 'tone']] = pd.DataFrame(exploded['pinyin_breakdown'].tolist(), index=exploded.index)
  exploded.drop(columns=['pinyin_breakdown'], inplace=True)

  tqdm.pandas(desc="Ensuring valid pinyin")
  valid_chinese_mask = exploded.progress_apply(lambda x: is_valid_pinyin(x['initial'], x['final']), axis=1)
  clean_df = exploded[valid_chinese_mask]
  tqdm.pandas(desc="Processing data")

  # -----------Sampling and Splitting------------------------ #

  print('Sampling data...')
  sampled_data = proportional_sample(clean_df)
  
  # test train split
  train_df, temp_df = train_test_split(sampled_data, test_size=0.2, stratify=sampled_data[['final']], random_state=RANDOM_SEED)
  val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[['final']], random_state=RANDOM_SEED)

  train_df = mark_augments(train_df)

  X_train, y_train, train_augs = train_df['word_files'].to_list(), generate_labels(train_df), train_df['augment']
  X_val, y_val = val_df['word_files'].tolist(), generate_labels(val_df)
  X_test, y_test = test_df['word_files'].tolist(), generate_labels(test_df)
  
  # ------------Params setup----------------------- #

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

  # ----------Train extractions------------------------- #

  # AUGMENTS - TRAIN feature extraction & pairing labels
  total = len(train_params)
  chunksize = max(1, total // (POOL_NUM * 2))  
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

  # -----------Test Extractions------------------------ #
  
  # NO AUGMENTS - TEST feature extraction & pairing labels
  total = len(test_params)
  chunksize = max(1, total // (POOL_NUM * 2))  
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

  # -----------Validation Extractions----------------------- #

  # NO AUGMENTS - VALIDATION feature extraction & pairing labels
  total = len(val_params)
  chunksize = max(1, total // (POOL_NUM * 2))  
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
  
  
if __name__ == '__main__':
  main()
