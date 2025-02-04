

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
import pandas as pd
from pypinyin import lazy_pinyin, Style
from sklearn.utils import resample
import re
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from utils.constants import *

valid_initials = ['EMPTY', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j',
       'q', 'x', 'z', 'c', 's', 'zh', 'ch', 'sh', 'r']

valid_finals = ['i', 'a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'uei', 'uen', 'ueng', 'uo', 'v', 'van', 've', 'vn']

def breakdown_pinyin_v_to_u(phrase):
  clean_phrase = re.sub(r'[^\w]', '', phrase)
  initial = lazy_pinyin(clean_phrase, style=Style.INITIALS, strict=True)
  final = lazy_pinyin(clean_phrase, style=Style.FINALS, strict=True)
  tone = [ word[-1] for word in lazy_pinyin(clean_phrase, style=Style.FINALS_TONE3, strict=False, v_to_u=True, neutral_tone_with_five=True, tone_sandhi=True)]
  
  initial = [init if init != '' else "EMPTY" for init in initial ]

  return list(zip(initial, final, tone))

def is_valid_pinyin(initial, final):
    if initial not in valid_initials or final not in valid_finals:
        return False
    return True
  
def main():
  df = pd.read_csv('./large-corpus/other.tsv', sep='\t')

  df = df[['path', 'sentence', 'age', 'gender', 'accents']]

  print('WIP - Segmenting sentences...')

  print('Breaking down pinyin...')
  df['pinyin_breakdown'] = df['sentence'].apply(breakdown_pinyin_v_to_u)
  
  print('Cleaning up...')
  exploded = df.explode('pinyin_breakdown', ignore_index=True)
  exploded[['initial', 'final', 'tone']] = pd.DataFrame(exploded['pinyin_breakdown'].tolist(), index=exploded.index)
  exploded.drop(columns=['sentence', 'pinyin_breakdown'], inplace=True)
  mask = exploded.apply(lambda x: is_valid_pinyin(x['initial'], x['final'], x['tone']), axis=1)

  clean_df = exploded[mask]
  print('Sampling data...')


  



  
  
  return
  
  
if __name__ == '__main__':
  main()

  
  
def better_sample(df: pd.DataFrame):


  
  
  return
  

def sample_data(df, sample_size=10000, min_size=30000, max_size=150000, sampling_diff_slack=0.05):
  original = df.copy()
  dataset = pd.DataFrame(columns=df.columns)
  
  og_initial_dist = df['initial'].value_counts(normalize=True)
  og_final_dist = df['final'].value_counts(normalize=True)
  og_tone_dist = df['tone'].value_counts(normalize=True)

  min_init_values = og_initial_dist * min_size
  min_final_values = og_final_dist * min_size
  min_tone_values = og_tone_dist * min_size


  while len(dataset) < max_size:
    # want to do some sort of distribution balancing here, but also want to take into account some are more common than others
    test = resample(original, replace=True, n_samples=sample_size, random_state=RANDOM_SEED)
    dataset = pd.concat([dataset, test])

    # checking the distributions
    initials = dataset['initial'].value_counts()
    finals = dataset['final'].value_counts()
    tones = dataset['tone'].value_counts()

    cur_initial_dist = initials / len(dataset['initial'])
    cur_final_dist = finals / len(dataset['final'])
    cur_tone_dist = tones / len(dataset['tone'])

    has_all_classes = set(initials.index) >= set(og_initial_dist.index) \
                      and set(finals.index) >= set(og_final_dist.index) \
                      and set(tones.index) >= set(og_tone_dist.index)

    initial_is_prop = (abs(1 - cur_initial_dist / og_initial_dist ) > sampling_diff_slack).all()
    final_is_prop = (abs(1 - cur_final_dist / og_final_dist ) > sampling_diff_slack).all()
    tone_is_prop = (abs(1 - cur_tone_dist / og_tone_dist ) > sampling_diff_slack).all()


    if has_all_classes and initial_is_prop and final_is_prop and tone_is_prop:
      break

  # RESAMPLE RARE & MARK FOR AUGMENTATION
  rare_classes = (initials < min_init_values) | (finals < min_final_values) | (tones < min_tone_values)
  rare_samples = original[original['initial'].isin(rare_classes[rare_classes].index) |
                          original['final'].isin(rare_classes[rare_classes].index) |
                          original['tone'].isin(rare_classes[rare_classes].index)]
  dataset['augment'] = 0

  if not rare_samples.empty:
      augmented_samples = resample(rare_samples, replace=True, n_samples=len(rare_samples) * 2, random_state=RANDOM_SEED)
      augmented_samples['augment'] = 1
      dataset = pd.concat([dataset, augmented_samples])
    
  # SAMPLE AND MAKE INSANE ONES
  dataset['sanity'] = 1
  sanity_proportion = 0.2 * len(dataset)

  insane = resample(dataset, replace=True, n_samples=sanity_proportion, random_state=RANDOM_SEED)
  insane['augment'] = 1
  insane['sanity'] = 0

  with open(DATA_DIR / 'invalid_initial_final_mappings.json', 'r') as file:
    invalid_mappings = json.load(file)
  
  np.random.seed(RANDOM_SEED)
  insane.apply(lambda row: np.random.choice(invalid_mappings[row['initial']]) if row['initial'] in invalid_mappings else None, axis=1)

  dataset = pd.concat([dataset, insane])

  return dataset