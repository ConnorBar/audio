import re
import json
import numpy as np
import pandas as pd

from pypinyin import Style, lazy_pinyin
from sklearn.utils import resample
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

from utils.constants import *

valid_initials = ['EMPTY', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j',
       'q', 'x', 'z', 'c', 's', 'zh', 'ch', 'sh', 'r']

valid_finals = ['i', 'a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'uei', 'uen', 'ueng', 'uo', 'v', 'van', 've', 'vn']


def breakdown_pinyin(phrase):
  clean_phrase = re.sub(r'[^\w]', '', phrase)
  initial = lazy_pinyin(clean_phrase, style=Style.INITIALS, strict=True)
  final = lazy_pinyin(clean_phrase, style=Style.FINALS, strict=True)
  tone = [ word[-1] for word in lazy_pinyin(clean_phrase, style=Style.FINALS_TONE3, strict=False, neutral_tone_with_five=True, tone_sandhi=True)]
  
  initial = [init if init != '' else "EMPTY" for init in initial ]

  return list(zip(initial, final, tone))

def is_valid_pinyin(initial, final):
    if initial not in valid_initials or final not in valid_finals:
        return False
    return True


def proportional_sample(df: pd.DataFrame):
  clean_df = df.copy()
  for label in ['initial', 'final']:
    unique_classes = np.unique(clean_df[label])
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=clean_df[label])
    class_weight_dict = dict(zip(unique_classes, class_weights))
    clean_df[f"{label}_weight"] = clean_df[label].map(class_weight_dict)

  clean_df["sample_weight"] = clean_df[['initial_weight', 'final_weight']].mean(axis=1)

  # Sample using computed weights
  sampled_df = clean_df.sample(n=500_000, weights=clean_df["sample_weight"], random_state=42)
  
  # drop some cols
  sampled_df.drop(columns=['pinyin_breakdown', 'initial_weight', 'final_weight', 'sample_weight'], inplace=True)
  sampled_df['sanity'] = 1
  sampled_df['augment'] = 0

  return sampled_df

  
def mark_augments(df: pd.DataFrame, sample_frac=0.7, augment_frac=0.2, insane_frac=0.1, ueng_boost=0.5):
  # resampling calcs
  unaugmented = len(df)
  total_size = int(unaugmented / sample_frac)
  to_augment = int(total_size * augment_frac)
  to_insane = int(total_size * insane_frac)

  # augment sampling
  augmented = resample(df, replace=True, n_samples=to_augment, random_state=RANDOM_SEED)
  augmented['augment'] = 1
  augmented['sanity'] = 1

  # insane sampling
  insane = resample(df, replace=True, n_samples=to_insane, random_state=RANDOM_SEED)

  with open(DATA_DIR / 'invalid_initial_final_mappings.json', 'r') as file:
    invalid_mappings = json.load(file)

  insane['final'] = insane['initial'].map(lambda x: np.random.choice(invalid_mappings.get(x, [None])))
  insane['augment'] = 1
  insane['sanity'] = 0

  overall = pd.concat([df, augmented, insane], ignore_index=True)

  # getting more ueng bc not many uengs
  ueng = overall.loc[overall['final'] == 'ueng']
  more_ueng = resample(ueng, replace=True, n_samples=int(len(ueng) * ueng_boost), random_state=RANDOM_SEED)
  more_ueng['augment'] = 1
  
  done = pd.concat([overall, more_ueng], ignore_index=True)

  # shuffling
  done = done.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

  return done
