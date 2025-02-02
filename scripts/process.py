

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
import re

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

def is_valid_pinyin(pinyin_breakdown):
    for initial, final, tone in pinyin_breakdown:
        if initial not in valid_initials or final not in valid_finals:
            return False
    return True

def main():
  df = pd.read_csv('./large-corpus/other.tsv', sep='\t')

  df = df[['path', 'sentence', 'age', 'gender', 'accents']]

  df['pinyin_breakdown'] = df['sentence'].apply(breakdown_pinyin_v_to_u)
  
  # filter out all non real initials and
  df = df[df['pinyin_breakdown'].apply(is_valid_pinyin)]

  



  
  
  return
  
  
if __name__ == '__main__':
  main()

