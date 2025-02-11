import re
import json
import librosa
import torch
import torchaudio
import numpy as np
import pandas as pd
from typing import List, Tuple
import torch.nn.functional as F
import torchaudio.transforms as T

from pypinyin import Style, lazy_pinyin
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.discriminant_analysis import StandardScaler

from utils.augmentations import augment_raw_audio
from utils.constants import *
from utils.selectmodel import GetDevice

valid_initials = ['EMPTY', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j',
       'q', 'x', 'z', 'c', 's', 'zh', 'ch', 'sh', 'r']

valid_finals = ['i', 'a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'uei', 'uen', 'ueng', 'uo', 'v', 'van', 've', 'vn']

device = GetDevice()

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

def split_chinese_non_chinese(text):
    # Regular expression to match Chinese characters
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    
    result = []
    buffer = ""  # Buffer to collect non-Chinese characters

    for char in text:
        if chinese_pattern.match(char):  
            if buffer:
                result.append(buffer)  # Append the previous non-Chinese sequence
                buffer = ""  # Reset buffer
            result.append(char)  # Append the Chinese character itself
        else:
            buffer += char  # Group non-Chinese characters together
    
    if buffer:
        result.append(buffer)  # Append any remaining non-Chinese characters
    
    return result
valid_pinyin = set("abcdefghijklmnopqrstuvwxyz")  # Allowed characters

def clean_pinyin(pinyin_list):
  return [''.join(c for c in word if c in valid_pinyin) for word in pinyin_list]


def proportional_sample(df: pd.DataFrame, n_samples=500_000):
  clean_df = df.copy()
  for label in ['initial', 'final']:
    unique_classes = np.unique(clean_df[label])
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=clean_df[label])
    class_weight_dict = dict(zip(unique_classes, class_weights))
    clean_df[f"{label}_weight"] = clean_df[label].map(class_weight_dict)

  clean_df["sample_weight"] = clean_df[['initial_weight', 'final_weight']].mean(axis=1)

  # Sample using computed weights
  sampled_df = clean_df.sample(n=n_samples, weights=clean_df["sample_weight"], random_state=RANDOM_SEED)
  
  # drop some cols
  sampled_df.drop(columns=['initial_weight', 'final_weight', 'sample_weight'], inplace=True)
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

# ------------------------------------------- #

def feature_extraction(args) -> Tuple[List, List[Tuple[int, int, int, int]]]:
  """ takes in wav and returns extracted feature and its labels

  Args:
      args (_type_): _description_

  Returns:
      Tuple[List, List[Tuple[int, int, int, int]]]: ( mfcc, (initial, final, tone, sanity) )
  """
  TARGET_LENGTH = 16000  # 1 second @16kHz
  N_FFT = 512           # 32ms window
  HOP_LENGTH = 160      # 10ms stride
  MAX_FRAMES = (TARGET_LENGTH - N_FFT) // HOP_LENGTH + 1  # 96 frames

  wav_file, labels, augment = args
  try:
    waveform = torch.load(LARGE_WAV_DIR / wav_file, weights_only=True).to(device)

    if waveform.size(0) > TARGET_LENGTH:
      waveform = waveform[:TARGET_LENGTH]
    else:
      waveform = torch.nn.functional.pad(waveform, (0, TARGET_LENGTH - waveform.size(0)))

    # RAW AUDIO AUGMENTATION:
    if augment:
      waveform = augment_raw_audio(waveform)

    # Update MFCC transform
    mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=13,
    melkwargs={
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'n_mels': 40,        # Standard for speech
        'center': True,      # Avoid padding artifacts
    }).to(device)

    # print("BEFORE TRANS dims: ", waveform.shape)
    mfccs = mfcc_transform(waveform).squeeze(0)
    # print("AFTER TRANS dims: ", mfccs.shape)
    # Normalize features
    mean = mfccs.mean(dim=1, keepdim=True)
    std = mfccs.std(dim=1, keepdim=True)
    mfccs = (mfccs - mean) / (std + 1e-8)

    # Ensure fixed length (padding or truncating)
    if mfccs.shape[1] < MAX_FRAMES:
      mfccs = F.pad(mfccs, (0, MAX_FRAMES - mfccs.shape[1]))  # Ensure padding
    else:
      mfccs = mfccs[:, :MAX_FRAMES]
    # print("FINAL dims: ", mfccs.shape)
    return mfccs, labels
 
    # RAW AUDIO AUGMENTATION:
    if augment:
      waveform = augment_raw_audio(waveform)
    # waveform = waveform.to(device)
    
    # mfcc normalization 
    mfcc_transform = T.MFCC(
        sample_rate=16000,
        n_mfcc=13,
        melkwargs={
            'n_mels': 40,
            'n_fft': 512,
            'hop_length': 160,
            'win_length': 400
        }
    )
    mfcc_tensor = mfcc_transform(waveform)  # Shape: [1, 13, time]
    mfcc_tensor = mfcc_tensor.squeeze(0)  # Shape: [13, time]

    # use gpu for scaling instead
    mean = mfcc_tensor.mean(dim=1, keepdim=True).to(torch.float64)
    std = mfcc_tensor.std(dim=1, keepdim=True).to(torch.float64)
    std = std.clamp(min=1e-6)

    mfcc_tensor = ((mfcc_tensor.to(torch.float64) - mean) / std).to(torch.float32)

    # Ensure fixed length (padding or truncating)
    mfcc_tensor = F.pad(mfcc_tensor, (0, max(0, MAX_FRAMES - mfcc_tensor.shape[1])))  # Pad if shorter
    mfcc_tensor = mfcc_tensor[:, :MAX_FRAMES]  # Truncate if longer

    return mfcc_tensor.cpu().numpy(), labels  # Move back to CPU for return

  except Exception as e:
    print(f"Error processing {wav_file}: {e}")
    return None, None  
