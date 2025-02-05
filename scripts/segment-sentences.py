import re
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd

import cProfile
import pstats
import json


from typing import List
from pypinyin import lazy_pinyin
from torchaudio import transforms as T
from torchaudio.pipelines import MMS_FA as bundle
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


from utils.constants import *

# ------------------------------------------------ #

device = 'cpu' # sad not on mps
model = bundle.get_model()
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

# ------------------------------------------------ #

def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
  with torch.inference_mode():
    emission, _ = model(waveform.to(device))
    token_spans = aligner(emission[0], tokenizer(transcript))
  return emission, token_spans

def get_words(waveform, spans, num_frames):
  ratio = waveform.size(1) / num_frames

  return [
    waveform[:, int(ratio * span[0].start) : int(ratio * span[-1].end)]
    for span in spans
  ]

def driver(args):
  path, transcript, sentence = args

  waveform16k, _ = torchaudio.load(LARGE_16K_DIR / path)
  waveform16k = waveform16k[0:1] # keep mono channel

  emission, token_spans = compute_alignments(waveform16k, transcript)
  num_frames = emission.size(1)

  words = get_words(waveform16k, token_spans, num_frames)

  return words, sentence
 

# ------------------------------------------------ #

def main():

  df = pd.read_csv('./large-corpus/prepped.csv', memory_map=True)
  df = df[:100]  # debugging size
  df['transcript'] = df['transcript'].apply(json.loads)

  params = df[['path', 'transcript', 'sentence']].to_records(index=False).tolist()
  
  total = len(df)
  pool_size = min(cpu_count(), 8)
  chunksize = max(1, total // (pool_size * 4))

  with Pool(pool_size) as p:
    segmentations = list(
      tqdm(
        p.imap(driver, params, chunksize=chunksize), 
           total=total, 
           desc="Sentence Segmentation"
           )
      )
  
  
  return


if __name__ == '__main__':
  mp.set_start_method('spawn', force=True)  # ensures safe multiprocessing on macOS
  main()