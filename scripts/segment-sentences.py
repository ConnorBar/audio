import re
import json
import sys
from pypinyin import lazy_pinyin
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
import multiprocessing as mp
import jieba

import cProfile
import pstats

from tqdm import tqdm
from typing import List
from torchaudio import transforms as T
from torchaudio.pipelines import MMS_FA as bundle

from utils.constants import *

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"  # Reduces fragmentation[27][30]

torch.cuda.synchronize()

# ------------------------------------------------ #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = bundle.get_model().to(device)
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

  try:
    waveform16k, _ = torchaudio.load(LARGE_16K_DIR / path)
    waveform16k = waveform16k[0:1].to(device)

    emission, token_spans = compute_alignments(waveform16k, transcript)
    num_frames = emission.size(1)

    words = get_words(waveform16k, token_spans, num_frames)

    # save each word tensor and store filenames
    word_filenames = []
    for i, word in enumerate(words):
      filename = f"{Path(path).stem}_word{i}.pt"
      filepath = WORD_TENSORS_DIR / filename
      torch.save(word.cpu(), filepath)
      word_filenames.append(str(filepath))

    return word_filenames, sentence

  except Exception as e:
    if "targets length is too long for CTC" in str(e):
      error_message = f"Alignment failed for {path}: {e}\n"
      sys.stderr.write(error_message)  # Print to stderr to avoid tqdm interference
      sys.stderr.flush()  # Ensure immediate writing to stderr

    else:
      sys.stderr.write(f"something else went wrong: {e}")  # Print to stderr to avoid tqdm interference
      sys.stderr.flush()  # Ensure immediate writing to stderr

    return None  # Ensures the pipeline doesn’t crash

# ------------------------------------------------ #

def main():
  print('Reading in dataframe and cleaning...')
  df = pd.read_csv('./large-corpus/validated.tsv', sep='\t', memory_map=True)
  # df = df[:100]  # debugging size
  df['sentence'] = df['sentence'].apply(lambda row: re.sub(r'[^\w]', '', row)) 

  valid_pinyin = set("abcdefghijklmnopqrstuvwxyz")  # Allowed characters

  def clean_pinyin(pinyin_list):
    return [''.join(c for c in word if c in valid_pinyin) for word in pinyin_list]

  df['transcript'] = df['sentence'].apply(lambda row: clean_pinyin(lazy_pinyin(row)))


  params = df[['path', 'transcript', 'sentence']].to_records(index=False).tolist()
  
  # total = len(df)
  # pool_size = min(cpu_count(), 5) 
  # chunksize = max(1, total // (pool_size * 2))

  # with Pool(pool_size) as p:
  #   segmentations = list(
  #     tqdm(
  #       p.imap_unordered(driver, params, chunksize=chunksize), 
  #          total=total, 
  #          desc="Sentence Segmentation"
  #          )
  #     )
  #   sys.stdout.flush()  

  segmentations = [driver(param) for param in tqdm(params) ]
  segmentations = [s for s in segmentations if s is not None]

  
  metadata_df = pd.DataFrame(segmentations, columns=['word_files', 'sentence'])
  metadata_df.to_csv(DATA_DIR / "metadata.csv", index=False)
  
  return


if __name__ == '__main__':
  mp.set_start_method('spawn', force=True)  # ensures safe multiprocessing
  main()