import re
import sys
from pypinyin import lazy_pinyin
import torch
import torchaudio
import pandas as pd
import multiprocessing as mp

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

# ------------------------------------------------ #

def process_batch(batch_params):
  batch_results = []
  batch_waveforms = []

  for path, transcript, sentence in batch_params:
    try:
      waveform16k, _ = torchaudio.load(LARGE_16K_DIR / path)
      batch_waveforms.append((waveform16k[0:1].to(device), transcript, sentence))
    except Exception as e:
      sys.stderr.write(f"Error loading {path}: {e}\n")
      sys.stderr.flush()


  for waveform, transcript, sentence in batch_waveforms:
    try:
      emission, token_spans = compute_alignments(waveform, transcript)
      num_frames = emission.size(1)
      words = get_words(waveform, token_spans, num_frames)

      # save each word tensor and store filenames
      word_filenames = []
      for i, word in enumerate(words):
        filename = f"{Path(path).stem}_word{i}.pt"
        filepath = WORD_TENSORS_DIR / filename
        torch.save(word.cpu(), filepath)
        word_filenames.append(str(filepath))


      batch_results.append((word_filenames, sentence))
    except Exception as e:
      sys.stderr.write(f"Error processing {path}: {e}")
      sys.stderr.flush()

  return batch_results

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

  # ------------------------------------------------ #

  params = df[['path', 'transcript', 'sentence']].to_records(index=False).tolist()

  BATCH_SIZE = 32  # Adjust based on your GPU memory
  batches = [params[i:i + BATCH_SIZE] for i in range(0, len(params), BATCH_SIZE)]
    
  segmentations = []
  for batch in tqdm(batches, desc="Processing batches"):
        batch_results = process_batch(batch)
        segmentations.extend([r for r in batch_results if r is not None])

  metadata_df = pd.DataFrame(segmentations, columns=['word_files', 'sentence'])
  metadata_df.to_csv(DATA_DIR / "metadata.csv", index=False)
  
  return


if __name__ == '__main__':
  mp.set_start_method('spawn', force=True)  # ensures safe multiprocessing
  main()