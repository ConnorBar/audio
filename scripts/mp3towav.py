from multiprocessing import Pool
import pydub
from tqdm import tqdm
import pandas as pd
import os

from data.constants import RESULTS_PATH, LARGE_WAV_DIR, LARGE_MP3_DIR, POOL_NUM

result = pd.read_csv(RESULTS_PATH)

def f(path):
  try:
    audio = pydub.AudioSegment.from_mp3(os.path.join(LARGE_MP3_DIR, path))
    audio.export(os.path.join(LARGE_WAV_DIR, path.split('_')[-1][:-4] + ".wav"), format="wav")
  except Exception as e:
    print(f"Error processing {path}: {str(e)}")

def main():
  with Pool(POOL_NUM) as p:
    total = len(result['path'])
    list(tqdm(p.imap(f, result['path']), total=total, desc="Converting audio"))

if __name__ == '__main__':
  main()