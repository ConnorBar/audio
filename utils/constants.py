from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SCRIPTS_DIR = BASE_DIR / 'scripts'
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'

CUR_FEATS_DIR = DATA_DIR / 'cur_feats'
TEST_DATA_DIR = DATA_DIR / 'test'
PKL_DATA_DIR = DATA_DIR / 'encoders'
METADATA_PATH = DATA_DIR / 'metadata.csv'
WORD_TENSORS_DIR = DATA_DIR / 'word_tensors'
LARGE_WAV_DIR = DATA_DIR / 'DANGER' / 'large-corpus'

LARGE_MP3_DIR = LARGE_WAV_DIR / 'clips'
LARGE_16K_DIR = LARGE_WAV_DIR / 'clips16k'

POOL_NUM = os.cpu_count() - 2

MAX_FRAMES = 237 # calced from notebook, 95th percentile of frame counts

RANDOM_SEED = 42