from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
CUR_FEATS_DIR = DATA_DIR / 'cur_feats'
TEST_DATA_DIR = DATA_DIR / 'test'
PKL_DATA_DIR = DATA_DIR / 'encoders'
SCRIPTS_DIR = BASE_DIR / 'scripts'
MODELS_DIR = BASE_DIR / 'models'

RESULTS_PATH = DATA_DIR / 'result.csv'
LARGE_WAV_DIR = BASE_DIR / 'DANGER' / 'large-wav'
LARGE_MP3_DIR = BASE_DIR / 'large-corpus' / 'clips'

POOL_NUM = os.cpu_count() - 2

MAX_FRAMES = 237 # calced from notebook, 95th percentile of frame counts

RANDOM_SEED = 42