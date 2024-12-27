import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')

RESULTS_PATH = os.path.join(DATA_DIR, 'result.csv')
LARGE_WAV_DIR = os.path.join(BASE_DIR, 'DANGER', 'large-wav')
LARGE_MP3_DIR = os.path.join(BASE_DIR, 'large-corpus', 'clips')

POOL_NUM = os.cpu_count() - 2

MAX_FRAMES = 237 # calced from notebook, 95th percentile of frame counts
