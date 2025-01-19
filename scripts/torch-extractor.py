import torch
import torchaudio
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

from data.constants import RESULTS_PATH, LARGE_WAV_DIR

"""
was trying to do the feature extraction on gpu to compare to multiprocessed cpu but couldnt figure it out, sadge
"""

MAX_FRAMES = 150
# Constants
MPS_DEVICE = torch.device("mps")
POOL_NUM = 4
BATCH_SIZE = 4  # Number of files per process
# DATA_DIR = "./data"

def feature_extraction_batch(batch):
    results = []
    for wav_file, tone in batch:
        try:
            wav_path = os.path.join(LARGE_WAV_DIR, wav_file)
            waveform, sr = torchaudio.load(wav_path)
            waveform = waveform.to(MPS_DEVICE)

            # Resample and extract MFCC
            target_sr = 16000
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr).to(MPS_DEVICE)
                waveform = resampler(waveform)

            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=target_sr,
                n_mfcc=13,
                melkwargs={
                    'n_fft': 512,
                    'n_mels': 40,
                    'hop_length': 256,
                }
            ).to(MPS_DEVICE)


            print(f"WAVEFORM PRIOR Shape: {waveform.shape}")
            max_samples = MAX_FRAMES * target_sr // 100
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            print(f"WAVEFORM shape: {waveform.shape}")
 
            mfccs = mfcc_transform(waveform)
            print(f"MFCC shape: {mfccs.shape}")

            mfccs_standardized = stable_standardize(mfccs)


            # Ensure consistent shape
            if mfccs_standardized.shape[-1] > MAX_FRAMES:
                mfccs_standardized = mfccs_standardized[:, :, :MAX_FRAMES]
            else:
                pad_width = MAX_FRAMES - mfccs_standardized.shape[-1]
                mfccs_standardized = torch.nn.functional.pad(mfccs_standardized, (0, pad_width))
            print(f"MFCC STANDARDIZED shape: {mfccs_standardized.shape}")

            results.append((mfccs_standardized.cpu().numpy(), tone))
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            results.append((None, None))
    return results
  
def stable_standardize(data, eps=1e-6):
    """
    Perform stable standardization on a tensor to avoid numerical instability.
    data: PyTorch tensor of shape (..., frames)
    eps: Small constant to avoid division by zero
    """
    # Calculate mean and std along the last dimension (features or frames)
    mean = data.mean(dim=-1, keepdim=True)
    var = data.var(dim=-1, keepdim=True, unbiased=False)
    
    # Add epsilon to avoid instability when variance is very small
    std = torch.sqrt(var + eps)
    standardized_data = (data - mean) / std
    return standardized_data

def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    result = pd.read_csv(RESULTS_PATH)
    data = list(zip(result['wav_path'], result['tone']))

    # Batch data
    batches = [data[i:i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]

    with Pool(POOL_NUM) as p:
        results = list(tqdm(p.imap(feature_extraction_batch, batches), total=len(batches), desc="Feature Extraction"))

    # Flatten results and filter out None values
    results = [item for batch in results for item in batch if item[0] is not None]
    all_features, all_labels = zip(*results)

    X = np.array(all_features)
    y = np.array(all_labels)

    # os.makedirs(DATA_DIR, exist_ok=True)
    # np.save(os.path.join(DATA_DIR, 'features.npy'), X)
    # np.save(os.path.join(DATA_DIR, 'labels.npy'), y)
    print('\nFeatures and labels created and saved in \'data\' directory.')

if __name__ == '__main__':
    main()
