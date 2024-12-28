## [WIP] chinese tone identification project
the data comes from common voice mozilla. i am using the Common Voice Corpus 20.0 which is around 21.21 GB

### Current Work
- currently filtering out all sentences and only using single words to make the process easier at first. (only around 2.3 GB of clips to process)
  - will likely go back and add sentence handling
  - ideally will also add more features in `feature-extraction.py` instead of just mfccs & spectral centroid
  - also need to research more about what the features actually mean and which ones are useful
  - pipeline should remain fairly the same. minor changes to `feature-extraction.py` and input dimensions for CNN would be expected
- eventually should make the whole pipeline smoother but it works for now

### Directory Structure
directory structure is mainly based on `constants.py` which can be modified.

- corpus directories should contain the `.tsv` files and `.mp3` or `.wav` files if you have dataset downloaded. i removed the `zh-CN` directory from path since its unnecessary
- the `test` directory contains some select `.wav` files for testing, not necessary
- the `data` direcotry is where the modified database (`results.csv`) & extracted features and labels numpy files (`features.npy` and `labels.npy`) are stored

### Scripts
- both `mp3towav.py` and `feature-extraction.py` use multiprocessing for speed boost
- modify how many pools to create in `constants.py` - currently set to `os.cpu_count() - 2`

### Libraries
only using a few main libraries which are in `environment.yaml`. 
- using `librosa` for feature extraction & visualiation of audio data
- using `pydub` for `.mp3` to `.wav`
- using `pytorch` for cnn model
- other self explanatory libraries (i.e. `multiprocessing`)

### To Run
- use the `tone-recognition.ipynb` notebook up until it gets to exporting to wav
- run `python -m scripts.mp3towav.py` from the ***home directory***
- run `python -m scripts.feature-extraction.py` from the ***home directory***
- [WIP] building model  
