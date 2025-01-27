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
- using `librosa` for feature extraction & visualiation of audio data
- using `pydub` for `.mp3` to `.wav`
- using `pytorch` for models
- using `lightning` for organization of training process
- other self explanatory libraries (i.e. `multiprocessing`)

### To Run
- run `python -m scripts.single-word-filter` from the ***home directory***
- run `python -m scripts.mp3towav` from the ***home directory***
- run `python -m scripts.feature-extraction` from the ***home directory***
- refer to the READ.me in `scripts` to run `resnet-testing.py` to train

### To Do/Future:
- Raytune for hyper parameter tuning
- Use wavelets for extracted features
- add analysis of model performance - PyTorch Lightning
  - https://lightning.ai/docs/pytorch/stable/
  - loss, acc etc graphs
  - tensorboard?
- Test using an RNN
- sentence segmentation
  - could use these as a validation dataset or just more training examples
  - need the tone tho which might kind of harder since much more words to classify - if i can get the tone classification good on this dataset then i could maybe use this to classify the tones for the new words? sandhis might get messy tho
- Test using some basic transformer architecture
  - i think i would need different features for this?

### FEB 10TH DEADLINE (self imposed)
- classify tone and pinyin for the inputted words
- essentialyl could have like 3 different classifiers
  - one to predict the tone
  - one to predict the vowel - (zh, ch, q, s, c, etc)
  - one to predict the consonants - (ou, ia, etc)
- to do this i will need much more data for different words
  - sentence segmentation is a must and data labeling too
