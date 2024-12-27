### [WIP] chinese tone identification project
the data comes from common voice mozilla. i am using the Common Voice Corpus 20.0 which is around 21.21 GB

i am currently filtering out all sentences and only using single words to make the process easier at first. 
- will likely go back and add sentence handling
- ideally will also add more features in `feature-extraction.py` instead of just mfccs & spectral centroid

directory structure is mainly based on `constants.py` which can be modified

to run:
- use the `tone-recognition.ipynb` notebook up until it gets to exporting to wav
- run `python -m scripts.mp3towav.py`
- run `python -m scripts.feature-extraction.py`
- will get to the rest of it
