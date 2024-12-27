### [WIP] chinese tone identification project
the data comes from common voice mozilla. i am using the Common Voice Corpus 20.0 which is around 21.21 GB

i am currently filtering out all sentences and only using single words to make the process easier at first. 
- will likely go back and add sentence handling
- ideally will also add more features in `feature-extraction.py` instead of just mfccs & spectral centroid
- also need to research more about what the features actually mean and which ones are useful - pipeline should remain fairly the same. minor changes to `feature-extraction.py` and input dimensions for CNN would be expected

directory structure is mainly based on `constants.py` which can be modified.

corpus directories should contain the .tsv files and `.mp3` or `.wav` files if you have dataset downloaded. i removed the `zh-CN` directory from path since its unnecessary

test contained some select `.wav` files for testing, not necessary

both `mp3towav.py` and `feature-extraction.py` use multiprocessing for speed boost. modify how many pools to create in `constants.py` - currently set to `os.cpu_count() - 2`

to run:
- use the `tone-recognition.ipynb` notebook up until it gets to exporting to wav
- run `python -m scripts.mp3towav.py` from the ***home directory***
- run `python -m scripts.feature-extraction.py` from the ***home directory***
- [WIP] building model  
