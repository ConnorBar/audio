### Augmentation
- added proportion based sampling for augmentation selection
- current augs are just on the raw wav, not on the extracted feats yet

### Spectrogram Augments
- as of right now i just standardize the mfccs
- can also look into wavelets as opposed to mfccs

none yet, can do any of these:
- Frequency mask
- Time mask
Both of these block out certain bands of the audio making it harder to identify, need to careful not to remove too much info


### Extraction tools 
On Mac, make a file named `tones` -- do not name is `tones.sh`, just `tones` -- and place this is `/usr/local/bin/`. It will likely need sudo access and you need to run `chmod +x tones` as well. Modify the path as needed

```
#!/bin/bash

case "$1" in
  --extract)
    shift
    cd /Users/{cb/work/Personal}/chinese_voice && python -m scripts.feature-extraction
    ;;
  --test)
    shift
    cd /Users/cb/work/Personal/chinese_voice && python -m scripts.testing
    ;;
  *)
    echo "Usage: ch --extract | --test"
    ;;
esac
```

### training:
```
python -m scripts.train-resnet --accelerator 'gpu' --devices 1 --model 'resnet
```