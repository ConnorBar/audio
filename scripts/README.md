### Raw .wav File Augments
none yet, can do any of these:
- time shift: need to be careful though to make sure am not splitting the word itself
- pitch shift: modifies the frequency of part of sound
- time stretch: make it slower or faster
- add noise: makes it a little more noisy for model to learn from

### Augmentation
- added proportion based sampling for augmentation selection


### Spectrogram Augments
- as of right now i just standardize the mfccs
- can also look into wavelets as opposed to mfccs

none yet, can do any of these:
- Frequency mask
- Time mask
Both of these block out certain bands of the audio making it harder to identify


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



### thoughts:

mimic boosting, pick random samples and choose how to augment them and how many to make

have an algorithmic approach to augmenting, so like for second tones which are more rare, i can make way more examples for