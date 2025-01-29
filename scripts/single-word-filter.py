import pandas as pd
from utils.constants import RESULTS_PATH

from pypinyin import lazy_pinyin, Style
import re

def breakdown_pinyin(phrase):
  clean_phrase = re.sub(r'[^\w]', '', phrase)
  initial = lazy_pinyin(clean_phrase, style=Style.INITIALS, strict=False)[0]
  final = lazy_pinyin(clean_phrase, style=Style.FINALS, strict=False)[0]
  tone = [ word[-1] for word in lazy_pinyin(clean_phrase, style=Style.FINALS_TONE3, strict=False, neutral_tone_with_five=True, tone_sandhi=True)][0]

  return (initial if initial != '' else "EMPTY", final, str(tone))


def main():
  # Specify dtype for columns to avoid DtypeWarning
  dtype_spec = {'column4': str, 'column8': str, 'column12': str}
  df = pd.read_csv('./large-corpus/other.tsv', sep='\t', dtype=dtype_spec, low_memory=False)
  mdf = df[['path', 'sentence']]  # make a copy to avoid the SettingWithCopyWarning

  mdf = mdf[mdf['sentence'].str.len() == 1]

  value_counts = mdf['sentence'].value_counts()
  filtered_value_counts = value_counts[value_counts >= 16]

  # Filter the original DataFrame to only include the rows where the 'sentence' column is in the filtered values
  result = mdf[mdf['sentence'].isin(filtered_value_counts.index)].copy()
  result.loc[:, 'wav_path'] = result['path'].apply(lambda x: x.split('_')[-1][:-4] + ".wav")

  # map characters to pinyin - inits, finals, tones
  labels = result['sentence'].apply(lambda x: pd.Series(breakdown_pinyin(x))).rename(columns={0: 'initial', 1: 'final', 2: 'tone'})
  newDF = pd.concat([result, labels], axis=1)

  newDF = newDF.reset_index(drop=True)

  newDF.to_csv(RESULTS_PATH, index=False)


if __name__ == '__main__':
  main()
