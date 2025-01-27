import pandas as pd
from utils.constants import RESULTS_PATH

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

sentence_to_tone = {
  '一': 1, '八': 1, '六': 4, '三': 3, '七': 1, '五': 3,
  '四': 4, '是': 4, '九': 3, '零': 2, '二': 4, '否': 3
}

result.loc[:, 'tone'] = result['sentence'].map(sentence_to_tone)

word_labels = { word: i for i, word in enumerate(list(result['sentence'].unique())) }
result.loc[:, 'word_label'] = result['sentence'].map(word_labels)

result = result.reset_index(drop=True)

result.to_csv(RESULTS_PATH, index=False)