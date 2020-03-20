import pandas as pd
from Constant import CACHE_PATH
import os

path = os.path.join(CACHE_PATH, 'FOMC_token_separated_col.xlsx')
df = pd.read_excel(path)

FOMC1 = df[df['Section'] == 1]
FOMC2 = df[df['Section'] == 2]

word_list = []
for line in FOMC1['content']:
    try:
        words = line.split(' ')
        for word in words:
            if word not in word_list:
                word_list.append(word)
    except AttributeError:
        pass