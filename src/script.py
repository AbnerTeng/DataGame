# %%
import pandas as pd
from tqdm import tqdm
# %%
test = pd.read_parquet('../data/label_test_source.parquet')
# %%
session_id = test['session_id'].unique()
session_grp = test.groupby('session_id')
# %%
lst = []
for session in tqdm(session_id):
    session_song = session_grp.get_group(session)['song_id'].unique().tolist()
    lst.append(len(session_song))
# %%
import matplotlib.pyplot as plt
# %%
plt.hist(lst, bins=100)
# %%
