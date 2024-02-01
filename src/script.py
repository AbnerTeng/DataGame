# %%
import pandas as pd
# %%
train_source = pd.read_parquet("../data/label_train_source.parquet")
train_target = pd.read_parquet("../data/label_train_target.parquet")
test_source = pd.read_parquet("../data/label_test_source.parquet")
# %%
print(f"train source shape: {train_source.shape}")
print(f"train target shape: {train_target.shape}")
print(f"test source shape: {test_source.shape}")
# %%
import numpy as np
song = np.array(train_source.song_id)
song = song.reshape(
    (
        len(song) // 20, 20
    )
).tolist()
song = pd.DataFrame(
    {
        'song_seq': song
    }
)
song.song_seq = song.song_seq.apply(lambda x: ", ".join(x))
# %%
