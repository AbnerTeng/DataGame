## Sequential Recommendation

Sequential Recommendation system for song based on traditional statistical language model (NGram, SkipGram), Neural Network Language Model (Gated Recurrent Unit), and LLM (T5)

## Module Structure

```plaintext
root/
  |──src/
  |   |── __init__.py
  |   |── main.py
  |   |── utils.py
  |   └── eda.py 
  |── data/
  |   |── label_test_source.parquet
  |   |── label_train_source
  |   |── label_train_target
  |   |── meta_song_composer
  |   |── meta_song_genre
  |   |── meta_song_lyricist
  |   |── meta_song_producer
  |   |── meta_song_titletext
  |   |── meta_song
  |   └── sample.csv
  |── .gitignore
  |── README.md
  |── requirements.txt
  └── eda.sh
```

## EDA

Execute `eda.sh` to generate EDA report for specific dataset.

```bash
chmod +x eda.sh
eda.sh
```

Preliminary and incomplete

- Stats LM: Need to Refactor
- NNLM: Developing
- LLM: Need to Refactor
