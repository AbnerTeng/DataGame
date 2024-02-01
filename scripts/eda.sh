#!/bin/bash
echo "EDA session"

echo "data list
1. label_test_source
2. label_train_source
3. label_train_target
4. meta_song_composer
5. meta_song_genre
6. meta_song_lyricist
7. meta_song_producer
8. meta_song_titletext
9. meta_song
"

read -p "Enter data to eda (type name) : " data

echo "EDA for $data"

python -m src.eda --path data/$data.parquet
