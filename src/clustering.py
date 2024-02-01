"""
Song clustering by metadata
"""
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


class Clustering:
    """
    Clustering class
    """
    def __init__(self) -> None:
        self.train_data = pd.read_parquet('data/label_train_source.parquet')
        self.song_data = pd.read_parquet('data/meta_song.parquet')
        self.title_data = pd.read_parquet('data/meta_song_titletext.parquet')
        self.genre_data = pd.read_parquet('data/meta_song_genre.parquet')


    def merge_data(self) -> pd.DataFrame:
        """
        merge all data with song_id as key
        """
        result = pd.merge(
            self.song_data, self.title_data, on='song_id', how='left'
        )
        result = pd.merge(
            result, self.genre_data, on='song_id', how='left'
        )
        return result


    def preprocess_data(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        preprocess data
        """
        result = result[result['album_month'].notna()]
        result = result[result['song_length'].notna()]
        result['album_month'] = result['album_month'].apply(
            lambda x: datetime.strptime(x, "%Y-%m").timestamp()
        )
        label_encoder = LabelEncoder()
        result['genre_id'] = label_encoder.fit_transform(result['genre_id'])
        result['title_text_id'] = label_encoder.fit_transform(result['title_text_id'])
        return result


    def modling(self, result: pd.DataFrame) -> None:
        """
        clustering model
        """
        model = KMeans( n_clusters=15, random_state=42, n_init="auto").fit(
            result.drop(columns='song_id')
        )
        result['label'] = model.labels_
        print(result.head())
        result.to_parquet('data/song_cluster.parquet')


if __name__ == "__main__":
    clustering = Clustering()
    result_df = clustering.merge_data()
    result_df = clustering.preprocess_data(result_df)
    clustering.modling(result_df)