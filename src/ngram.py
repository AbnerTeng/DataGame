"""
Recommendation system using ngram
"""
import argparse
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm


class NGram:
    """
    ngram language model
    """
    def __init__(self, dataset_path: str, n: int, num_generate: int=5) -> None:
        self.n = n
        self.num_generate = num_generate
        self.train_data = pd.read_parquet("data\label_train_source.parquet")
        self.train_result = pd.read_parquet("data\label_train_target.parquet")
        self.first20 = pd.read_parquet(dataset_path)
        self.session_group = self.first20.groupby('session_id')
        self.unique_sessions = list(self.session_group.groups.keys())
        self.all_songs = self.first20['song_id'].tolist()
        self.most_popular_song = Counter(self.all_songs).most_common()[0][0]

    def songlist_split(self, df:pd.DataFrame) -> list:
        songlist = df.groupby("session_id")["song_id"].apply(list).tolist()
        songlist_split = []
        for i in range(len(songlist)):
            songlist_split +=  songlist[i]
            songlist_split.extend(["na"] * (self.n-1))
        return songlist_split

    def find_ngrams(self) -> dict[tuple, list]:
        """
        get ngrams from all songs
        """
        songlist = self.songlist_split(self.train_data)+self.songlist_split(self.first20)
        ngrams = defaultdict(list)
        for i in range(len(songlist) - self.n + 1):
            ngram_key = tuple(songlist[i: i + self.n - 1])
            if i + self.n < len(songlist) and songlist[i + self.n-1] not in songlist[i: i + self.n - 1] and songlist[i + self.n-1] != "na":
                ngrams[ngram_key].append(songlist[i + self.n - 1])
        return ngrams


    def count_ngrams(self, session: list, ngrams: dict) -> str:
        """
        count ngrams
        """
        if ngrams[tuple(session[-self.n+1: ])] != []:
            if len(set(session)) <= 5:
                most_common = Counter(session).most_common()[0][0]
            elif len(set(session)) > 5:
                most_common = Counter(ngrams[tuple(session[-self.n+1: ])]).most_common()[0][0]
        else:
            most_common = self.most_popular_song
        return most_common


    def iterate(self, session: list, ngrams: dict) -> list:
        """
        generate 5 songs iteratively
        """
        generate_lst = []
        for _ in range(self.num_generate):
            most_common = self.count_ngrams(session, ngrams)
            session = session[1:] + [most_common]
            generate_lst.append(most_common)
        return generate_lst


    def merge_with_session_id(self, ngrams: dict) -> pd.DataFrame:
        """
        merge with session id
        """
        full_rec = []
        for session_id in tqdm(self.unique_sessions):
            session_song = self.session_group.get_group(session_id)['song_id'].tolist()
            gen_list = self.iterate(session_song, ngrams)
            full_rec.append([session_id] + gen_list)
        return pd.DataFrame(full_rec)


    def join_with_sample(self, sample_path: str, opt_data: pd.DataFrame) -> pd.DataFrame:
        """
        join with sample
        """
        opt_data = opt_data.rename(
            columns={
                0: 'session_id',
                1: 'top1',
                2: 'top2',
                3: 'top3',
                4: 'top4',
                5: 'top5'
            }
        )
        sample_data = pd.read_csv(sample_path)
        sample_data = sample_data.merge(
            opt_data,
            on='session_id',
            how='inner'
        )
        sample_data.drop(
            columns=['top1_x', 'top2_x', 'top3_x', 'top4_x', 'top5_x'],
            inplace=True
        )
        sample_data = sample_data.rename(
            columns={
                'top1_y': 'top1',
                'top2_y': 'top2',
                'top3_y': 'top3',
                'top4_y': 'top4',
                'top5_y': 'top5'
            }
        )
        return sample_data


def parse_args() -> argparse.ArgumentParser:
    """
    parsing arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=2, help="ngram"
    )
    return parser.parse_args()


def num2str(num: int) -> str:
    """
    convert number to string
    """
    _map = {
        1: 'uni',
        2: 'bi',
        3: 'tri'
    }
    return _map[num]


if __name__ == "__main__":
    args = parse_args()
    rec = NGram(
        'data/label_test_source.parquet',
        n=args.n
    )
    ngrams_opt = rec.find_ngrams()
    data = rec.merge_with_session_id(ngrams_opt)
    data = rec.join_with_sample('data/sample.csv', data)
    data.to_csv(f'data/sub_{num2str(args.n)}gram.csv', index=False)
    print(f"sub_{num2str(args.n)}gram.csv is saved")
