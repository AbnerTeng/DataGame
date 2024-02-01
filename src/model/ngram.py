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
        self.train_data = pd.read_parquet("data/label_train_source.parquet")
        self.train_result = pd.read_parquet("data/label_train_target.parquet")
        self.first20 = pd.read_parquet(dataset_path)
        self.session_group = self.first20.groupby('session_id')
        self.unique_sessions = list(self.session_group.groups.keys())
        self.all_songs = self.first20['song_id'].tolist()
        self.most_popular_song = [
            song_name for song_name, _ in Counter(self.all_songs).most_common()
        ]
        self.count = -1
        self.next_ngrams = []


    def songlist_split(self, df:pd.DataFrame) -> list:
        """
        split songlist
        """
        self.songlist = df.groupby("session_id")["song_id"].apply(list).tolist()
        songlist_split = []
        for _, song in enumerate(self.songlist):
            songlist_split += song
            songlist_split.extend(["na"] * (self.n-1))
        return songlist_split


    def find_ngrams(self, _next: int=1) -> dict[tuple, list]:
        """
        get ngrams from all songs
        """
        ngrams = defaultdict(list)
        for i in range(len(self.songlist) - _next):
            ngram_key = tuple(self.songlist[i: i+1])
            if (i + _next < len(self.songlist)) \
                and (self.songlist[i + _next] not in self.songlist[i]) \
                and self.songlist[i + _next] != "na":
                ngrams[ngram_key].append(self.songlist[i + _next])
        return ngrams


    def build_ngrams(self,bynumber:int=1):
        self.bynumber = bynumber
        for i in tqdm(range(1,bynumber+1)):
            print(f"\nBuilding next{i}_bigrams")
            result = self.find_ngrams(i)
            self.next_ngrams.append(result)


    def count_ngrams(self, session: list, bynumber: int=1) -> str:
        """
        count ngrams
        """
        counter = Counter(self.next_ngrams[0][tuple(session[-1:])])
        if bynumber > 1:
            for i in range(1, bynumber):
                counter.update(self.next_ngrams[i][tuple(session[-1-i:-i])])

        if self.next_ngrams[0][tuple(session[-1: ])] != []:
            if len(set(session)) <= self.num_generate:
                most_common = Counter(session).most_common()[0][0]
            elif len(set(session)) > self.num_generate:
                most_common = counter.most_common()[0][0]
        else:
            most_common = self.most_popular_song[self.count]
            self.count -= 1
        return most_common


    def iterate(self, session: list,  bynumber: int=1) -> list:
        """
        generate 5 songs iteratively
        """
        generate_lst = []
        for _ in range(self.num_generate):
            most_common = self.count_ngrams(session,bynumber)
            session = session[1:] + [most_common]
            generate_lst.append(most_common)

        #replace repeat like ABABA with coverage
        generate_lst = list(set(generate_lst))
        while len(generate_lst) < self.num_generate:
            generate_lst.append(self.most_popular_song[self.count])
            self.count -= 1
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
    BYNUMBER = 5
    rec.build_ngrams(BYNUMBER)
    data = rec.merge_with_session_id(BYNUMBER)
    data = rec.join_with_sample('data/sample.csv', data)
    data.to_csv(f'data/sub_{num2str(args.n)}gram.csv', index=False)
    print(f"sub_{num2str(args.n)}gram.csv is saved")
