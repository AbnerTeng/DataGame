import os
from collections import Counter
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
from .ngram import NGram


class NGram2(NGram):
    """
    New NGram
    """
    def __init__(self, dataset_path: str, n: int, num_generate: int=5) -> None:
        super(NGram2).__init__(dataset_path, n, num_generate)


    def count_ngrams(self, session: list, bynumber: int=1) -> str:
        """
        count ngrams
        """
        if bynumber == 1:
            counter = Counter(self.next_ngrams[0][tuple(session[-1:])])
            return counter
        else:
            counter_temp = Counter(
                self.next_ngrams[bynumber-1][tuple(session[-bynumber: -bynumber + 1])]
            )
            return counter_temp


    def nextisong_csv(self, folder_name: str, bynumber: int) -> None:
        """
        pred next song
        """
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        csv_file_path = os.path.join(folder_name, f'next{bynumber}_songs.parquet')

        data = []
        for session_id in tqdm(self.unique_sessions):
            session_song = self.session_group.get_group(session_id)['song_id'].tolist()
            counter = self.count_ngrams(session_song, bynumber)
            for key, value in counter.items():
                data.append((session_song[-bynumber:][0], key, value))
        df = pd.DataFrame(data, columns=['song_id', f'next{bynumber}_songs','count'])
        df.to_parquet(csv_file_path, index=False)


def parse_args() -> ArgumentParser:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=2, help="ngram"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = NGram2('data/label_test_source.parquet',n=args.n)
    BYNUMBER = 5
    model.build_ngrams(bynumber = BYNUMBER)
    for i in range(1, BYNUMBER + 1):
        model.nextisong_csv("next_ngrams", bynumber=i)
