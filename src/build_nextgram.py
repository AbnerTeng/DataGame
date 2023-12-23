from ngram import NGram
import argparse
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm

class NGram2(NGram):

    def __init__(self, dataset_path: str, n: int, num_generate: int=5) -> None:
        super().__init__(dataset_path, n, num_generate)

    def count_ngrams(self, session: list, bynumber: int=1) -> str:
        """
        count ngrams
        """
        if bynumber == 1:
            counter = Counter(self.next_ngrams[0][tuple(session[-1:])])
            return counter
        else:
            counter_temp = Counter(self.next_ngrams[bynumber-1][tuple(session[-bynumber:-bynumber+1])])
            return counter_temp

    def nextisong_csv(self,folder_name,bynumber):
        import os
        import csv

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        csv_file_path = os.path.join(folder_name, f'next{bynumber}_songs.parquet')

        data = []
        for session_id in tqdm(self.unique_sessions):
            session_song = self.session_group.get_group(session_id)['song_id'].tolist()
            counter = self.count_ngrams(session_song, bynumber)
            for key, value in counter.items():
                data.append((session_song[-1:][0], key, value))
        df = pd.DataFrame(data, columns=['song_id', f'next{bynumber}_songs','count'])

        df.to_parquet(csv_file_path, index=False)
            
        
def parse_args() -> argparse.ArgumentParser:
    """
    parsing arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=2, help="ngram"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = NGram2('data/label_test_source.parquet',n=args.n)
    bynumber = 5
    model.build_ngrams(bynumber=bynumber)
    for i in range(1,bynumber+1):
        model.nextisong_csv("next_ngrams",bynumber=i)