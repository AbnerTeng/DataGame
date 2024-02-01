"""
general utils
"""
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a given path
    - param 
        path: path to data

    - return: 
        pandas dataframe
    """
    data = pd.read_parquet(path, engine="pyarrow")
    return data


def data_reshape(src_path: str, trg_path: str, _type: str) -> dict:
    """
    Preprocess data
    
    Origin data:
    
    +---------+------+
    | session | song |
    +---------+------+
    |   751   | aaaa |
    |   751   | bbbb |
    ...
    
    Output train data:
    {
        "session": 751,
        "source_song": List[str],
        "target_song": List[str]
    }
    Output test data:
    {
        "session": 751,
        "source_song": List[str]
    }
    """
    if _type == "train":
        src_data, trg_data = load_data(src_path), load_data(trg_path)
        grp_src = src_data.groupby("session_id")["song_id"].agg(list).reset_index()
        grp_trg = trg_data.groupby("session_id")["song_id"].agg(list).reset_index()
        grp_src.columns, grp_trg.columns = ["session", "source_song"], ["session", "target_song"]
        data = pd.merge(grp_src, grp_trg, on="session", how="left")


    elif _type == "test":
        src_data = load_data(src_path)
        grp_src = src_data.groupby("session_id")["song_id"].agg(list).reset_index()
        grp_src.columns = ["session", "source_song"]
        data = grp_src

    return data
    