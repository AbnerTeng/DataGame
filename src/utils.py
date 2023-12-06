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