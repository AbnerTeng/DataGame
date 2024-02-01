"""
Next song prediction using LSTM
"""
from argparse import ArgumentParser
import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from ..utils.data_utils import data_reshape


class SongDataset(Dataset):
    """
    Song id dataset
    """
    def __init__(self, songdata: pd.DataFrame) -> None:
        self.songdata = songdata
        self.source = self.songdata["source_song"]
        self.target = \
            self.songdata["target_song"] if "target_song" in self.songdata.columns else None


    def __len__(self) -> int:
        return len(self.songdata)


    def __getitem__(self, idx: int) -> dict:
        if self.target is None:
            return {
                "source": self.source[idx]
            }
        else:
            return {
                "source": self.source[idx],
                "target": self.target[idx]
            }


class SongLSTM(nn.Module):
    """
    LSTM model for song prediction
    -----
    Training session:
        input: label_train_source.parquet
        output: label_train_target.parquet
    
    Inference session:
        input: label_test_source.parquet
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int) -> None:
        super(SongLSTM, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True),
            nn.Linear(hidden_dim, output_dim)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.model(x)
        return x

def train(
        model, optimizer, criterion, train_loader: DataLoader, valid_loader: DataLoader, cfg: dict
    ) -> None:
    device = cfg["device"] if torch.backends.mps.is_available() else "cpu"
    for epoch in range(cfg["n_epochs"]):
        for batch in tqdm(train_loader):
            source, target = batch
            source, target = source.to(device), target.to(device)
            output = model(source)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                source, target = batch
                source, target = source.to(device), target.to(device)
                output = model(source)
                loss = F.cross_entropy(output, target)
            print(f"Epoch: {epoch}, Loss: {loss.item()}")


def parse_args() -> ArgumentParser:
    """
    parsing argument
    """
    parser = ArgumentParser()
    parser.add_argument("--dat_type", type=str, default="train")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open("config/lstm_v1.yaml", "r", encoding="utf-8") as yml:
        config = yaml.safe_load(yml)
    train_data = data_reshape(
        "../data/label_train_source.parquet",
        "../data/label_train_target.parquet",
        "train"
    )
    test_data = data_reshape(
        "../data/label_test_source.parquet",
        None,
        "test"
    )
    if args.dat_type == "train":
        train_dataset = SongDataset(train_data)
        train_size = int(len(train_dataset) * 0.8)
        valid_size = len(train_dataset) - train_size
        train_set, valid_set = random_split(train_dataset, [train_size, valid_size])
        train_loader, valid_loader = \
            DataLoader(train_set, batch_size=config['batch_size']), \
            DataLoader(valid_set, batch_size=config['batch_size'])
        lstm_model = SongLSTM(1, 128, 2, 1)
        adamw = torch.optim.AdamW(lstm_model.parameters(), lr=config['lr'])
        criterion = nn.CrossEntropyLoss()

    elif args.dat_type == "test":
        test_dataset = SongDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
