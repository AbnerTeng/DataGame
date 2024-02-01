import warnings
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

warnings.filterwarnings('ignore')


class SongDataset(Dataset):
    """
    Song dataset
    """
    def __init__(self, source_sequence, target_sequence) -> None:
        self.source_sequence = source_sequence
        if target_sequence is None:
            self.target_sequence = None
        self.target_sequence = target_sequence


    def __len__(self) -> int:
        return len(self.source_sequence["input_ids"])


    def __getitem__(self, idx: int) -> dict:
        if self.target_sequence is None:
            source_ids = torch.tensor(self.source_sequence['input_ids'][idx]).squeeze()
            return {
                'source': {
                    'input_ids': source_ids,
                    'attention_mask': torch.tensor(
                        self.source_sequence['attention_mask'][idx]
                    ).squeeze()
                }
            }
        else:
            source_ids = torch.tensor(self.source_sequence['input_ids'][idx]).squeeze()
            target_ids = torch.tensor(self.target_sequence['input_ids'][idx]).squeeze()
            return {
                'source': {
                    'input_ids': source_ids,
                    'attention_mask': torch.tensor(
                        self.source_sequence['attention_mask'][idx]
                    ).squeeze()
                },
                'target': {
                    'input_ids': target_ids,
                    'attention_mask': torch.tensor(
                        self.target_sequence['attention_mask'][idx]
                    ).squeeze()
                }
            }

def get_song_list(data: pd.DataFrame, _type: str) -> list:
    """
    get song list
    """
    if _type == "train":
        _len = 20
    elif _type == "test":
        _len = 5
    song = np.array(data.song_id)
    song = song.reshape(
        (
            len(song) // _len, _len
        )
    ).tolist()
    song = pd.DataFrame(
        {
            'song_seq': song
        }
    )
    song.song_seq = song.song_seq.apply(lambda x: ", ".join(x))
    return song


def train(model, tokenizer, dataloader, optimizer, device, args, early_stop=5):
    """
    training session
    """
    ## TODO: Write early stopping
    best_loss = 100
    tolerance = 0
    for epoch in tqdm(range(args.n_epoch)):
        model.train()
        loss_lst = []
        for batch in dataloader:
            inputs, targets = batch['source'], batch['target']
            outputs = model(**inputs, labels=targets['input_ids'].to(device))
            preds = torch.functional.F.softmax(outputs.logits, dim=-1).argmax(dim=-1)
            torch.cuda.empty_cache()
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_lst.append(loss.item())
        train_loss = sum(loss_lst) / len(loss_lst)
        if train_loss < best_loss:
            print(f"Epoch: {epoch + 1} | Loss: {train_loss} --> best")
            model.save_pretrained('model/t5_finetune')
            tokenizer.save_pretrained('model/t5_finetune')
            best_loss = train_loss
            tolerance = 0
        elif train_loss >= best_loss:
            print(f"Epoch: {epoch + 1} | Loss: {train_loss}")
            tolerance += 1
        if tolerance == early_stop:
            print("Early stopping, model is not improving...")
            break


def evaluate(model, tokenizer, dataloader, device, args):
    """
    Model evaluation
    """
    with torch.no_grad():
        model.eval()
        result = []
        for batch in tqdm(dataloader):
            inputs = batch['source']
            outputs = model.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                do_sample=args.do_sample,
                top_k=50,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=1
            )
            result.append(
                tokenizer.batch_decode(
                    outputs.tolist(),
                    skip_special_tokens=True
                )
            )
    return result


def collate_fn(batch):
    """
    collate function
    """
    has_target = 'target' in batch[0]
    batch_input_ids = torch.stack([item['source']['input_ids'] for item in batch])
    batch_attention_mask = torch.stack([item['source']['attention_mask'] for item in batch])

    if has_target:
        batch_target_ids = torch.stack([item['target']['input_ids'] for item in batch])
        batch_target_mask = torch.stack([item['target']['attention_mask'] for item in batch])
        return {
            'source': {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask
            },
            'target': {
                'input_ids': batch_target_ids,
                'attention_mask': batch_target_mask
            }
        }
    else:
        return {
            'source': {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask
            }
        }


def parse_args() -> argparse.ArgumentParser:
    """
    parsing arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, help="train or eval",
        default="eval"
    )
    parser.add_argument(
        "--do_sample", type=bool, help="do sample",
        default=False
    )
    parser.add_argument(
        "--n_epoch", type=int, help="Number of epochs to train",
        default=20
    )
    parser.add_argument(
        "--lr", type=float, help="learning rate",
        default=1e-5
    )
    parser.add_argument(
        "--exec", type=str, help="execute type",
        default="mps"
    )
    parser.add_argument(
        "--early_stop", type=int, help="early stopping",
        default=5
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_name = "google/flan-t5-small" if args.mode == "train" else "model/t5_finetune"
    if args.exec == "mps":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
    elif args.exec == "cuda":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device type: {device}")
    t5tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=True)
    t5model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    if args.mode == "train":
        if args.do_sample:
            train_data = pd.read_parquet('data/label_train_source.parquet').iloc[:1000, :]
            test_data = pd.read_parquet('data/label_train_target.parquet').iloc[:250, :]
        else:
            train_data = pd.read_parquet('data/label_train_source.parquet').iloc[:, :]
            test_data = pd.read_parquet('data/label_train_target.parquet').iloc[:, :]
        print("Getting Song list...")
        train_song, test_song = get_song_list(train_data, "train"), get_song_list(test_data, "test")
        print(f"train song shape: {train_song.shape}")
        print(f"test song shape: {test_song.shape}")
        print("===================================")
        print("Tokenizing...")
        train_token = t5tokenizer(
            train_song.song_seq.tolist(), return_tensors='pt', padding=True, max_length=768
        ).to(device)
        test_token = t5tokenizer(
            test_song.song_seq.tolist(), return_tensors='pt', padding=True, max_length=768
        ).to(device)
        song_dataset = SongDataset(train_token, test_token)
        song_dataloader = DataLoader(song_dataset, batch_size=4, shuffle=True)
        adamw = torch.optim.AdamW(t5model.parameters(), lr=args.lr)
        print("===================================")
        print("Start Training...")
        train(
            model=t5model,
            tokenizer=t5tokenizer,
            dataloader=song_dataloader,
            optimizer=adamw,
            device=device,
            args=args
        )
    if args.mode == "eval":
        if args.do_sample:
            new_test = pd.read_parquet('data/label_test_source.parquet').iloc[:1000, :]
        else:
            new_test = pd.read_parquet('data/label_test_source.parquet')
        print("Getting Song list...")
        new_test_song = get_song_list(new_test, "train")
        print(f"new test song shape: {new_test_song.shape}")
        print("===================================")
        print("Tokenizing...")
        new_test_token = t5tokenizer(
            new_test_song.song_seq.tolist(), return_tensors='pt', padding=True, max_length=768
        ).to(device)
        print(new_test_token['input_ids'].shape) ## torch.Size([50, 505])
        song_dataset_test = SongDataset(new_test_token, None) ## idx can traverse from 0 to 49
        print(len(song_dataset_test))
        song_dataloader_test = DataLoader(song_dataset_test, collate_fn=collate_fn, batch_size=4)
        print(len(song_dataloader_test))
        print("===================================")
        print("Start Evaluation...")
        rec_result = evaluate(
            model=t5model,
            tokenizer=t5tokenizer,
            dataloader=song_dataloader_test,
            device=device,
            args=args
        )
        rec_result = [item for sublist in rec_result for item in sublist]
        rec_result = pd.DataFrame(rec_result, columns=['song_seq'])
        rec_result.to_csv('data/submission.csv', index=False)
