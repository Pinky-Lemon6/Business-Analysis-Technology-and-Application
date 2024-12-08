import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict
from pathlib import Path

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': label
        }


def load_amazon_reviews(file_path):
    reviews = []
    with open(file_path, 'r') as f:
        for line in f:
            reviews.append(json.loads(line))
    return pd.DataFrame(reviews)

def preprocess_reviews(df):
    df['reviewText'] = df['reviewText'].fillna('')
    df['reviewText'] = df['reviewText'].str.lower()
    df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '')
    return df




class DataManager:
    def __init__(self, config):
        self.config = config
        
    def load_data(self, data_path: Path) -> pd.DataFrame:
        reviews = []
        with open(data_path) as f:
            for line in f:
                reviews.append(json.loads(line))
        df = pd.DataFrame(reviews)
        return self.preprocess_data(df)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['reviewText'] = df['reviewText'].fillna('')
        df['reviewText'] = df['reviewText'].str.lower()
        df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '')
        return df
    
    def create_data_loaders(
        self, df: pd.DataFrame, tokenizer
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # Split data
        train_size = int(len(df) * self.config.TRAIN_RATIO)
        val_size = int(len(df) * self.config.VAL_RATIO)
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size+val_size]
        test_df = df[train_size+val_size:]
        
        # Create datasets
        train_dataset = ReviewDataset(
            train_df['reviewText'],
            train_df['class'],
            tokenizer,
            self.config.MAX_SEQUENCE_LENGTH
        )
        val_dataset = ReviewDataset(
            val_df['reviewText'],
            val_df['class'],
            tokenizer,
            self.config.MAX_SEQUENCE_LENGTH
        )
        test_dataset = ReviewDataset(
            test_df['reviewText'],
            test_df['class'],
            tokenizer,
            self.config.MAX_SEQUENCE_LENGTH
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE
        )
        
        return train_loader, val_loader, test_loader