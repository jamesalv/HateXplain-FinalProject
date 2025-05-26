import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer
from typing import Dict, List, Tuple, Any
from sklearn.utils.class_weight import compute_class_weight
import json
import numpy as np

class HateSpeechDataset(Dataset):
    """PyTorch dataset for HateXplain data"""
    
    def __init__(self, texts: List[str], labels: List[str], tokenizer: PreTrainedTokenizer, max_length: int = 128):
        """
        Initialize HateSpeechDataset.
        
        Args:
            texts: List of text samples
            labels: List of labels for each sample
            tokenizer: Transformer tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = self._create_label_map()
        
    def _create_label_map(self) -> Dict[str, int]:
        """Create mapping from label text to numeric ID"""
        unique_labels = sorted(list(set(self.labels)))
        return {label: i for i, label in enumerate(unique_labels)}
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Convert label to numeric
        label_id = self.label_map[label]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }

def prepare_data_loaders(
    df: pd.DataFrame, 
    tokenizer: PreTrainedTokenizer, 
    batch_size: int = 16, 
    max_length: int = 128, 
    test_size: float = 0.1, 
    val_size: float = 0.1, 
    random_state: int = 42,
    custom_split: bool = False,
    custom_split_path: str = None,
    auto_weighted: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], pd.DataFrame]:
    """
    Prepare train, validation, and test data loaders.
    
    Args:
        df: Preprocessed dataframe
        tokenizer: Transformer tokenizer
        batch_size: Batch size for data loaders
        max_length: Maximum sequence length
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        custom_split: Whether to use a custom split
        custom_split_path: Path to custom split file (if custom_split is True)
        auto_weighted: Whether to use auto-weighted sampling
    Returns:
        train_dataloader, val_dataloader, test_dataloader, label_map, test_df
    """
    if(custom_split and custom_split_path is not None):
        # Load custom split
        with open(custom_split_path, 'r') as f:
            custom_split_data = json.load(f)
        
        train_df = df[df['post_id'].isin(custom_split_data['train'])]
        val_df = df[df['post_id'].isin(custom_split_data['val'])]
        test_df = df[df['post_id'].isin(custom_split_data['test'])]
    else: 
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['final_label'], random_state=random_state)
        train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df['final_label'], random_state=random_state)
    
    print(f"Train set: {len(train_df)}, Validation set: {len(val_df)}, Test set: {len(test_df)}")
    
    # Set class weights for auto-weighted sampling
    label_to_weight = {label: 1.0 for label in train_df['final_label'].unique()}
    if auto_weighted:
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array((train_df['final_label'].unique())),
            y=train_df['final_label']
        )
        
        # Create a mapping from label to weight
        label_to_weight = {label: weight for label, weight in zip(train_df['final_label'].unique(), class_weights)}
    
    # Create datasets
    train_dataset = HateSpeechDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['final_label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = HateSpeechDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['final_label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = HateSpeechDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['final_label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size
    )
    
    return train_dataloader, val_dataloader, test_dataloader, train_dataset.label_map, test_df, label_to_weight