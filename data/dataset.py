import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
from sklearn.utils.class_weight import compute_class_weight
import json
import numpy as np

class HateXplainDataset(Dataset):
    def __init__(self, df, task="3-class"):
        self.data = df.reset_index(drop=True)  # Reset index for cleaner access
        self.task = task
        self.label_map = self._create_label_map()
        print(f"Created dataset with {len(self.data)} samples")
        print(f"Label mapping: {self.label_map}")
        
    def _create_label_map(self) -> Dict[str, int]:
        """Create mapping from label text to numeric ID"""
        unique_labels = sorted(list(set(self.data['final_label'])))
        return {label: i for i, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the data at index from the dataframe
        row = self.data.iloc[index]
        
        # Extract pre-tokenized inputs and ensure they're tensors
        input_ids = row["inputs"]["input_ids"]
        attention_mask = row["inputs"]["attention_mask"]
        
        # Handle different tensor formats
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() > 1:
                input_ids = input_ids.squeeze(0)  # Remove batch dimension if present
        else:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            
        if isinstance(attention_mask, torch.Tensor):
            if attention_mask.dim() > 1:
                attention_mask = attention_mask.squeeze(0)  # Remove batch dimension if present
        else:
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Handle rationales
        rationales = row["rationales"]
        if isinstance(rationales, torch.Tensor):
            if rationales.dim() > 1:
                rationales = rationales.squeeze(0)  # Remove batch dimension if present
        else:
            rationales = torch.tensor(rationales, dtype=torch.float32)

        # Get label and convert to numeric ID
        label_text = row['final_label']
        label = torch.tensor(self.label_map[label_text], dtype=torch.long)
        
        # Debug print for first few samples (remove after testing)
        # if index < 3:
        #     print(f"Sample {index}:")
        #     print(f"  Input IDs shape: {input_ids.shape}")
        #     print(f"  Attention mask shape: {attention_mask.shape}")
        #     print(f"  Rationales shape: {rationales.shape}")
        #     print(f"  Label: {label_text} -> {label.item()}")
        #     print(f"  Rationales sample: {rationales[:10]}")  # First 10 values
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "rationales": rationales,
            "labels": label,  # Changed from "label" to "labels" to match your training code
        }

def prepare_data_loaders(
    df: pd.DataFrame, 
    batch_size: int = 16, 
    test_size: float = 0.1, 
    val_size: float = 0.1, 
    random_state: int = 42,
    custom_split: bool = False,
    custom_split_path: str = None,
    auto_weighted: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], pd.DataFrame, Dict[str, float]]:
    """
    Prepare train, validation, and test data loaders.
    
    Args:
        df: Preprocessed dataframe
        batch_size: Batch size for data loaders
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        custom_split: Whether to use a custom split
        custom_split_path: Path to custom split file (if custom_split is True)
        auto_weighted: Whether to use auto-weighted sampling
        
    Returns:
        train_dataloader, val_dataloader, test_dataloader, label_map, test_df, label_to_weight
    """
    if custom_split and custom_split_path is not None:
        # Load custom split
        with open(custom_split_path, 'r') as f:
            custom_split_data = json.load(f)
        
        train_df = df[df['post_id'].isin(custom_split_data['train'])]
        val_df = df[df['post_id'].isin(custom_split_data['val'])]
        test_df = df[df['post_id'].isin(custom_split_data['test'])]
    else: 
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['final_label'], 
            random_state=random_state
        )
        train_df, val_df = train_test_split(
            train_df, 
            test_size=val_size, 
            stratify=train_df['final_label'], 
            random_state=random_state
        )
    
    print(f"Train set: {len(train_df)}, Validation set: {len(val_df)}, Test set: {len(test_df)}")
    
    # Set class weights for auto-weighted sampling
    label_to_weight = {label: 1.0 for label in train_df['final_label'].unique()}
    if auto_weighted:
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(sorted(train_df['final_label'].unique())),  # Ensure consistent order
            y=train_df['final_label']
        )
        
        # Create a mapping from label to weight
        unique_labels = sorted(train_df['final_label'].unique())
        label_to_weight = {label: weight for label, weight in zip(unique_labels, class_weights)}
        print(f"Class weights: {label_to_weight}")
    
    # Create datasets
    train_dataset = HateXplainDataset(
        df=train_df,
        task="3-class" if len(train_df['final_label'].unique()) == 3 else "binary"
    )
    
    val_dataset = HateXplainDataset(
        df=val_df,
        task="3-class" if len(val_df['final_label'].unique()) == 3 else "binary"
    )
    
    test_dataset = HateXplainDataset(
        df=test_df,
        task="3-class" if len(test_df['final_label'].unique()) == 3 else "binary"
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size,
    )
    
    return train_dataloader, val_dataloader, test_dataloader, train_dataset.label_map, test_df, label_to_weight