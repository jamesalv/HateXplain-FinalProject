"""Efficiency analysis utilities for hate speech classification models."""

import time
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

from models.classifier import TransformerClassifier
from data.dataset import HateSpeechDataset

def analyze_efficiency(
    models_to_compare: List[str], 
    data_df: pd.DataFrame, 
    num_classes: int = 3, 
    batch_size: int = 16,
    test_size: float = 0.05,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Analyze computational efficiency of models
    
    Args:
        models_to_compare: List of model names to compare
        data_df: DataFrame with the dataset
        num_classes: Number of classes (2 or 3)
        batch_size: Batch size for inference
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with efficiency results
    """
    print(f"\n=== Efficiency Analysis (num_classes={num_classes}) ===\n")
    
    efficiency_results = []
    
    # Prepare a small test set for efficiency testing
    _, test_df = train_test_split(data_df, test_size=test_size, random_state=random_state)
    
    for model_name in models_to_compare:
        print(f"\nAnalyzing {model_name}...")
        
        # Initialize model
        classifier = TransformerClassifier(model_name, num_classes)
        
        # Prepare test data
        test_dataset = HateSpeechDataset(
            texts=test_df['text'].tolist(),
            labels=test_df['final_label'].tolist(),
            tokenizer=classifier.tokenizer
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Measure model size
        model_size = sum(p.numel() for p in classifier.model.parameters())
        
        # Warmup
        classifier.predict(DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        ))
        
        # Measure inference time
        start_time = time.time()
        classifier.predict(test_dataloader)
        inference_time = time.time() - start_time
        
        # Calculate inference per second
        samples_per_second = len(test_df) / inference_time
        
        # Record results
        efficiency_results.append({
            'Model': model_name.split('/')[-1],
            'Parameters (M)': model_size / 1e6,
            'Inference Time (s)': inference_time,
            'Samples/Second': samples_per_second
        })
    
    # Create DataFrame and display results
    efficiency_df = pd.DataFrame(efficiency_results)
    efficiency_df = efficiency_df.sort_values('Samples/Second', ascending=False)
    
    print("\nEfficiency Results:")
    print(efficiency_df)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot inference speed
    plt.subplot(1, 2, 1)
    plt.bar(efficiency_df['Model'], efficiency_df['Samples/Second'])
    plt.title('Inference Speed')
    plt.xlabel('Model')
    plt.ylabel('Samples/Second')
    plt.xticks(rotation=45, ha='right')
    
    # Plot model size
    plt.subplot(1, 2, 2)
    plt.bar(efficiency_df['Model'], efficiency_df['Parameters (M)'])
    plt.title('Model Size')
    plt.xlabel('Model')
    plt.ylabel('Parameters (M)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return efficiency_df