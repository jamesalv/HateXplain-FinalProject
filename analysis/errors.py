import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Union, Optional

from models.classifier import TransformerClassifier
from data.dataset import HateXplainDataset

def analyze_errors(
    results: Dict[str, Dict[str, Any]], 
    data_df: pd.DataFrame, 
    task_type: str = '3-class', 
    model_name: Optional[str] = None,
    model_dir: str = "saved_models"
) -> pd.DataFrame:
    """
    Analyze error patterns in model predictions
    
    Args:
        results: Dictionary with model results
        data_df: DataFrame with the dataset
        task_type: Type of classification task ('binary' or '3class')
        model_name: Specific model to analyze (if None, use the best model)
        model_dir: Directory where models are saved
        
    Returns:
        DataFrame with incorrectly classified examples
    """
    # If model_name not provided, select best model based on F1 score
    if model_name is None:
        best_f1 = -1
        for name, result in results[task_type].items():
            if result['metrics']['f1_score'] > best_f1:
                best_f1 = result['metrics']['f1_score']
                model_name = name
    
    print(f"\n=== Error Analysis for {model_name} on {task_type} Classification ===\n")
    
    # Load the model
    model_type = "binary" if task_type == 'binary' else "3class"
    model_path = f"{model_dir}/{model_name.replace('/', '_')}_{model_type}"
    
    # Initialize model
    num_classes = 2 if task_type == 'binary' else 3
    classifier = TransformerClassifier(model_name, num_classes)
    
    # Load saved model
    classifier.load_model(model_path)
    
    # Prepare test data (10% of dataset)
    _, test_df = train_test_split(data_df, test_size=0.1, stratify=data_df['final_label'], random_state=42)
    
    # Create test dataset
    test_dataset = HateXplainDataset(
        test_df,
        task_type=task_type,
    )
    
    # Create test dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )
    
    # Get predictions
    predictions, true_labels, probabilities, attention_weights = classifier.predict(test_dataloader)
    
    # Create label map
    label_map = test_dataset.label_map
    inv_label_map = {v: k for k, v in label_map.items()}
    
    # Add predictions to dataframe
    test_df = test_df.copy()
    test_df['predicted'] = [inv_label_map[p] for p in predictions]
    test_df['true'] = [inv_label_map[l] for l in true_labels]
    test_df['correct'] = test_df['predicted'] == test_df['true']
    
    # Add probability information
    for i, label in enumerate(label_map.keys()):
        test_df[f'prob_{label}'] = [probs[i] for probs in probabilities]
    
    # Find incorrectly classified examples
    incorrect_df = test_df[~test_df['correct']]
    
    # Calculate confidence scores
    if task_type == 'binary':
        # For binary classification, use probability of predicted class
        incorrect_df['confidence'] = incorrect_df.apply(
            lambda row: row[f'prob_{row["predicted"]}'], axis=1
        )
    else:
        # For multi-class, use max probability
        incorrect_df['confidence'] = incorrect_df.apply(
            lambda row: max([row[f'prob_{label}'] for label in label_map.keys()]), axis=1
        )
    
    # Sort by confidence (high to low)
    high_conf_errors = incorrect_df.sort_values('confidence', ascending=False).head(10)
    
    print("High-confidence errors (model was confident but wrong):")
    for i, (_, row) in enumerate(high_conf_errors.iterrows()):
        print(f"\nExample {i+1}:")
        print(f"Text: {row['text']}")
        print(f"True label: {row['true']}")
        print(f"Predicted label: {row['predicted']}")
        print(f"Confidence: {row['confidence']:.4f}")
        if 'target_groups' in row and row['target_groups']:
            print(f"Target groups: {', '.join(row['target_groups'])}")
    
    # Analyze error patterns by class
    print("\nError distribution by true class:")
    error_by_class = incorrect_df.groupby('true')['text'].count()
    total_by_class = test_df.groupby('true')['text'].count()
    error_rate_by_class = (error_by_class / total_by_class * 100).fillna(0)
    
    for class_name, error_rate in error_rate_by_class.items():
        print(f"{class_name}: {error_rate:.2f}% error rate ({error_by_class.get(class_name, 0)} of {total_by_class.get(class_name, 0)})")
    
    # Analyze confusion patterns
    print("\nConfusion patterns:")
    confusion_patterns = incorrect_df.groupby(['true', 'predicted'])['text'].count().sort_values(ascending=False)
    
    for (true_label, pred_label), count in confusion_patterns.items():
        print(f"{true_label} misclassified as {pred_label}: {count} instances")
    
    # Analyze target groups (if available)
    if 'target_groups' in test_df.columns:
        print("\nError rates by target group:")
        
        target_counts = {}
        target_errors = {}
        
        for _, row in test_df.iterrows():
            for target in row['target_groups']:
                target_counts[target] = target_counts.get(target, 0) + 1
                if not row['correct']:
                    target_errors[target] = target_errors.get(target, 0) + 1
        
        # Calculate error rates
        target_error_rates = {}
        for target in target_counts:
            if target_counts[target] >= 10:  # Only consider targets with sufficient samples
                error_rate = (target_errors.get(target, 0) / target_counts[target]) * 100
                target_error_rates[target] = error_rate
        
        # Sort by error rate
        for target, error_rate in sorted(target_error_rates.items(), key=lambda x: x[1], reverse=True):
            print(f"{target}: {error_rate:.2f}% error rate ({target_errors.get(target, 0)} of {target_counts[target]})")
    
    # Analyze text length
    test_df['text_length'] = test_df['text'].apply(len)
    
    print("\nError rate by text length:")
    bins = [0, 50, 100, 150, 200, 300, float('inf')]
    labels = ['0-50', '51-100', '101-150', '151-200', '201-300', '300+']
    
    test_df['length_bin'] = pd.cut(test_df['text_length'], bins=bins, labels=labels)
    
    length_stats = test_df.groupby('length_bin').agg({
        'text': 'count',
        'correct': lambda x: (~x).sum()  # Count of errors
    }).rename(columns={'text': 'total', 'correct': 'errors'})
    
    length_stats['error_rate'] = length_stats['errors'] / length_stats['total'] * 100
    
    print(length_stats[['total', 'errors', 'error_rate']])
    
    return incorrect_df