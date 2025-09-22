import json
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
import more_itertools as mit
import pandas as pd

from models.explainability import add_faithfulness_scores

def find_ranges(iterable):
    """Yield range of consecutive numbers - from original HateXplain code"""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


def convert_predictions_to_eraser_format(
    predictions: np.ndarray,
    probabilities: np.ndarray, 
    attention_weights: np.ndarray,
    true_labels: np.ndarray,
    test_df: pd.DataFrame,
    label_map: Dict[str, int],
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Convert model predictions to ERASER format
    
    Args:
        predictions: Array of predicted class indices [batch_size]
        probabilities: Array of class probabilities [batch_size, num_classes]
        attention_weights: Array of attention weights [batch_size, seq_len]
        true_labels: Array of true class indices [batch_size]
        test_df: Test dataframe with post_ids and metadata
        label_map: Mapping from label names to indices
        k: Number of top attention tokens to use as rationales
        
    Returns:
        List of ERASER prediction entries
    """
    eraser_predictions = []
    
    # Create inverse label mapping
    inv_label_map = {v: k for k, v in label_map.items()}
    
    print(f"Converting {len(predictions)} predictions to ERASER format (top-{k} rationales)...")
    
    for i, (pred, prob, attention, true_label) in enumerate(zip(
        predictions, probabilities, attention_weights, true_labels
    )):
        # Get corresponding row from test dataframe
        row = test_df.iloc[i]
        post_id = row['post_id']
        true_label_name = row['final_label']
        
        # Skip normal posts (following HateXplain paper convention)
        if true_label_name == 'normal':
            continue
            
        # Get predicted label name
        predicted_label = inv_label_map[pred]
        
        # Create classification scores dictionary
        classification_scores = {}
        for label_name, label_idx in label_map.items():
            classification_scores[label_name] = float(prob[label_idx])
        
        # Extract top-k rationales from attention weights
        hard_rationales = extract_top_k_rationales(attention, k)
        
        # Create ERASER prediction entry
        eraser_entry = {
            "annotation_id": post_id,
            "classification": predicted_label,
            "classification_scores": classification_scores,
            "rationales": [{
                "docid": post_id,
                "hard_rationale_predictions": hard_rationales,
                # Soft rationales are the attention weights themselves without paddings (0)
                "soft_rationale_predictions": [attention[attention > 0.0].tolist()][0],
                "truth": label_map[true_label_name]
            }]
        }
        
        eraser_predictions.append(eraser_entry)
    print(f"Created {len(eraser_predictions)} ERASER prediction entries")
    return eraser_predictions

#  Update the main conversion function to include faithfulness
def convert_predictions_to_eraser_format_with_faithfulness(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    attention_weights: np.ndarray, 
    true_labels: np.ndarray,
    test_df: pd.DataFrame,
    label_map: Dict[str, int],
    model,
    tokenizer,
    device: torch.device,
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Convert model predictions to ERASER format WITH faithfulness scores
    
    This is the complete function that does everything in one go.
    """
    # Step 1: Convert to basic ERASER format
    eraser_predictions = convert_predictions_to_eraser_format(
        predictions, probabilities, attention_weights, true_labels,
        test_df, label_map, k
    )
    
    # Step 2: Add faithfulness scores
    enhanced_predictions = add_faithfulness_scores(
        eraser_predictions, model, tokenizer, test_df, device
    )
    
    return enhanced_predictions


def extract_top_k_rationales(attention_weights: np.ndarray, k: int = 5) -> List[Dict[str, int]]:
    """
    Extract top-k most important tokens from attention weights
    
    Args:
        attention_weights: Array of attention weights [seq_len]
        k: Number of top tokens to extract
        
    Returns:
        List of hard rationale predictions in ERASER format
    """
    # Find valid token positions (exclude padding/special tokens with 0 attention)
    valid_positions = []
    for i, weight in enumerate(attention_weights):
        if weight > 0.0:  # Non-special tokens
            valid_positions.append((i, weight))
    
    if not valid_positions:
        return []
    
    # Sort by attention weight (highest first) and take top-k
    valid_positions.sort(key=lambda x: x[1], reverse=True)
    top_k_positions = valid_positions[:k]
    
    # Extract indices and sort them for consecutive grouping
    rationale_indices = sorted([pos for pos, _ in top_k_positions])
    
    # Group consecutive indices into spans
    hard_rationales = []
    span_list = list(find_ranges(rationale_indices))
    
    for span in span_list:
        if isinstance(span, int):
            start_idx = span
            end_idx = span + 1
        else:
            start_idx = span[0]
            end_idx = span[1] + 1
        
        # Create rationale entry
        rationale_entry = {
            "end_token": end_idx,
            "start_token": start_idx
        }
        hard_rationales.append(rationale_entry)
    
    return hard_rationales


def save_eraser_predictions(
    eraser_predictions: List[Dict[str, Any]], 
    output_path: str
) -> None:
    """
    Save ERASER predictions to JSONL file
    
    Args:
        eraser_predictions: List of ERASER prediction entries
        output_path: Path to save the predictions
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSONL format
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in eraser_predictions:
            f.write(json.dumps(entry) + '\n')
    
    print(f"ERASER predictions saved to: {output_path}")
    print(f"Total entries: {len(eraser_predictions)}")