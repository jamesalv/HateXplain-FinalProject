from typing import Any, Dict, List
import pandas as pd
import torch


def add_faithfulness_scores(
    eraser_predictions: List[Dict[str, Any]], 
    model, 
    tokenizer,
    test_df: pd.DataFrame,
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Add sufficiency and comprehensiveness scores to ERASER predictions
    
    Args:
        eraser_predictions: List of ERASER prediction entries
        model: Trained transformer model
        tokenizer: Model tokenizer
        test_df: Test dataframe with original texts
        device: Device to run model on
        
    Returns:
        Enhanced ERASER predictions with faithfulness scores
    """
    print(f"Adding faithfulness scores to {len(eraser_predictions)} predictions...")
    
    enhanced_predictions = []
    
    for pred_entry in eraser_predictions:
        post_id = pred_entry['annotation_id']
        
        # Find corresponding text in test dataframe
        row = test_df[test_df['post_id'] == post_id]
        if row.empty:
            print(f"Warning: Could not find post_id {post_id} in test data")
            enhanced_predictions.append(pred_entry)
            continue
            
        original_text = row.iloc[0]['raw_text']
        rationale_indices = extract_rationale_indices(pred_entry)
        
        # Get original prediction scores
        original_scores = pred_entry['classification_scores']
        
        # Calculate sufficiency scores (using ONLY rationale tokens)
        sufficiency_scores = calculate_sufficiency_scores(
            original_text, rationale_indices, model, tokenizer, device
        )
        
        # Calculate comprehensiveness scores (REMOVING rationale tokens)
        comprehensiveness_scores = calculate_comprehensiveness_scores(
            original_text, rationale_indices, model, tokenizer, device
        )
        
        # Add faithfulness scores to prediction entry
        pred_entry['sufficiency_classification_scores'] = sufficiency_scores
        pred_entry['comprehensiveness_classification_scores'] = comprehensiveness_scores
        
        enhanced_predictions.append(pred_entry)
    
    print("Faithfulness scores added successfully!")
    return enhanced_predictions


def extract_rationale_indices(eraser_entry: Dict[str, Any]) -> List[int]:
    """
    Extract rationale token indices from ERASER prediction entry
    
    Args:
        eraser_entry: ERASER prediction entry
        
    Returns:
        List of rationale token indices
    """
    rationale_indices = []
    
    if 'rationales' in eraser_entry and eraser_entry['rationales']:
        hard_rationales = eraser_entry['rationales'][0].get('hard_rationale_predictions', [])
        
        for rationale in hard_rationales:
            start_idx = rationale['start_token']
            end_idx = rationale['end_token']
            
            # Add all indices in the span
            rationale_indices.extend(range(start_idx, end_idx))
    
    return sorted(list(set(rationale_indices)))  # Remove duplicates and sort


def calculate_sufficiency_scores(
    original_text: str,
    rationale_indices: List[int],
    model,
    tokenizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Calculate sufficiency scores: How well can model predict using ONLY rationale tokens?
    
    Args:
        original_text: Original text string
        rationale_indices: List of important token indices
        model: Trained model
        tokenizer: Model tokenizer
        device: Device to run on
        
    Returns:
        Dictionary with class probabilities using only rationales
    """
    if not rationale_indices:
        # If no rationales, return uniform distribution
        return {"hatespeech": 0.33, "normal": 0.33, "offensive": 0.34}
    
    # Tokenize original text
    tokens = tokenizer.tokenize(original_text)
    
    # Extract only rationale tokens
    rationale_tokens = []
    for idx in rationale_indices:
        if idx < len(tokens):
            rationale_tokens.append(tokens[idx])
    
    if not rationale_tokens:
        return {"hatespeech": 0.33, "normal": 0.33, "offensive": 0.34}
    
    # Create text with only rationale tokens
    rationale_text = tokenizer.convert_tokens_to_string(rationale_tokens)
    
    # Get prediction for rationale-only text
    return get_model_prediction_scores(rationale_text, model, tokenizer, device)


def calculate_comprehensiveness_scores(
    original_text: str,
    rationale_indices: List[int], 
    model,
    tokenizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Calculate comprehensiveness scores: How much does removing rationales hurt performance?
    
    Args:
        original_text: Original text string
        rationale_indices: List of important token indices
        model: Trained model
        tokenizer: Model tokenizer
        device: Device to run on
        
    Returns:
        Dictionary with class probabilities without rationales
    """
    # Tokenize original text
    tokens = tokenizer.tokenize(original_text)
    
    # Remove rationale tokens
    remaining_tokens = []
    for i, token in enumerate(tokens):
        if i not in rationale_indices:
            remaining_tokens.append(token)
    
    if not remaining_tokens:
        # If no tokens left, use a neutral text
        remaining_text = "[MASK]"
    else:
        # Create text without rationale tokens
        remaining_text = tokenizer.convert_tokens_to_string(remaining_tokens)
    
    # Get prediction for text without rationales
    return get_model_prediction_scores(remaining_text, model, tokenizer, device)


def get_model_prediction_scores(
    text: str,
    model,
    tokenizer, 
    device: torch.device,
    max_length: int = 128
) -> Dict[str, float]:
    """
    Get model prediction scores for a given text
    
    Args:
        text: Input text
        model: Trained model
        tokenizer: Model tokenizer
        device: Device to run on
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with class probabilities
    """
    model.eval()
    
    # Tokenize input text
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    # Convert to class score dictionary
    class_scores = {
        "hatespeech": float(probabilities[0]),
        "normal": float(probabilities[1]), 
        "offensive": float(probabilities[2])
    }
    
    return class_scores