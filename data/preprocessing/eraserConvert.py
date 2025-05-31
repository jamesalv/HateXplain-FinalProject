import json
import os
import more_itertools as mit
from typing import List, Dict, Any, Tuple
import numpy as np
import torch

def find_ranges(iterable):
    """Yield range of consecutive numbers - from original HateXplain code"""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]

def get_evidence_from_rationales(post_id: str, rationale_tensor: torch.Tensor, 
                                tokenizer: Any, k: int = 5) -> List[Dict[str, Any]]:
    """
    Convert rationale tensor to ERASER evidence format using top-k approach
    
    Args:
        post_id: Post identifier
        rationale_tensor: Tensor of rationale scores [1, max_length]
        tokenizer: Tokenizer used for preprocessing
        k: Number of top-scoring tokens to use as evidence (default: 5)
        
    Returns:
        List of evidence dictionaries in ERASER format
    """
    # Convert tensor to numpy and remove batch dimension
    if rationale_tensor.dim() > 1:
        rationale_scores = rationale_tensor.squeeze(0).numpy()
    else:
        rationale_scores = rationale_tensor.numpy()
    
    # Get valid token positions with their scores (exclude special tokens with score = 0)
    valid_positions = []
    for i, score in enumerate(rationale_scores):
        if score > 0.0:  # Non-special tokens (CLS, SEP, PAD have score = 0)
            valid_positions.append((i, score))
    
    if not valid_positions:
        return []
    
    # Sort by score (highest first) and take top-k
    valid_positions.sort(key=lambda x: x[1], reverse=True)
    top_k_positions = valid_positions[:k]
    
    # Extract just the indices and sort them for consecutive grouping
    evidence_indices = sorted([pos for pos, _ in top_k_positions])
    
    # Group consecutive indices into spans
    evidence_spans = []
    span_list = list(find_ranges(evidence_indices))
    
    for span in span_list:
        if isinstance(span, int):
            start_idx = span
            end_idx = span + 1
        else:
            start_idx = span[0]
            end_idx = span[1] + 1
        
        # Create evidence entry
        evidence_entry = {
            "docid": post_id,
            "end_sentence": -1,
            "end_token": end_idx,
            "start_sentence": -1, 
            "start_token": start_idx,
            "text": f"tokens_{start_idx}_{end_idx-1}"  # Placeholder for actual token text
        }
        evidence_spans.append(evidence_entry)
    
    return evidence_spans

def get_actual_token_text(input_ids: torch.Tensor, tokenizer: Any, 
                         start_idx: int, end_idx: int) -> str:
    """
    Get actual token text from input_ids
    
    Args:
        input_ids: Tensor of input token IDs
        tokenizer: Tokenizer to decode tokens
        start_idx: Start token index
        end_idx: End token index (exclusive)
        
    Returns:
        Decoded text string
    """
    if input_ids.dim() > 1:
        token_ids = input_ids.squeeze(0)
    else:
        token_ids = input_ids
    
    # Extract the relevant token IDs
    relevant_tokens = token_ids[start_idx:end_idx]
    
    # Decode to text
    decoded_text = tokenizer.decode(relevant_tokens, skip_special_tokens=True)
    return decoded_text.strip()

def convert_entry_to_eraser_ground_truth(entry: Dict[str, Any], tokenizer: Any, k: int = 5) -> Dict[str, Any]:
    """
    Convert a single preprocessed entry to ERASER ground truth format
    
    Args:
        entry: Preprocessed data entry with rationales
        tokenizer: Tokenizer used for preprocessing
        k: Number of top-scoring tokens to use as evidence (default: 5)
        
    Returns:
        ERASER ground truth entry
    """
    post_id = entry['post_id']
    final_label = entry['final_label']
    rationale_tensor = entry['rationales']
    input_ids = entry['inputs']['input_ids']
    
    # Skip normal posts (following HateXplain paper convention)
    if final_label == 'normal':
        return None
    
    # Get evidence from rationales using top-k approach
    evidence_list = get_evidence_from_rationales(post_id, rationale_tensor, tokenizer, k=k)
    
    # Enhance evidence with actual token text
    for evidence in evidence_list:
        start_idx = evidence['start_token']
        end_idx = evidence['end_token']
        actual_text = get_actual_token_text(input_ids, tokenizer, start_idx, end_idx) # Get actual text for the token span, for example "nigg", "#er" (in tokeinzed number form) will be converted to "nigger"
        evidence['text'] = actual_text if actual_text else   f"tokens_{start_idx}_{end_idx-1}"
    
    # Create ERASER ground truth entry
    eraser_entry = {
        "annotation_id": post_id,
        "classification": final_label,
        "evidences": [evidence_list] if evidence_list else [[]],
        "query": "What is the class?",
        "query_type": None
    }
    
    return eraser_entry

def create_eraser_ground_truth_from_df(df, tokenizer: Any, output_dir: str, 
                                     model_name: str, task_type: str, k: int = 5,
                                     split_file_path: str = 'Raw Data/post_id_divisions.json') -> Dict[str, str]:
    """
    Create ERASER ground truth files from preprocessed dataframe with train/val/test splits
    
    Args:
        df: Preprocessed dataframe
        tokenizer: Tokenizer used for preprocessing
        output_dir: Directory to save ERASER files
        model_name: Model name for file naming
        task_type: Either "binary" or "3class"
        k: Number of top-scoring tokens to use as evidence (default: 5)
        split_file_path: Path to JSON file containing post_id divisions
        
    Returns:
        Dictionary with paths to created ground truth files
    """
    # Load split information
    try:
        with open(split_file_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        print(f"Loaded split information from: {split_file_path}")
    except FileNotFoundError:
        print(f"Warning: Split file not found at {split_file_path}")
        print("Creating test.jsonl only...")
        split_data = None
    
    # Create output directory
    safe_model_name = model_name.replace('/', '_')
    eraser_dir = os.path.join(output_dir, 'eraser_ground_truth', safe_model_name, task_type)
    os.makedirs(eraser_dir, exist_ok=True)
    
    # Initialize split containers
    split_entries = {
        'train': [],
        'val': [],
        'test': []
    }
    
    # Counters
    skipped_normal = 0
    processed_counts = {'train': 0, 'val': 0, 'test': 0}
    unmatched_posts = 0
    
    print(f"Converting {len(df)} entries to ERASER ground truth format (top-{k} approach)...")
    
    for idx, row in df.iterrows():
        eraser_entry = convert_entry_to_eraser_ground_truth(row.to_dict(), tokenizer, k=k)
        
        if eraser_entry is None:
            skipped_normal += 1
            continue
        
        post_id = row['post_id']
        
        # Determine which split this post belongs to
        if split_data is not None:
            split_found = False
            for split_name, post_ids in split_data.items():
                if post_id in post_ids:
                    split_entries[split_name].append(eraser_entry)
                    processed_counts[split_name] += 1
                    split_found = True
                    break
            
            if not split_found:
                unmatched_posts += 1
                # Default to test if post_id not found in any split
                split_entries['test'].append(eraser_entry)
                processed_counts['test'] += 1
        else:
            # If no split file, put everything in test
            split_entries['test'].append(eraser_entry)
            processed_counts['test'] += 1
    
    # Save split files
    created_files = {}
    
    for split_name, entries in split_entries.items():
        if entries:  # Only create file if there are entries
            split_file = os.path.join(eraser_dir, f'{split_name}.jsonl')
            
            with open(split_file, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')
            
            created_files[split_name] = split_file
            print(f"ERASER {split_name} ground truth saved to: {split_file}")
            print(f"  {split_name.capitalize()} entries: {len(entries)}")
    
    # Print summary
    print(f"\n=== ERASER Ground Truth Summary ===")
    print(f"Total processed entries: {sum(processed_counts.values())}")
    print(f"Skipped normal entries: {skipped_normal}")
    if split_data is not None and unmatched_posts > 0:
        print(f"Unmatched post_ids (added to test): {unmatched_posts}")
    print(f"Using top-{k} most important tokens per entry")
    print(f"Split distribution:")
    for split_name, count in processed_counts.items():
        if count > 0:
            print(f"  {split_name}: {count} entries")
    
    return created_files

def create_eraser_docs_directory(df, output_dir: str, model_name: str, task_type: str):
    """
    Create docs directory with individual text files (required by ERASER)
    
    Args:
        df: Preprocessed dataframe
        output_dir: Directory to save ERASER files
        model_name: Model name for directory naming
        task_type: Either "binary" or "3class"
    """
    safe_model_name = model_name.replace('/', '_')
    docs_dir = os.path.join(output_dir, 'eraser_ground_truth', safe_model_name, task_type, 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    
    print(f"Creating docs directory: {docs_dir}")
    
    for idx, row in df.iterrows():
        # Skip normal posts
        if row['final_label'] == 'normal':
            continue
            
        post_id = row['post_id']
        raw_text = row['raw_text']
        
        # Save individual text file
        doc_file = os.path.join(docs_dir, post_id)
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(raw_text)
    
    print(f"Created {len([f for f in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir, f))])} doc files")
    print(f"Docs directory ready at: {docs_dir}")