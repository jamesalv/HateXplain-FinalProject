import numpy as np
from typing import List, Tuple

import torch

from data.preprocessing.textPreprocessing import preprocess_text

# Rationale processing
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def calculate_rationale(rationale_list: List[int]) -> List[float]:
    att_arr = np.array(rationale_list)
    att_mean = att_arr.mean(axis=0)
    return softmax(att_mean)

def create_text_segment(text_tokens: List[str], rationale_mask: List[int]) -> Tuple[List[str], List[int]]:
    """
    Process a rationale mask to identify contiguous segments of highlighted text.
    Then create a segmented representation of the tokens
    
    Args:
        text_tokens: Original text tokens
        mask: Binary mask where 1 indicates a highlighted token (this consists of mask from 3 annotators)
        
    Returns:
        A tuple of (text segments, corresponding masks)
    """
    # Handle case where mask is empty (no rationale provided), usually this is normal classification
    all_segments = []
    all_rationale_mask = rationale_mask.copy()
    while len(all_rationale_mask) < 3:
        all_rationale_mask.append([0] * len(text_tokens))
    
    for mask in all_rationale_mask:
        # Find breakpoints (transitions between highlighted/1 and non-highlighted/0)
        breakpoints = []
        mask_values = []
        
        # Always start with position 0
        breakpoints.append(0)
        mask_values.append(mask[0])
        
        # Find transitions in the mask
        for i in range(1, len(mask)):
            if mask[i] != mask[i-1]:
                breakpoints.append(i)
                mask_values.append(mask[i])
        
        # Always end with the length of the text
        if breakpoints[-1] != len(mask):
            breakpoints.append(len(mask))
        
        # Create segments based on breakpoints
        segments = []
        for i in range(len(breakpoints) - 1):
            start = breakpoints[i]
            end = breakpoints[i+1]
            segments.append((text_tokens[start:end], mask_values[i]))
        all_segments.append(segments)
    
    return all_segments

def align_rationales(tokens, rationales, tokenizer, max_length=128):
    """
    Align rationales with tokenized text while handling different tokenizer formats.
    
    Args:
        tokens: Original text tokens
        rationales: Original rationale masks
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with tokenized inputs and aligned rationale masks
    """
    all_segments = create_text_segment(tokens, rationales)
    all_human_rationales = []
    
    # Process each segment
    for segments in all_segments:
        # Initialize lists to store processed data
        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        all_rationales = []
        for text_segment, rationale_value in segments:
            inputs = {}
            concatenated_text = " ".join(text_segment)
            # Textual preprocessing
            processed_segment = preprocess_text(concatenated_text)
            
            # Tokenize without special tokens
            tokenized = tokenizer(processed_segment, add_special_tokens=False, return_tensors='pt')
            
            # Extract the relevant data
            segment_input_ids = tokenized['input_ids'][0]
            segment_attention_mask = tokenized['attention_mask'][0]
            # Handle token_type_ids if present
            if 'token_type_ids' in tokenized:
                segment_token_type_ids = tokenized['token_type_ids'][0]
                all_token_type_ids.extend(segment_token_type_ids)
            
            # Add input IDs and attention mask
            all_input_ids.extend(segment_input_ids)
            all_attention_mask.extend(segment_attention_mask)
            
            # Add rationales (excluding special tokens)
            segment_rationales = [rationale_value] * len(segment_input_ids)
            all_rationales.extend(segment_rationales)
        
        # Get special token IDs
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        
        # Add special tokens at the beginning and end
        all_input_ids = [cls_token_id] + all_input_ids + [sep_token_id]
        all_attention_mask = [1] + all_attention_mask + [1]
        
        # Handle token_type_ids if the model requires it
        if hasattr(tokenizer, 'create_token_type_ids_from_sequences'):
            all_token_type_ids = tokenizer.create_token_type_ids_from_sequences(all_input_ids[1:-1]) 
        elif all_token_type_ids:
            all_token_type_ids = [0] + all_token_type_ids + [0]
        else:
            all_token_type_ids = [0] * len(all_input_ids)
        
        # Check tokenized vs rationales length
        if(len(all_input_ids) != len(all_attention_mask)):
            print("Warning: length of tokens and rationales do not match")
        
        # Add zero rationale values for special tokens
        all_rationales = [0] + all_rationales + [0]
        
        # Truncate to max length if needed
        if len(all_input_ids) > max_length:
            all_input_ids = all_input_ids[:max_length]
            all_attention_mask = all_attention_mask[:max_length]
            all_token_type_ids = all_token_type_ids[:max_length]
            all_rationales = all_rationales[:max_length]
        
        # Pad to max_length if needed
        pad_token_id = tokenizer.pad_token_id
        padding_length = max_length - len(all_input_ids)
        
        if padding_length > 0:
            all_input_ids = all_input_ids + [pad_token_id] * padding_length
            all_attention_mask = all_attention_mask + [0] * padding_length
            all_token_type_ids = all_token_type_ids + [0] * padding_length
            # Add zeros for padding tokens in rationales
            all_rationales = all_rationales + [0] * padding_length
        
        # Convert lists to tensors
        inputs = {
            'input_ids': torch.tensor([all_input_ids], dtype=torch.long),
            'attention_mask': torch.tensor([all_attention_mask], dtype=torch.long),
            'token_type_ids': torch.tensor([all_token_type_ids], dtype=torch.long) if 'token_type_ids' in tokenizer.model_input_names else None,
        }
        
        # Remove None values
        inputs = {k: v for k, v in inputs.items() if v is not None}
        
        # Add human rationales with zeros for special tokens
        human_rationales = torch.tensor([all_rationales], dtype=torch.float)
        all_human_rationales.append(human_rationales)
    
    # all_human_rationales contains rationale from multiple annotators
    # We need to aggregate them, but make sure to preserve zeros for special tokens
    
    # Stack all rationales
    stacked_rationales = torch.stack(all_human_rationales, dim=0)
    
    # Create a mask for non-special tokens
    # This will be 1 for content tokens and 0 for special tokens (CLS, SEP, PAD)
    content_token_mask = (inputs['attention_mask'] != 0).float()  # Non-padding tokens
    special_token_mask = torch.zeros_like(content_token_mask)
    special_token_mask[0, 0] = 1  # CLS token
    special_token_mask[0, -padding_length-1] = 1  # SEP token
    non_special_token_mask = content_token_mask * (1 - special_token_mask)
    
    # Average the rationales (only for non-special tokens)
    mean_rationales = torch.mean(stacked_rationales, dim=0)
    
    # Create mask for non_special_indices
    non_special_indices = non_special_token_mask[0].bool()
    non_special_values = mean_rationales[0, non_special_indices]
    
    # Create a copy to hold softmaxed values (length is the same as max_length)
    softmaxed_rationales = torch.zeros_like(mean_rationales)
    
    # Apply softmax to non-special-values (the length is the same as the tokenized input)
    softmaxed_values = torch.nn.functional.softmax(non_special_values, dim=0)
    
    # Put the softmaxed values back to the placeholder on the respective position
    softmaxed_rationales[0, non_special_indices] = softmaxed_values
    
    return inputs, softmaxed_rationales