from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
from collections import Counter
import json
from tqdm import tqdm
import os
import pickle
import torch
from transformers import AutoTokenizer
from data.preprocessing.rationalePreprocessing import (
    process_rationale,
    map_rationales_to_bert_tokens,
)
from data.preprocessing.textPreprocessing import preprocess_text


def load_raw_data(file_path: str) -> Dict[str, Any]:
    """
    Load the raw JSON data from file.

    Args:
        file_path: Path to the JSON dataset file

    Returns:
        Loaded JSON data
    """
    with open(file_path, "r") as file:
        return json.load(file)


def process_raw_entries(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process raw data entries without classification-specific steps.

    Args:
        data: Raw dataset

    Returns:
        Tuple of (processed_entries, count_confused)
    """
    processed_entries = []
    count_confused = 0

    # Process each entry with progress bar
    for key, value in tqdm(data.items(), desc="Processing entries", unit="entry"):
        processed_entry = {}
        processed_entry["post_id"] = key

        # Combine post_tokens to a single string and preprocess
        processed_entry["raw_tokens"] = value["post_tokens"]  # Store original text
        raw_text = " ".join(value["post_tokens"])
        processed_entry["processed_text"] = preprocess_text(
            raw_text
        )  # Store preprocessed text

        # Extract labels and target groups
        # Only select target groups which at least selected by twon annotator
        labels = [annot["label"] for annot in value["annotators"]]
        target_groups = []
        for annot in value["annotators"]:
            target_groups.extend(annot["target"])
        counter_groups = Counter(target_groups)
        target_groups = [group for group, count in counter_groups.items() if count > 1]

        # Skip entries where all annotators disagree
        if len(set(labels)) == 3:
            count_confused += 1
            continue

        # Store labels for later classification
        processed_entry["labels"] = labels

        # Process target groups (remove duplicates)
        processed_entry["target_groups"] = list(set(target_groups))

        # Store rationales for later processing
        if(len(value['rationales']) == 0):
            # Creat the rationale with zeros
            value['rationales'] = [[0]*len(value['post_tokens']) for _ in range(3)]
        processed_entry["rationales"] = value["rationales"]

        processed_entries.append(processed_entry)

    return processed_entries, count_confused


def process_rationales(data: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Process rationales for a single entry.

    Args:
        data: Single data entry with raw tokens and rationales
        tokenizer: HuggingFace tokenizer

    Returns:
        Updated data entry with human rationales and tokenized text
    """
    # Tokenize
    tokenized = tokenizer(
        data["processed_text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    bert_tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
    original_tokens = data["raw_tokens"]
    original_rationales = process_rationale(data["rationales"])

    # Align the rationales
    data["human_rationales"] = map_rationales_to_bert_tokens(
        original_tokens, bert_tokens, original_rationales
    )
    data["tokenized_text"] = tokenized

    return data


def process_rationales_batch(
    entries: List[Dict[str, Any]], tokenizer, batch_size: int = 32
) -> List[Dict[str, Any]]:
    """
    Process rationales in batches for efficiency.

    Args:
        entries: List of data entries
        tokenizer: HuggingFace tokenizer
        batch_size: Number of entries to process in each batch

    Returns:
        List of processed entries
    """
    all_processed = []

    # Process in batches
    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + min(batch_size, len(entries) - i)]

        # Process each entry in the batch
        for entry in batch:
            processed_entry = process_rationales(entry, tokenizer)
            all_processed.append(processed_entry)

    return all_processed


def create_dataset_with_labels(
    processed_entries: List[Dict[str, Any]], num_classes: int = 3
) -> pd.DataFrame:
    """
    Create final dataset with appropriate labels based on the number of classes.

    Args:
        processed_entries: List of processed data entries
        num_classes: Number of classes for classification (2 or 3)

    Returns:
        DataFrame with appropriate labels
    """
    dataset_entries = []

    for entry in processed_entries:
        # Create a new entry with all the existing fields
        new_entry = entry.copy()

        # Determine final label
        new_entry["final_label"] = Counter(entry["labels"]).most_common()[0][0]

        # Convert to binary classification if needed
        if num_classes == 2:
            if new_entry["final_label"] in ("hatespeech", "offensive"):
                new_entry["final_label"] = "toxic"
            else:
                new_entry["final_label"] = "non-toxic"

        # Remove the temporary labels field
        del new_entry["labels"]

        dataset_entries.append(new_entry)

    return pd.DataFrame(dataset_entries)


def validate_processed_entries(entries: List[Dict[str, Any]]) -> bool:
    """
    Validate that processed entries have all required fields.

    Args:
        entries: List of processed entries

    Returns:
        True if validation passes, raises Exception otherwise
    """
    required_fields = [
        "post_id",
        "raw_tokens",
        "processed_text",
        "human_rationales",
        "tokenized_text",
        "final_label",
        "target_groups",
    ]

    # Check for missing fields
    issues_found = False
    for i, entry in enumerate(entries):
        missing = [field for field in required_fields if field not in entry]
        if missing:
            print(
                f"Warning: Entry {i} (post_id: {entry.get('post_id', 'unknown')}) "
                f"is missing required fields: {missing}"
            )
            issues_found = True

    # Check that human_rationales match tokenized_text length
    for i, entry in enumerate(entries):
        if "human_rationales" in entry and "tokenized_text" in entry:
            human_rationales = entry["human_rationales"]
            input_ids = entry["tokenized_text"]["input_ids"][0]

            if len(human_rationales) != len(input_ids):
                print(
                    f"Warning: Entry {i} (post_id: {entry['post_id']}) has mismatched "
                    f"lengths: human_rationales ({len(human_rationales)}) vs "
                    f"input_ids ({len(input_ids)})"
                )
                issues_found = True

    if issues_found:
        print("Validation found issues with the processed data.")
    else:
        print(
            "Validation passed: All entries have required fields and correct dimensions."
        )

    return not issues_found


def get_cache_filename(
    data_path: str, model_name: str, cache_dir: str = "cache"
) -> str:
    """
    Generate a cache filename based on the data path and model name.

    Args:
        data_path: Path to the raw dataset
        model_name: Name of the model (for tokenizer)
        cache_dir: Directory for cache files

    Returns:
        Cache filename
    """
    # Create a cache filename based on data path and model name
    base_name = os.path.basename(data_path).split(".")[0]
    model_id = model_name.replace("/", "_")
    cache_file = os.path.join(cache_dir, f"{base_name}_{model_id}_processed.pkl")
    return cache_file


def preprocess_datasets(
    data_path: str,
    model_name: str = "bert-base-uncased",
    use_cache: bool = True,
    cache_dir: str = "cache",
    validate: bool = True,
    batch_size: int = 32,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess data, creating both 2-class and 3-class versions efficiently.

    Args:
        data_path: Path to raw dataset file
        model_name: Name of the model to use for tokenization
        use_cache: Whether to use cached preprocessed data
        cache_dir: Directory for cache files
        validate: Whether to validate the processed data
        batch_size: Batch size for processing rationales

    Returns:
        Tuple of (data_3class, data_2class)
    """
    # Initialize tokenizer
    print(f"Initializing tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Check for cached processed data
    cache_file = get_cache_filename(data_path, model_name, cache_dir)

    if use_cache and os.path.exists(cache_file):
        print(f"Loading preprocessed data from cache: {cache_file}")
        try:
            with open(cache_file, "rb") as f:
                data_3class, data_2class = pickle.load(f)

            if validate:
                print("Validating cached data...")
                validate_processed_entries(data_3class.to_dict("records"))

            return data_3class, data_2class
        except Exception as e:
            print(f"Error loading from cache: {e}")
            print("Proceeding with full preprocessing...")

    print("Loading raw data...")
    raw_data = load_raw_data(data_path)

    print("Processing raw data...")
    processed_entries, count_confused = process_raw_entries(raw_data)

    print("Processing rationales...")
    processed_entries = process_rationales_batch(
        processed_entries, tokenizer, batch_size=batch_size
    )

    # Print statistics
    print(f"Initial data: {len(raw_data)}")
    print(f"Uncertain data: {count_confused}")
    print(f"Total processed entries: {len(processed_entries)}")

    print("Creating 3-class dataset...")
    data_3class = create_dataset_with_labels(processed_entries, num_classes=3)

    print("Creating 2-class dataset...")
    data_2class = create_dataset_with_labels(processed_entries, num_classes=2)

    # Validate the processed data
    if validate:
        print("Validating processed data...")
        validate_processed_entries(data_3class.to_dict("records"))

    # Cache the results
    if use_cache:
        print(f"Caching preprocessed data to: {cache_file}")
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump((data_3class, data_2class), f)

    return data_3class, data_2class
