from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
from collections import Counter
import json
from tqdm import tqdm
import os
import pickle
import torch
from transformers import AutoTokenizer
from data.preprocessing.eraserConvert import save_eraser_ground_truth
from data.preprocessing.rationalePreprocessing import (
    align_rationales,
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


def process_raw_entries(
    data: Dict[str, Any], tokenizer: Any
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process raw data entries without classification-specific steps.

    Args:
        data: Raw dataset

    Returns:
        Tuple of (processed_entries, count_confused)
    """
    processed_entries = []
    count_confused = 0

    # Process each entry
    for key, value in tqdm(data.items(), desc="Processing entries", unit="entry"):
        processed_entry = {}
        processed_entry["post_id"] = key

        # Combine post_tokens to a single string and preprocess
        raw_text = " ".join(value["post_tokens"])
        processed_entry["raw_text"] = raw_text  # Store original text

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

        # Process Rationales
        processed_entry["inputs"], processed_entry["rationales"] = align_rationales(
            value["post_tokens"], value["rationales"], tokenizer
        )

        processed_entries.append(processed_entry)

    return processed_entries, count_confused


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


def save_preprocessed_data(
    df_3class, df_2class, model_name, save_dir="preprocessed_data"
):
    """Save preprocessed data for later use"""
    # Create model-specific directory
    model_save_dir = os.path.join(
        save_dir, model_name.replace("/", "_")
    )  # Replace '/' to avoid path issues
    os.makedirs(model_save_dir, exist_ok=True)

    # Save as pickle to preserve tensor objects
    with open(os.path.join(model_save_dir, "data_3class.pkl"), "wb") as f:
        pickle.dump(df_3class, f)

    with open(os.path.join(model_save_dir, "data_2class.pkl"), "wb") as f:
        pickle.dump(df_2class, f)

    print(f"Data saved to {model_save_dir}/")


def preprocess_datasets(
    data_path: str,
    model_name: str,
    save_dir="preprocessed_data",
    create_eraser: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess data, creating both 2-class and 3-class versions.

    Args:
        data_path: Path to raw dataset file
        model_name: HuggingFace model name
        save_dir: Directory to save/load preprocessed data
        create_eraser: Whether to create ERASER ground truth format

    Returns:
        Tuple of (data_3class, data_2class)
    """
    # Create model-specific directory path
    model_save_dir = os.path.join(save_dir, model_name.replace("/", "_"))
    data_3class_path = os.path.join(model_save_dir, "data_3class.pkl")
    data_2class_path = os.path.join(model_save_dir, "data_2class.pkl")

    # Check if both files exist
    if os.path.exists(data_3class_path) and os.path.exists(data_2class_path):
        print(f"Preprocessed data for {model_name} is already available!")
        print("Using available data...")
        with open(data_3class_path, "rb") as file:
            data_3class = pickle.load(file)

        with open(data_2class_path, "rb") as file:
            data_2class = pickle.load(file)

        print("Data loaded successfully!")
    else:
        print("No preprocessed data found. Processing from scratch...")
        print("Loading raw data...")
        raw_data = load_raw_data(data_path)

        print("Processing and preprocessing entries...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processed_entries, count_confused = process_raw_entries(raw_data, tokenizer)

        # Print statistics
        print(f"Initial data: {len(raw_data)}")
        print(f"Uncertain data: {count_confused}")
        print(f"Total processed entries: {len(processed_entries)}")

        print("Creating 3-class dataset...")
        data_3class = create_dataset_with_labels(processed_entries, num_classes=3)

        print("Creating 2-class dataset...")
        data_2class = create_dataset_with_labels(processed_entries, num_classes=2)

        # Save the processed data
        save_preprocessed_data(data_3class, data_2class, model_name, save_dir)

    # Create ERASER ground truth if requested
    if create_eraser:
        eraser_files = save_eraser_ground_truth(
            data_3class, data_2class, model_name, save_dir
        )
        print(f"ERASER ground truth files created:")
        print(f"  3-class: {eraser_files['3class']}")
        print(f"  Binary: {eraser_files['binary']}")

    return data_3class, data_2class
