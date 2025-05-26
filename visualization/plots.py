import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def plot_confusion_matrix(
    cm: np.ndarray, 
    classes: List[str], 
    model_name: str, 
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        classes: Class labels
        model_name: Name of the model
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.show()

def plot_class_distribution(df: pd.DataFrame, label_col: str = 'final_label') -> None:
    """
    Plot class distribution
    
    Args:
        df: DataFrame with data
        label_col: Column name containing labels
    """
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x=label_col)
    
    # Add count labels
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                height, ha="center")
    
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_text_length_distribution(df: pd.DataFrame, text_col: str = 'text', bins: int = 30) -> None:
    """
    Plot text length distribution
    
    Args:
        df: DataFrame with data
        text_col: Column name containing text
        bins: Number of bins for histogram
    """
    # Calculate text lengths
    text_lengths = df[text_col].apply(len)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(text_lengths, bins=bins)
    plt.title('Text Length Distribution')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Count')
    plt.axvline(text_lengths.mean(), color='r', linestyle='--', label=f'Mean: {text_lengths.mean():.1f}')
    plt.axvline(text_lengths.median(), color='g', linestyle='--', label=f'Median: {text_lengths.median():.1f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_target_group_distribution_by_class(df: pd.DataFrame, target_col: str = 'target_groups', label_col: str = 'final_label') -> None:
    """
    Plot target group distribution per class using subplots
    
    Args:
        df: DataFrame with data
        target_col: Column name containing target groups
        label_col: Column name containing class labels
    """
    # Get unique classes
    classes = df[label_col].unique().tolist()
    
    # Create figure with subplots (one per class)
    fig, axes = plt.subplots(len(classes), 1, figsize=(14, 6 * len(classes)))
    
    # If only one class, make axes iterable
    if len(classes) == 1:
        axes = [axes]
    
    # Process each class
    for i, class_label in enumerate(classes):
        # Filter dataframe for this class
        class_df = df[df[label_col] == class_label]
        
        # Extract all target groups for this class
        class_targets = []
        for targets in class_df[target_col]:
            class_targets.extend(targets)
        
        # Count occurrences
        from collections import Counter
        target_counts = Counter(class_targets)
        
        # Convert to DataFrame for plotting
        target_df = pd.DataFrame({
            'Target': list(target_counts.keys()),
            'Count': list(target_counts.values())
        }).sort_values('Count', ascending=False)
        
        # Plot on the corresponding subplot
        ax = axes[i]
        bars = sns.barplot(data=target_df, x='Target', y='Count', ax=ax)
        
        # Add count labels
        for p in bars.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                    int(height), ha="center")
        
        ax.set_title(f'Target Group Distribution for Class: {class_label}')
        ax.set_xlabel('Target Group')
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()