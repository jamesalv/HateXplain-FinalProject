import os
import random
import numpy as np
import torch
import json
import pandas as pd
from typing import Dict, Any

def set_seed(seed_value: int = 42) -> None:
    """
    Set seed for reproducibility
    
    Args:
        seed_value: Seed value for random number generators
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        
def save_results(
    comparison_results: Dict[str, Dict[str, Any]], 
    summary_table: pd.DataFrame, 
    efficiency_results: pd.DataFrame = None,
    results_dir: str = "results"
) -> None:
    """
    Save all results for future reference
    
    Args:
        comparison_results: Dictionary with model comparison results
        summary_table: Summary DataFrame
        efficiency_results: Efficiency analysis results
        results_dir: Directory to save results
    """
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Save summary table
    summary_table.to_csv(os.path.join(results_dir, 'model_comparison_summary.csv'), index=False)
    
    # Save efficiency results if available
    if efficiency_results is not None:
        efficiency_results.to_csv(os.path.join(results_dir, 'model_efficiency_results.csv'), index=False)
    
    # Save full results (excluding large objects like models)
    serializable_results = {}
    
    for task_type in comparison_results:
        serializable_results[task_type] = {}
        
        for model_name, result in comparison_results[task_type].items():
            # Create a copy without non-serializable objects
            serializable_result = {
                'model_name': result['model_name'],
                'num_classes': result['num_classes'],
                'history': result['history'],
                'metrics': result['metrics'],
                'confusion_matrix': result['confusion_matrix'].tolist() if isinstance(result['confusion_matrix'], np.ndarray) else result['confusion_matrix'],
                'label_map': result['label_map'],
                'target_metrics': result['target_metrics'],
                'bias_auc_metrics': result['bias_auc_metrics'],
            }
            
            serializable_results[task_type][model_name] = serializable_result
    
    # Save as JSON
    with open(os.path.join(results_dir, 'detailed_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"All results saved to the '{results_dir}' directory")

def add_new_model(
    model_name: str, 
    data_df: pd.DataFrame, 
    num_classes: int, 
    batch_size: int = 16, 
    epochs: int = 4
) -> Dict[str, Any]:
    """
    Add a new model to the comparison
    
    Args:
        model_name: Name of the model to add
        data_df: DataFrame with preprocessed data
        num_classes: Number of classes (2 or 3)
        batch_size: Batch size for training
        epochs: Number of training epochs
        
    Returns:
        Dictionary with results for the new model
    """
    from training.train import train_and_evaluate_model
    
    print(f"\nAdding new model: {model_name} for {num_classes}-class classification")
    
    # Train and evaluate model
    result = train_and_evaluate_model(
        model_name, 
        data_df, 
        num_classes=num_classes,
        batch_size=batch_size,
        epochs=epochs
    )
    
    return result