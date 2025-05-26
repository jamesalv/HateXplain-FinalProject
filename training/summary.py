import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, roc_auc_score, confusion_matrix
)
from typing import Dict, List, Tuple, Any, Union

def create_summary_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a summary table of all model results
    
    Args:
        results: Dictionary with model results
        
    Returns:
        DataFrame with summary statistics
    """
    # Initialize summary data
    summary_data = []
    
    for task_type in ['binary', '3class']:
        for model_name, result in results[task_type].items():
            # Extract key metrics
            metrics = result['metrics']
            
            # Add row to summary data
            summary_data.append({
                'Model': model_name.split('/')[-1],
                'Task': task_type,
                'Accuracy': metrics['accuracy'],
                'F1 Score': metrics['f1_score'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'AUROC': metrics['auroc'],
            })
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by task type and then by F1 score
    summary_df = summary_df.sort_values(['Task', 'F1 Score'], ascending=[True, False])
    
    return summary_df

def generate_conclusions(
    summary_table: pd.DataFrame, 
    efficiency_results: pd.DataFrame = None
) -> None:
    """
    Generate conclusions and recommendations based on results
    
    Args:
        summary_table: Summary table of model results
        efficiency_results: Efficiency analysis results (optional)
    """
    print("\n=== FINAL CONCLUSIONS ===\n")
    
    # Best model for each task by F1 score
    binary_best = summary_table[summary_table['Task'] == 'binary'].sort_values('F1 Score', ascending=False).iloc[0]
    multiclass_best = summary_table[summary_table['Task'] == '3class'].sort_values('F1 Score', ascending=False).iloc[0]
    
    print("Best model for binary classification:")
    print(f"{binary_best['Model']}: F1 Score = {binary_best['F1 Score']:.4f}, Accuracy = {binary_best['Accuracy']:.4f}")
    
    print("\nBest model for 3-class classification:")
    print(f"{multiclass_best['Model']}: F1 Score = {multiclass_best['F1 Score']:.4f}, Accuracy = {multiclass_best['Accuracy']:.4f}")
    
    # Best model for efficiency (if available)
    if efficiency_results is not None and not efficiency_results.empty:
        most_efficient = efficiency_results.iloc[0]
        print(f"\nMost efficient model: {most_efficient['Model']} ({most_efficient['Samples/Second']:.2f} samples/second)")
        
        # Best balance of performance and efficiency
        # Calculate a balanced score (F1 * Samples/Second / Parameters)
        if not efficiency_results.empty and not summary_table.empty:
            balanced_scores = []
            
            for model in efficiency_results['Model'].unique():
                eff_row = efficiency_results[efficiency_results['Model'] == model].iloc[0]
                
                # Find corresponding rows in summary table
                summary_rows = summary_table[summary_table['Model'] == model]
                
                for _, summary_row in summary_rows.iterrows():
                    balanced_score = (summary_row['F1 Score'] * eff_row['Samples/Second']) / eff_row['Parameters (M)']
                    
                    balanced_scores.append({
                        'Model': model,
                        'Task': summary_row['Task'],
                        'F1 Score': summary_row['F1 Score'],
                        'Samples/Second': eff_row['Samples/Second'],
                        'Parameters (M)': eff_row['Parameters (M)'],
                        'Balanced Score': balanced_score
                    })
            
            balanced_df = pd.DataFrame(balanced_scores)
            if not balanced_df.empty:
                best_balanced = balanced_df.sort_values('Balanced Score', ascending=False).iloc[0]
                
                print("\nBest balance of performance and efficiency:")
                print(f"{best_balanced['Model']} on {best_balanced['Task']} task")
                print(f"F1 Score: {best_balanced['F1 Score']:.4f}")
                print(f"Inference Speed: {best_balanced['Samples/Second']:.2f} samples/second")
                print(f"Model Size: {best_balanced['Parameters (M)']:.2f} million parameters")
    
    print("\nRecommendations:")
    print("1. For production environments where speed is critical, consider using a distilled model")
    print("2. For applications requiring high accuracy, especially in detecting hate speech, use the best performing model")
    print("3. For a balance of performance and efficiency, consider the model with the best balanced score")
    print("4. Further research should focus on reducing bias across target groups, as performance varies significantly")