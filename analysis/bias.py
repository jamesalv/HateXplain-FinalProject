import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import pandas as pd

def analyze_bias(results: Dict[str, Dict[str, Any]], task_type: str = '3class') -> None:
    """
    Analyze bias in model predictions across different target groups
    
    Args:
        results: Dictionary with model results
        task_type: Type of classification task ('binary' or '3class')
    """
    print(f"\n=== Bias Analysis for {task_type} Classification ===\n")
    
    for model_name, result in results[task_type].items():
        print(f"\nModel: {model_name}")
        
        if not result['target_metrics']:
            print("Target metrics not available for bias analysis")
            continue
        
        # Calculate standard deviation of performance across target groups
        target_values = list(result['target_metrics'].values())
        target_std = np.std(target_values)
        target_mean = np.mean(target_values)
        target_min = np.min(target_values)
        target_max = np.max(target_values)
        
        print(f"Performance across target groups:")
        print(f"Mean accuracy: {target_mean:.4f}")
        print(f"Standard deviation: {target_std:.4f}")
        print(f"Min accuracy: {target_min:.4f}")
        print(f"Max accuracy: {target_max:.4f}")
        print(f"Max-Min difference: {target_max - target_min:.4f}")
        
        # Identify target groups with worst and best performance
        worst_target = min(result['target_metrics'].items(), key=lambda x: x[1])
        best_target = max(result['target_metrics'].items(), key=lambda x: x[1])
        
        print(f"Target group with worst performance: {worst_target[0]} (accuracy: {worst_target[1]:.4f})")
        print(f"Target group with best performance: {best_target[0]} (accuracy: {best_target[1]:.4f})")
        
        # Print GMB metrics if they're available
        if 'bias_auc_metrics' in result:
            print("\nGMB Metrics:")
            for metric, value in result['bias_auc_metrics'].items():
                print(f"{metric}: {value:.4f}")

def plot_target_group_performance(results: Dict[str, Dict[str, Any]], task_type: str = '3class') -> None:
    """
    Plot performance by target group
    
    Args:
        results: Dictionary with model results
        task_type: Type of classification task ('binary' or '3class')
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    target_groups = set()
    for model_result in results[task_type].values():
        if model_result['target_metrics']:
            for target in model_result['target_metrics'].keys():
                target_groups.add(target)
    
    target_groups = sorted(target_groups)
    
    # Create data for plotting
    data = []
    for model_name, model_result in results[task_type].items():
        model_name_short = model_name.split('/')[-1]  # Get just the model name without path
        
        if model_result['target_metrics']:
            model_data = [model_result['target_metrics'].get(target, float('nan')) for target in target_groups]
            data.append((model_name_short, model_data))
    
    # Create plot
    x = np.arange(len(target_groups))
    width = 0.8 / len(data)  # Width of the bars
    
    for i, (model_name, model_data) in enumerate(data):
        offset = width * i - width * len(data) / 2 + width / 2
        plt.bar(x + offset, model_data, width, label=model_name)
    
    plt.xlabel('Target Group')
    plt.ylabel('AUROC Score')
    plt.title(f'Model Performance by Target Group ({task_type} Classification)')
    plt.xticks(x, target_groups, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def calculate_gmb_metrics(
    predictions: np.ndarray, 
    probabilities: np.ndarray, 
    true_labels: np.ndarray, 
    test_df: pd.DataFrame, 
    target_groups: List[str]
) -> Dict[str, float]:
    """
    Calculate GMB (Generalized Mean of Bias) AUC metrics from model predictions
    
    Args:
        predictions: Model's class predictions
        probabilities: Model's probability outputs
        true_labels: Ground truth labels
        test_df: DataFrame with test data including target groups
        target_groups: List of target groups to evaluate
        
    Returns:
        Dictionary with GMB metrics
    """
    # Create mappings from post_id to predictions and ground truth
    prediction_scores = {}
    ground_truth = {}
    
    for i, (_, row) in enumerate(test_df.iterrows()):
        post_id = row['post_id'] if 'post_id' in row else i
        
        # For binary classification, use probability of positive class
        # For multiclass, use probability of predicted class
        if probabilities.shape[1] == 2:  # Binary
            prediction_scores[post_id] = probabilities[i, 1]  # Score for positive class
        else:  # Multi-class
            prediction_scores[post_id] = probabilities[i, predictions[i]]
            
        # Convert to binary for bias evaluation (toxic vs non-toxic)
        if 'final_label' in row:
            if row['final_label'] in ['toxic', 'hatespeech', 'offensive']:
                ground_truth[post_id] = 1
            else:
                ground_truth[post_id] = 0
        else:
            # Use true_labels directly if final_label not in DataFrame
            ground_truth[post_id] = 1 if true_labels[i] > 0 else 0
    
    # Calculate metrics for each target group and method
    bias_metrics = defaultdict(lambda: defaultdict(dict))
    methods = ['subgroup', 'bpsn', 'bnsp']
    
    for method in methods:
        for group in target_groups:
            # Get positive and negative samples based on the method
            positive_ids, negative_ids = get_bias_evaluation_samples(test_df, method, group)
            
            if len(positive_ids) == 0 or len(negative_ids) == 0:
                print(f"Skipping {method} for group {group}: no samples found")
                continue  # Skip if no samples for this group/method
                
            # Collect ground truth and predictions
            y_true = []
            y_score = []
            
            for post_id in positive_ids:
                if post_id in ground_truth and post_id in prediction_scores:
                    y_true.append(ground_truth[post_id])
                    y_score.append(prediction_scores[post_id])
                
            for post_id in negative_ids:
                if post_id in ground_truth and post_id in prediction_scores:
                    y_true.append(ground_truth[post_id])
                    y_score.append(prediction_scores[post_id])
            
            # Calculate AUC if we have enough samples with both classes
            if len(y_true) > 10 and len(set(y_true)) > 1:
                try:
                    auc = roc_auc_score(y_true, y_score)
                    bias_metrics[method][group] = auc
                except ValueError:
                    # Skip if there's an issue with ROC AUC calculation
                    pass
    
    # Calculate GMB for each method
    gmb_metrics = {}
    power = -5  # Power parameter for generalized mean
    
    for method in methods:
        if not bias_metrics[method]:
            continue
            
        scores = list(bias_metrics[method].values())
        if not scores:
            continue
            
        # Calculate generalized mean with p=-5
        power_mean = np.mean([score ** power for score in scores]) ** (1/power)
        gmb_metrics[f'GMB-{method.upper()}-AUC'] = power_mean
    
    # Calculate a combined GMB score that includes all methods
    all_scores = []
    for method in methods:
        all_scores.extend(list(bias_metrics[method].values()))
    
    if all_scores:
        gmb_metrics['GMB-COMBINED-AUC'] = np.mean([score ** power for score in all_scores]) ** (1/power)
    
    return gmb_metrics, bias_metrics

def plot_gmb_metrics(results: Dict[str, Dict[str, Any]], task_type: str = '3class') -> None:
    """
    Plot Generalized Mean of Bias (GMB) metrics for all models
    
    Args:
        results: Dictionary with model results
        task_type: Type of classification task ('binary' or '3class')
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract metrics from results
    models = []
    subgroup_scores = []
    bpsn_scores = []
    bnsp_scores = []
    combined_scores = []
    
    for model_name, result in results[task_type].items():
        if 'bias_auc_metrics' not in result or not result['bias_auc_metrics']:
            continue
            
        # Get shortened model name for better display
        model_short = model_name.split('/')[-1]
        models.append(model_short)
        
        metrics = result['bias_auc_metrics']
        subgroup_scores.append(metrics.get('GMB-SUBGROUP-AUC', 0))
        bpsn_scores.append(metrics.get('GMB-BPSN-AUC', 0))
        bnsp_scores.append(metrics.get('GMB-BNSP-AUC', 0))
        combined_scores.append(metrics.get('GMB-COMBINED-AUC', 0))
    
    if not models:
        print("No GMB metrics available for plotting")
        return
        
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Position of bars on x-axis
    x = np.arange(len(models))
    width = 0.2  # Width of the bars
    
    # Create bars
    rects1 = ax.bar(x - width*1.5, subgroup_scores, width, label='Subgroup AUC')
    rects2 = ax.bar(x - width/2, bpsn_scores, width, label='BPSN AUC')
    rects3 = ax.bar(x + width/2, bnsp_scores, width, label='BNSP AUC')
    rects4 = ax.bar(x + width*1.5, combined_scores, width, label='Combined AUC')
    
    # Add labels, title and legend
    ax.set_xlabel('Models')
    ax.set_ylabel('GMB AUC Score')
    ax.set_title(f'Generalized Mean of Bias (GMB) Metrics - {task_type} Classification')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on top of bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    add_labels(rects4)
    
    # Set y-axis to start from 0.5 for better visualization of differences
    ax.set_ylim(0.5, 1.0)
    
    # Add a horizontal line at 0.8 as a reference for good performance
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.3)
    ax.text(x[-1] + 0.5, 0.8, '0.8 (Good Performance)', va='center', alpha=0.7)
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Ensure layout looks good
    fig.tight_layout()
    
    # Show plot
    plt.show()

    # Also create a separate plot focused on the combined score for clearer comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, combined_scores, color='teal')
    
    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Combined GMB AUC Score')
    ax.set_title(f'Combined Bias Score Comparison - {task_type} Classification')
    ax.set_xticklabels(models, rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Set y-axis to start from 0.5
    ax.set_ylim(0.5, 1.0)
    
    # Add a horizontal line at 0.8
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.3)
    ax.text(len(models) - 1, 0.8, '0.8 (Good Performance)', va='center', alpha=0.7)
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Ensure layout looks good
    fig.tight_layout()
    
    # Show plot
    plt.show()
    

def get_bias_evaluation_samples(data, method, group):
    """
    Get positive and negative sample IDs for bias evaluation based on method and group
    
    Args:
        data: DataFrame with test data
        method: Bias evaluation method ('subgroup', 'bpsn', or 'bnsp')
        group: Target group to evaluate
        
    Returns:
        Tuple of (positive_ids, negative_ids)
    """
    positive_ids = []
    negative_ids = []
    
    for _, row in data.iterrows():
        # Skip if no target_groups column or no post_id
        if 'target_groups' not in row or 'post_id' not in row:
            continue
            
        target_groups = row['target_groups']
        if target_groups is None:
            continue
            
        post_id = row['post_id']
        is_in_group = group in target_groups
        
        # Convert various label formats to binary toxic/non-toxic
        if 'final_label' in row:
            is_toxic = row['final_label'] in ['toxic', 'hatespeech', 'offensive']
        else:
            continue
        
        if method == 'subgroup':
            # Only consider samples mentioning the group
            if is_in_group:
                if is_toxic:
                    positive_ids.append(post_id)
                else:
                    negative_ids.append(post_id)
                    
        elif method == 'bpsn':
            # Compare non-toxic posts mentioning the group with toxic posts NOT mentioning the group
            if is_in_group and not is_toxic:
                negative_ids.append(post_id)
            elif not is_in_group and is_toxic:
                positive_ids.append(post_id)
                
        elif method == 'bnsp':
            # Compare toxic posts mentioning the group with non-toxic posts NOT mentioning the group
            if is_in_group and is_toxic:
                positive_ids.append(post_id)
            elif not is_in_group and not is_toxic:
                negative_ids.append(post_id)
    
    return positive_ids, negative_ids