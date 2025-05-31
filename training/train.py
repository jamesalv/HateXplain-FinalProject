import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Any, Union, Optional
from analysis.efficiency import analyze_efficiency
from analysis.errors import analyze_errors
from data.preprocessing.pipeline import preprocess_datasets
from models.classifier import TransformerClassifier
from data.dataset import prepare_data_loaders
from analysis.bias import calculate_gmb_metrics
from transformers import AutoTokenizer


def train_and_evaluate_model(
    model_name: str,
    data_df: pd.DataFrame,
    num_classes: int,
    batch_size: int = 16,
    epochs: int = 4,
    max_length: int = 128,
    learning_rate: float = 2e-5,
    warmup_steps: int = 0,
    model_dir: str = "saved_models",
    auto_weighted: bool = False,
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    classifier_dropout: float = 0.1,
    custom_classifier_head: bool = False,
    weight_decay: float = 0.01,
    patience: int = 2,
    min_delta: float = 0.001,
    monitor: str = "val_loss",
    # RATIONALE PARAMS
    lam: float = 1.0,  # Fixed: should be float, not int
    use_attention_supervision: bool = True,  # Fixed: missing parameter
    temperature: float = 1.0,  # Fixed: missing parameter
) -> Dict[str, Any]:
    """
    Train and evaluate a model with the given parameters

    Args:
        model_name: Name of the Hugging Face transformer model
        data_df: DataFrame with preprocessed data
        num_classes: Number of classes (2 or 3)
        batch_size: Batch size for training
        epochs: Number of training epochs
        max_length: Maximum sequence length for tokenization
        learning_rate: Learning rate for the optimizer
        warmup_steps: Number of warmup steps for learning rate scheduler
        model_dir: Directory to save models
        auto_weighted: Whether to use automatic weighting for classes
        hidden_dropout_prob: Dropout probability for hidden layers
        attention_probs_dropout_prob: Dropout probability for attention layers
        classifier_dropout: Dropout probability for the classifier head
        custom_classifier_head: Whether to use a custom classifier head
        weight_decay: Weight decay for the optimizer

    Returns:
        Dictionary with results
    """

    # Prepare data loaders
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        label_map,
        test_df,
        label_to_weight,
    ) = prepare_data_loaders(
        data_df,
        batch_size=batch_size,
        auto_weighted=auto_weighted,
    )

    # Invert label map for later use
    inv_label_map = {v: k for k, v in label_map.items()}

    class_weights = {
        label_map[label]: weight for label, weight in label_to_weight.items()
    }
    print(f"Class weights: {class_weights}")
    ordered_weights = np.array([class_weights[i] for i in range(len(class_weights))])

    # Initialize model
    classifier = TransformerClassifier(
        model_name,
        num_classes,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        classifier_dropout=classifier_dropout,
        custom_classifier_head=custom_classifier_head,
        class_weights=ordered_weights,
        lam=lam,
        use_attention_supervision=use_attention_supervision,
        temperature=temperature,
    )

    # Train model
    history = classifier.train(
        train_dataloader,
        val_dataloader,
        epochs=epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        monitor=monitor,
    )

    # Evaluate on test set
    test_loss, test_accuracy, test_f1 = classifier.evaluate(test_dataloader)
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1 Score: {test_f1:.4f}")

    # Get detailed metrics
    predictions, true_labels, probabilities, attention_weights = classifier.predict(test_dataloader)

    # Convert numeric labels back to text
    text_preds = [inv_label_map[pred] for pred in predictions]
    text_true = [inv_label_map[label] for label in true_labels]

    # Calculate additional metrics
    precision = precision_score(true_labels, predictions, average="macro")
    recall = recall_score(true_labels, predictions, average="macro")

    # For AUROC, we need to handle multiclass case
    if num_classes == 2:
        auroc = roc_auc_score(true_labels, probabilities[:, 1])
    else:
        # One-vs-Rest approach for multiclass
        auroc = roc_auc_score(
            np.eye(num_classes)[true_labels],
            probabilities,
            average="macro",
            multi_class="ovr",
        )

    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(label_map.keys()),
        yticklabels=list(label_map.keys()),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

    # Save model with descriptive name
    model_type = "binary" if num_classes == 2 else "3class"
    model_save_path = os.path.join(
        model_dir, f"{model_name.replace('/', '_')}_{model_type}"
    )
    classifier.save_model(model_save_path)

    if "target_groups" in test_df.columns:
        # Get list of target groups
        all_targets = []
        for targets in test_df["target_groups"]:
            if targets is not None:
                all_targets.extend(targets)
        all_targets = [
            target for target in all_targets if target not in ["None", "Other"]
        ]
        from collections import Counter

        top_targets = Counter(all_targets).most_common(10)
        target_groups = [target for target, _ in top_targets]

        # Calculate GMB metrics and bias_metrics (AUROC per target group)
        gmb_metrics, bias_metrics = calculate_gmb_metrics(
            predictions, probabilities, true_labels, test_df, target_groups
        )

        # Print GMB metrics
        print("\nBias AUC Metrics:")
        for metric, value in gmb_metrics.items():
            print(f"{metric}: {value:.4f}")

        # Print target-specific metrics using BNSP (like the original paper)
        if bias_metrics.get("bnsp"):
            print("\nBNSP AUROC by Target Group:")
            target_metrics = {}
            for target, auc_score in bias_metrics["bnsp"].items():
                print(f"{target}: AUROC = {auc_score:.4f}")
                target_metrics[target] = auc_score
        else:
            target_metrics = None
    else:
        gmb_metrics = None
        bias_metrics = None
        target_metrics = None

    # Return comprehensive results
    results = {
        "model_name": model_name,
        "num_classes": num_classes,
        "history": history,
        "metrics": {
            "accuracy": test_accuracy,
            "f1_score": test_f1,
            "precision": precision,
            "recall": recall,
            "auroc": auroc,
            "loss": test_loss,
        },
        "confusion_matrix": cm,
        "label_map": label_map,
        "target_metrics": target_metrics,
        "bias_auc_metrics": gmb_metrics,
    }

    return results  # Replace with actual results dictionary when model is implemented


def run_model_comparison(
    models_to_compare: List[str],
    batch_size: int = 16,
    epochs: int = 3,
    auto_weighted: bool = False,
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    classifier_dropout: float = 0.1,
    custom_classifier_head: bool = False,
    weight_decay: float = 0.01,
    patience: int = 2,
    min_delta: float = 0.001,
    monitor: str = "val_loss",
    lam: float = 1.0,  # Fixed: should be float, not int
    use_attention_supervision: bool = True,  # Fixed: missing parameter
    temperature: float = 1.0,  # Fixed: missing parameter
) -> Dict[str, Dict[str, Any]]:
    """
    Run comparison of multiple models on both binary and 3-class tasks

    Args:
        models_to_compare: List of model names to compare
        data_3class: DataFrame with 3-class data
        data_2class: DataFrame with binary data
        batch_size: Batch size for training
        epochs: Number of training epochs
        hidden_dropout_prob: Dropout probability for hidden layers
        attention_probs_dropout_prob: Dropout probability for attention layers
        classifier_dropout: Dropout probability for the classifier head
        custom_classifier_head: Whether to use a custom classifier head
        weight_decay: Weight decay for the optimizer

    Returns:
        Dictionary with all results
    """
    results = {"binary": {}, "3class": {}}

    # First run 3-class models
    print("\n=== Running 3-Class Classification Models ===\n")
    for model_name in models_to_compare:
        print(f"\nTraining {model_name} for 3-class classification")
        # Process the data for this model
        data_3class, data_2class = preprocess_datasets(
            data_path='Raw Data/dataset.json',
            model_name=model_name,
        )
        results["3class"][model_name] = train_and_evaluate_model(
            model_name,
            data_3class,
            num_classes=3,
            batch_size=batch_size,
            epochs=epochs,
            auto_weighted=auto_weighted,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            classifier_dropout=classifier_dropout,
            custom_classifier_head=custom_classifier_head,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            monitor=monitor,
            lam=lam,
            use_attention_supervision=use_attention_supervision,
            temperature=temperature,
        )
        # analyze_errors(
        #     results, 
        #     data_3class, 
        #     task_type='3class',
        #     model_name=model_name
        # )
        
        print(f"\nTraining {model_name} for binary classification")
        results["binary"][model_name] = train_and_evaluate_model(
            model_name,
            data_2class,
            num_classes=2,
            batch_size=batch_size,
            epochs=epochs,
            auto_weighted=auto_weighted,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            classifier_dropout=classifier_dropout,
            custom_classifier_head=custom_classifier_head,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            monitor=monitor,
            lam=lam,
            use_attention_supervision=use_attention_supervision,
            temperature=temperature,
        )
        # analyze_errors(
        #     results, 
        #     data_2class, 
        #     task_type='binary',
        #     model_name=model_name
        # )

    # efficiency_results = analyze_efficiency(
    #     models_to_compare,
    #     data_3class,
    #     num_classes=3,
    #     batch_size=32
    # )
    # results["efficiency"] = efficiency_results
    return results
