import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any


def plot_training_history(
    results: Dict[str, Dict[str, Any]], task_type: str = "3class"
) -> None:
    """
    Plot training history for all models

    Args:
        results: Dictionary containing model results
        task_type: Type of classification task ('binary' or '3class')
    """
    plt.figure(figsize=(15, 10))

    # Plot training loss
    plt.subplot(2, 2, 1)
    for model_name, result in results[task_type].items():
        plt.plot(result["history"]["train_loss"], label=model_name)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot validation loss
    plt.subplot(2, 2, 2)
    for model_name, result in results[task_type].items():
        plt.plot(result["history"]["val_loss"], label=model_name)
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot validation accuracy
    plt.subplot(2, 2, 3)
    for model_name, result in results[task_type].items():
        plt.plot(result["history"]["val_accuracy"], label=model_name)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot validation F1 score
    plt.subplot(2, 2, 4)
    for model_name, result in results[task_type].items():
        plt.plot(result["history"]["val_f1"], label=model_name)
    plt.title("Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_performance_comparison(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Plot performance metrics comparison between models

    Args:
        results: Dictionary containing model results
    """
    # Prepare data for plotting
    task_types = ["binary", "3class"]
    metrics = ["accuracy", "f1_score", "precision", "recall", "auroc"]

    for task_type in task_types:
        plt.figure(figsize=(15, 10))

        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i + 1)

            model_names = []
            metric_values = []

            for model_name, result in results[task_type].items():
                model_names.append(
                    model_name.split("/")[-1]
                )  # Get just the model name without path
                metric_values.append(result["metrics"][metric])

            # Create bar plot
            bars = plt.bar(model_names, metric_values)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    rotation=0,
                )

            plt.title(f'{metric.replace("_", " ").title()}')
            plt.ylim(0, 1.1)  # All metrics are between 0 and 1
            plt.xticks(rotation=45, ha="right")

        plt.suptitle(
            f"Performance Comparison for {task_type} Classification", fontsize=16
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make space for the suptitle
        plt.show()
