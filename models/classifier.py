import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
import torch.nn.functional as F


# Roberta classifier acts different than other models, causing
# dimension discrepancy
class CustomRobertaHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        # features: [batch_size, seq_len, hidden_size]
        x = features[:, 0, :]  # pool on the first token
        x = self.dense(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.out_proj(x)  # → [batch_size, num_labels]


class TransformerClassifier:
    """Transformer-based classifier for hate speech detection"""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        classifier_dropout: float = 0.1,
        custom_classifier_head: bool = False,
        class_weights: Optional[np.ndarray] = None,
        # RATIONALE PARAMS
        lam: float = 1.0,  # Fixed: should be float, not int
        use_attention_supervision: bool = True,  # Fixed: missing parameter
        temperature: float = 1.0,  # Fixed: missing parameter
        device=None,
    ):
        """
        Initialize the classifier.

        Args:
            model_name: Hugging Face model name
            num_labels: Number of output classes
            hidden_dropout_prob: Dropout rate for hidden layers
            attention_probs_dropout_prob: Dropout rate for attention probabilities
            classifier_dropout: Dropout rate for classifier
            custom_classifier_head: Whether to use a custom classifier head
            device: Torch device (will use GPU if available when None)
        """
        self.model_name = model_name
        self.num_labels = num_labels

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using device: {self.device}")

        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = attention_probs_dropout_prob

        # Some models don't have classifier_dropout in their config
        # If it exists in the config, use it; otherwise, add it later on our
        # custom classifier head
        # This is a workaround for models that don't have this parameter in their config
        if hasattr(config, "classifier_dropout"):
            config.classifier_dropout = classifier_dropout
        elif hasattr(config, "classifier_dropout_prob"):
            config.classifier_dropout_prob = classifier_dropout
        else:
            print(
                f"No classifier dropout parameter found in config. Model: {model_name}"
            )
            print(
                f"Please use custom classifier head if you want to implement for this model"
            )
            pass

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config
        )

        # Optional custom classifier head
        if custom_classifier_head:
            print(f"Using custom classifier head for model: {model_name}")
            # Get the size of the hidden states
            hidden_size = config.hidden_size

            if model_name == "roberta-base":
                self.model.classifier = CustomRobertaHead(
                    hidden_size=hidden_size,
                    num_labels=num_labels,
                    dropout=classifier_dropout,
                )
            else:
                # Replace the classifier with a custom one
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(classifier_dropout),
                    torch.nn.Linear(hidden_size, num_labels),
                )

        # Setup loss function with class weights if provided
        if class_weights is not None:
            # Convert class weights to tensor of type float32 and move to device
            weight_tensor = torch.tensor(
                [class_weights[i] for i in range(self.num_labels)],
                device=self.device,
                dtype=torch.float32,  # Explicitly specify float32
            )
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        self.att_loss_fn = nn.CrossEntropyLoss()
        # Rationale parameters
        self.lam = lam  # Add this
        self.use_attention_supervision = use_attention_supervision  # Add this  
        self.temperature = temperature  # Add this
        self.model.to(self.device)

    def calculate_attention_loss(
        self, human_rationales, model_attention, lambd: float = 1.0
    ):
        """Calculate attention loss using cross-entropy between human rationales and model attention."""
        if human_rationales is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Ensure tensors are on the same device
        human_rationales = human_rationales.to(self.device)
        model_attention = model_attention.to(self.device)

        # Apply temperature scaling to model attention
        model_attention_scaled = model_attention / self.temperature

        # Use CrossEntropyLoss with soft targets (probability distributions)
        # human_rationales are already softmaxed probabilities, use them directly
        attention_loss = self.att_loss_fn(model_attention_scaled, human_rationales)

        return lambd * attention_loss

    def extract_attention_weights(self, attentions):
        """Extract attention weights from transformer attention outputs from cls layer"""
        if attentions is None or len(attentions) == 0:
            return None

        # Use CLS token attention from the last layer
        last_layer_attention = attentions[
            -1
        ]  # [batch_size, num_heads, seq_len, seq_len]
        cls_attention = last_layer_attention.mean(dim=1)[
            :, 0, :
        ]  # [batch_size, seq_len]
        return cls_attention

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        patience: int = 2,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
    ) -> Dict[str, List[float]]:
        """
        Train the transformer model

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            class_weights: Class weights for loss function (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            warmup_steps: Number of warmup steps for scheduler
            weight_decay: Weight decay for optimizer
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in monitored value to qualify as improvement
            monitor: Metric to monitor for early stopping ('val_loss', 'val_accuracy', or 'val_f1')

        Returns:
            Dictionary with training history
        """
        # Initialize optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            eps=1e-8,
            weight_decay=weight_decay,
        )

        # Total number of training steps
        total_steps = len(train_dataloader) * epochs

        # Set up learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Initialize training history
        history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

        # Early stopping variables
        best_metric_value = float("inf") if monitor == "val_loss" else -float("inf")
        best_epoch = 0
        no_improve_count = 0
        best_model_state = None

        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # Training phase
            self.model.train()
            total_train_loss = 0

            # Progress bar for training
            progress_bar = tqdm(train_dataloader, desc="Training", unit="batch")

            for batch in progress_bar:
                # Clear previous gradients
                self.model.zero_grad()

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,  # Enable attention outputs
                )

                logits = outputs.logits
                # UPDATE: Added support for attention supervision
                classification_loss = self.loss_fn(logits, labels)
                # Extract attention weights if available
                attentions = (
                    outputs.attentions if hasattr(outputs, "attentions") else None
                )
                model_attention = self.extract_attention_weights(attentions)
                # Calculate attention loss if human rationales are provided
                human_rationales = batch.get("rationales", None)
                attention_loss = torch.tensor(0.0, device=self.device)
                if self.use_attention_supervision and human_rationales is not None:
                    model_attention = self.extract_attention_weights(attentions)
                    if model_attention is not None:
                        attention_loss = self.calculate_attention_loss(
                            human_rationales, model_attention, self.lam
                        )
                else:
                    print(
                        "Attention supervision is disabled or human rationales are not provided."
                    )
                # Total loss is a combination of classification and attention loss
                loss = classification_loss + attention_loss
                # Move loss to device
                loss = loss.to(self.device)
                # Accumulate training loss
                total_train_loss += loss.item()

                # Backward pass
                loss.backward()

                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters
                optimizer.step()

                # Update learning rate
                scheduler.step()

                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})

            # Calculate average training loss
            avg_train_loss = total_train_loss / len(train_dataloader)
            history["train_loss"].append(avg_train_loss)

            print(f"Average training loss: {avg_train_loss:.4f}")

            # Validation phase
            val_loss, val_accuracy, val_f1 = self.evaluate(val_dataloader)

            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            history["val_f1"].append(val_f1)

            print(f"Validation loss: {val_loss:.4f}")
            print(f"Validation accuracy: {val_accuracy:.4f}")
            print(f"Validation F1 score: {val_f1:.4f}")

            # Check for early stopping
            current_metric = (
                val_loss
                if monitor == "val_loss"
                else val_accuracy if monitor == "val_accuracy" else val_f1
            )
            if monitor == "val_loss":
                improved = current_metric < best_metric_value - min_delta
            else:
                improved = current_metric > best_metric_value + min_delta

            if improved:
                print(
                    f"Validation {monitor} improved from {best_metric_value:.4f} to {current_metric:.4f}"
                )
                best_metric_value = current_metric
                best_epoch = epoch
                no_improve_count = 0

                # Save best model state
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                no_improve_count += 1
                print(
                    f"Validation {monitor} did not improve. Best: {best_metric_value:.4f}, Current: {current_metric:.4f}"
                )
                print(f"Early stopping counter: {no_improve_count}/{patience}")

                if no_improve_count >= patience:
                    print(
                        f"Early stopping triggered. Best {monitor}: {best_metric_value:.4f} at epoch {best_epoch+1}"
                    )
                    break

        # Load best model if early stopping occurred
        if best_model_state is not None and best_epoch < epoch:
            print(f"Loading best model from epoch {best_epoch+1}")
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_model_state.items()}
            )

        return history

    def evaluate(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[float, float, float]:
        """
        Evaluate the model on a dataset

        Args:
            dataloader: DataLoader with evaluation data

        Returns:
            loss, accuracy, f1_score
        """
        from sklearn.metrics import accuracy_score, f1_score

        self.model.eval()

        total_loss = 0
        all_preds = []
        all_labels = []

        # No gradient computation for evaluation
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=True,
                )

                classification_loss = outputs.loss
                # Extract attention weights if available
                attentions = (
                    outputs.attentions if hasattr(outputs, "attentions") else None
                )
                model_attention = self.extract_attention_weights(attentions)
                # Calculate attention loss if human rationales are provided
                human_rationales = batch.get("rationales", None)
                attention_loss = torch.tensor(0.0, device=self.device)
                if self.use_attention_supervision and human_rationales is not None:
                    model_attention = self.extract_attention_weights(attentions)
                    if model_attention is not None:
                        attention_loss = self.calculate_attention_loss(
                            human_rationales, model_attention, self.lam
                        )
                # Total loss is a combination of classification and attention loss
                loss = classification_loss + attention_loss
                total_loss += loss.item()

                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                # Store predictions and labels
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        # Calculate loss
        avg_loss = total_loss / len(dataloader)

        return avg_loss, accuracy, f1

    def predict(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with the model

        Args:
            dataloader: DataLoader with test data

        Returns:
            predictions, true_labels, probabilities
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        # No gradient computation for prediction
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Get predictions
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                # Store predictions, probabilities, and labels
                all_preds.extend(preds)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert to numpy arrays
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def save_model(self, path: str) -> None:
        """
        Save model to disk

        Args:
            path: Directory path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save model
        self.model.save_pretrained(path)

        # Save tokenizer
        self.tokenizer.save_pretrained(path)

        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model from disk

        Args:
            path: Directory path to load the model from
        """
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        print(f"Model loaded from {path}")
