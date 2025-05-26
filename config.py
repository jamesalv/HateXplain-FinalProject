import torch
import os

# Paths
DATA_DIR = "Raw Data"
DATASET_PATH = os.path.join(DATA_DIR, "dataset.json")
RESULTS_DIR = "results"
MODEL_DIR = "saved_models"

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Training parameters
BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
EPOCHS = 5
WARMUP_STEPS = 0
TEST_SIZE = 0.1
VAL_SIZE = 0.1
RANDOM_SEED = 42

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default models to compare
MODELS = [
    'bert-base-uncased',
    'distilbert-base-uncased',
    'roberta-base',
    'xlnet-base-cased',
    'microsoft/deberta-v3-base'
]