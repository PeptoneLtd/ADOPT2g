"""Centralized configuration for the disorder predictor."""

from pathlib import Path

# Paths
ROOT_PATH = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = ROOT_PATH / "cache"
DATASETS_PATH = ROOT_PATH / "datasets"
MODELS_PATH = ROOT_PATH / "models"

# ESM model for residue-level embeddings (dim 1280)
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

# Sentence transformer base model for sequence-level embeddings (dim 320)
ST_BASE_MODEL = "facebook/esm2_t6_8M_UR50D"

# Dataset files
TRIZOD_DATASET = DATASETS_PATH / "cleaned_trizod_dataset.csv"
ST_CLASSES_DATASET = DATASETS_PATH / "st_classes.csv"

# Training hyperparameters
DEFAULT_SEED = 883

# Sentence transformer training
ST_NUM_ITERATIONS = 5
ST_BATCH_SIZE = 128
ST_NUM_EPOCHS = 4
ST_WARMUP_RATIO = 0.1

# MLP training
MLP_HIDDEN_DIMS = [512, 128]
MLP_DROPOUT = 0.1
MLP_LR = 1e-4
MLP_WEIGHT_DECAY = 1e-5
MLP_EPOCHS = 50
MLP_BATCH_SIZE = 4096

# Rebalancing weights for training, see `datasets/visualize.ipynb` for more details
REBALANCE_A = 16.229
REBALANCE_B = 10.670

# Input dimensions
ESM_EMBED_DIM = 1280
ST_EMBED_DIM = 320
TOTAL_EMBED_DIM = ESM_EMBED_DIM + ST_EMBED_DIM

# Max sequence length for ESM model
MAX_SEQUENCE_LEN = 1022

# Weights & Biases configuration
WANDB_PROJECT = "adopt2g"
WANDB_ENTITY = None

# Ensemble configuration
DEFAULT_ENSEMBLE_SIZE = 7
DEFAULT_MLP_VAL_FRAC = 0.1
DEFAULT_ST_TRAIN_FRAC = 0.9
