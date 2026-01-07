"""Data loading and dataset utilities for disorder prediction."""

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample

from . import config


def load_trizod_dataset(path: Path = config.TRIZOD_DATASET) -> pd.DataFrame:
    """Load the TriZOD dataset with g-scores parsed as numpy arrays."""
    return pd.read_csv(
        path,
        index_col="label",
        converters={"gscores": lambda x: np.array(json.loads(x), dtype=float)},
    )


def load_st_classes(path: Path = config.ST_CLASSES_DATASET) -> pd.DataFrame:
    """Load the sentence transformer classification dataset."""
    return pd.read_csv(path, index_col="label")


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train and test based on 'is_test' column."""
    return df[~df["is_test"]], df[df["is_test"]]


def generate_sequence_pairs(
    sequences: list[str],
    classes: np.ndarray,
    num_iterations: int = config.ST_NUM_ITERATIONS,
) -> list[InputExample]:
    """
    Generate positive and negative sequence pairs for contrastive learning.
    
    For each sequence, generates one positive pair (same class) and one negative pair (different class).
    """
    pairs = []
    unique_classes = np.unique(classes)
    class_indices = {c: np.where(classes == c)[0] for c in unique_classes}
    # Precompute negative indices for each class
    neg_indices_per_class = {c: np.where(classes != c)[0] for c in unique_classes}
    
    for _ in range(num_iterations):
        for idx_a, seq_a in enumerate(sequences):
            class_a = classes[idx_a]
            
            # Positive pair: same class
            idx_b = np.random.choice(class_indices[class_a])
            pairs.append(InputExample(texts=[seq_a, sequences[idx_b]], label=1.0))
            
            # Negative pair: different class
            idx_neg = np.random.choice(neg_indices_per_class[class_a])
            pairs.append(InputExample(texts=[seq_a, sequences[idx_neg]], label=0.0))
    
    return pairs


def prepare_st_training_data(
    df_classes: pd.DataFrame,
    num_training: int | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare balanced training data for sentence transformer.
    
    Returns sequences and class labels with equal samples per class.
    
    Args:
        df_classes: DataFrame with 'sequence', 'class', and 'is_test' columns
        num_training: Number of samples per class (default: min class count)
        random_state: Random seed for reproducible sampling
    """
    df_train = df_classes[~df_classes["is_test"]]
    
    if num_training is None:
        num_training = df_train["class"].value_counts().min()
    
    # Sample equal number from each class (reproducible)
    balanced_df = pd.concat([
        df_train[df_train["class"] == k].sample(num_training, random_state=random_state)
        for k in sorted(df_train["class"].unique())  # Sort for deterministic order
    ])
    
    return balanced_df["sequence"].to_numpy(), balanced_df["class"].to_numpy()


def compute_rebalancing_weights(
    y: np.ndarray,
    a: float = config.REBALANCE_A,
    b: float = config.REBALANCE_B,
) -> np.ndarray:
    """
    Compute sample weights to rebalance the training set.
    
    This addresses the class imbalance between ordered and disordered regions.
    weights = min(1, a / (1 + exp(-b * (y - 0.5))))
    """
    return np.minimum(1, a / (1 + np.exp(-b * (y - 0.5))))


class ResidueDataset(torch.utils.data.Dataset):
    """Dataset for residue-level g-score prediction.
    
    Supports both regular numpy arrays and memory-mapped arrays.
    Converts to tensors on-demand to support large datasets that don't fit in RAM.
    """
    
    def __init__(self, embeddings: np.ndarray, targets: np.ndarray, weights: np.ndarray | None = None):
        # Keep as numpy arrays (or memmap) - convert to tensor on __getitem__
        self.embeddings = embeddings
        self.targets = targets
        self.weights = weights
        self._default_weight = torch.tensor(1.0)
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = torch.from_numpy(np.asarray(self.embeddings[idx])).float()
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        if self.weights is not None:
            weight = torch.tensor(self.weights[idx], dtype=torch.float32)
            return emb, target, weight
        return emb, target, self._default_weight


class FastaIterator:
    """Iterator over FASTA file, yielding chunks of (header, sequence) tuples."""
    
    def __init__(self, filename: str | Path, chunk_size: int = 1000):
        self.filename = Path(filename)
        self.chunk_size = chunk_size
        self._length: int | None = None
    
    def __iter__(self) -> Iterator[list[tuple[str, str]]]:
        with open(self.filename) as f:
            header = None
            sequence_parts = []
            chunk = []
            
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if header is not None:
                        chunk.append((header, "".join(sequence_parts)))
                        if len(chunk) == self.chunk_size:
                            yield chunk
                            chunk = []
                    header = line[1:]
                    sequence_parts = []
                else:
                    sequence_parts.append(line)
            
            # Last sequence
            if header is not None:
                chunk.append((header, "".join(sequence_parts)))
            
            if chunk:
                yield chunk
    
    def __len__(self) -> int:
        if self._length is None:
            with open(self.filename) as f:
                n_records = sum(1 for line in f if line.startswith(">"))
            self._length = int(np.ceil(n_records / self.chunk_size))
        return self._length

