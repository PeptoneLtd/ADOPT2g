"""Data loading and dataset utilities for disorder prediction."""

import json
from collections.abc import Iterator
from pathlib import Path

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


def split_train_val(
    df: pd.DataFrame,
    val_frac: float = config.DEFAULT_MLP_VAL_FRAC,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training data into train and validation sets with stratification.

    Stratifies by mean_gscore >= 0.5 (ordered vs disordered) at sequence level.

    Args:
        df: DataFrame with training sequences (should NOT include test set)
        val_frac: Fraction of data to use for validation
        random_state: Random seed for reproducible splits

    Returns:
        (df_train, df_val) tuple
    """
    # Create stratification column based on mean_gscore
    strat_col = (df["mean_gscore"] >= 0.5).astype(int)

    # Perform stratified split
    rng = np.random.default_rng(random_state)
    train_indices = []
    val_indices = []

    for stratum in [0, 1]:
        stratum_indices = df.index[strat_col == stratum].tolist()
        rng.shuffle(stratum_indices)
        n_val = int(len(stratum_indices) * val_frac)
        val_indices.extend(stratum_indices[:n_val])
        train_indices.extend(stratum_indices[n_val:])

    return df.loc[train_indices], df.loc[val_indices]


def sample_st_training_data(
    df_classes: pd.DataFrame,
    train_frac: float = config.DEFAULT_ST_TRAIN_FRAC,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a subset of ST training data for ensemble diversity.

    Creates a balanced subset by sampling train_frac from each class.

    Args:
        df_classes: DataFrame with 'sequence', 'class', and 'is_test' columns
        train_frac: Fraction of each class to sample (default 0.9)
        random_state: Random seed for reproducible sampling

    Returns:
        (sequences, classes) arrays
    """
    df_train = df_classes[~df_classes["is_test"]]

    # Get min class count and compute sample size
    class_counts = df_train["class"].value_counts()
    min_count = class_counts.min()
    num_samples = max(1, int(min_count * train_frac))

    # Sample equal number from each class
    balanced_df = pd.concat(
        [
            df_train[df_train["class"] == k].sample(num_samples, random_state=random_state)
            for k in sorted(df_train["class"].unique())
        ]
    )

    return balanced_df["sequence"].to_numpy(), balanced_df["class"].to_numpy()


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
    balanced_df = pd.concat(
        [
            df_train[df_train["class"] == k].sample(num_training, random_state=random_state)
            for k in sorted(df_train["class"].unique())  # Sort for deterministic order
        ]
    )

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

    def __init__(self, embeddings: np.ndarray, targets: np.ndarray, weights: np.ndarray | None = None) -> None:
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

    def __init__(self, filename: str | Path, chunk_size: int = 1000) -> None:
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


class CsvIterator:
    """Iterator over CSV file, yielding chunks of (label, sequence) tuples."""

    def __init__(
        self,
        filename: str | Path,
        chunk_size: int = 1000,
        label_col: str = "label",
        sequence_col: str = "sequence",
    ) -> None:
        self.filename = Path(filename)
        self.chunk_size = chunk_size
        self.label_col = label_col
        self.sequence_col = sequence_col
        self._length: int | None = None

        # Read column names to verify existence of label and sequence
        df_head = pd.read_csv(self.filename, nrows=0)
        if self.label_col not in df_head.columns or self.sequence_col not in df_head.columns:
            raise ValueError(
                f"CSV file must contain '{self.label_col}' and '{self.sequence_col}' columns. "
                f"Found: {df_head.columns.tolist()}"
            )

    def __iter__(self) -> Iterator[list[tuple[str, str]]]:
        for chunk in pd.read_csv(self.filename, chunksize=self.chunk_size):
            yield list(
                zip(
                    chunk[self.label_col].astype(str),
                    chunk[self.sequence_col].astype(str),
                )
            )

    def __len__(self) -> int:
        if self._length is None:
            # Efficiently count lines for large CSVs
            with open(self.filename, "rb") as f:
                n_records = sum(1 for _ in f) - 1  # -1 for header
            self._length = int(np.ceil(n_records / self.chunk_size))
        return self._length
