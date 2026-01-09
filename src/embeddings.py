"""Embedding extraction and caching for ESM models."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from . import config


def load_esm_model(
    model_name: str = config.ESM_MODEL_NAME,
    device: str | None = None,
) -> tuple[AutoTokenizer, AutoModel]:
    """Load ESM model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float16)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return tokenizer, model


def extract_esm_embeddings(
    df: pd.DataFrame,
    output_dir: Path,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 16,
    overwrite: bool = False,
) -> None:
    """
    Extract and cache ESM embeddings for all sequences in a DataFrame.

    Embeddings are saved as .pt files named by the sequence label.
    Processes sequences in batches for better performance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    # Identify which labels need processing
    if not overwrite:
        indices_to_process = [idx for idx in df.index if not (output_dir / f"{idx}.pt").exists()]
    else:
        indices_to_process = df.index.tolist()

    if not indices_to_process:
        print("All ESM embeddings are already cached.")
        return

    for i in tqdm(range(0, len(indices_to_process), batch_size), desc="Extracting ESM embeddings"):
        batch_indices = indices_to_process[i : i + batch_size]
        batch_sequences = df.loc[batch_indices, "sequence"].tolist()

        # Tokenize and run model
        inputs = tokenizer(
            batch_sequences, return_tensors="pt", padding=True, truncation=True, max_length=config.MAX_SEQUENCE_LEN + 2,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        # Save each embedding in the batch
        for j, label in enumerate(batch_indices):
            # Get actual length (excluding padding) from attention mask
            seq_len = inputs["attention_mask"][j].sum().item()
            # Keep shape (1, seq_len, dim) for compatibility with load_esm_embedding
            embedding = output.last_hidden_state[j, :seq_len].unsqueeze(0).cpu()

            torch.save(
                {"last_hidden_state": embedding},
                output_dir / f"{label}.pt",
            )


def load_esm_embedding(path: Path) -> torch.Tensor:
    """Load a cached ESM embedding from disk."""
    # weights_only=True is safer but requires the saved format to be just tensors
    # Our format uses a dict with tensor values, which is supported with weights_only=True
    data = torch.load(path, weights_only=True)
    return data["last_hidden_state"][0]  # Remove batch dimension


def encode_sequences_with_st(
    sequences: list[str],
    model: SentenceTransformer,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode sequences using a SentenceTransformer model."""
    return model.encode(sequences, show_progress_bar=show_progress)


def _count_valid_residues(df: pd.DataFrame, drop_missing: bool) -> int:
    """Count total valid residues for pre-allocation."""
    total = 0
    for label in df.index:
        gscores = df.loc[label, "gscores"]
        if drop_missing:
            total += np.sum(~np.isnan(gscores))
        else:
            total += len(gscores)
    return int(total)


def prepare_residue_embeddings(
    df: pd.DataFrame,
    esm_cache_dir: Path,
    st_model: SentenceTransformer,
    drop_missing: bool = True,
    use_memmap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare concatenated residue-level embeddings for training/evaluation.

    For each residue, concatenates:
    - ESM residue embedding (dim 1280)
    - Sequence-level ST embedding (dim 320, tiled for all residues)

    Args:
        df: DataFrame with sequences and gscores
        esm_cache_dir: Directory with cached ESM embeddings
        st_model: SentenceTransformer model for sequence embeddings
        drop_missing: Whether to drop residues with NaN g-scores
        use_memmap: Use memory-mapped arrays to reduce RAM usage (recommended for large datasets)

    Returns (embeddings, g_scores) arrays.
    """
    # Compute sequence-level embeddings
    sequences = df["sequence"].tolist()
    st_embeddings = encode_sequences_with_st(sequences, st_model)
    st_repr = dict(zip(df.index, st_embeddings, strict=True))

    # Count total residues and pre-allocate arrays
    print("Counting residues for pre-allocation...")
    total_residues = _count_valid_residues(df, drop_missing)
    embed_dim = config.TOTAL_EMBED_DIM
    print(f"Total residues: {total_residues:,}, embedding dim: {embed_dim}")

    if use_memmap:
        # Use memory-mapped files to avoid RAM exhaustion
        import hashlib
        import tempfile

        memmap_dir = Path(tempfile.gettempdir()) / "adopt2g_memmap"
        memmap_dir.mkdir(exist_ok=True)

        # Use fast hash based on index and size only (avoid expensive full DataFrame hash)
        hash_input = f"{sorted(df.index.tolist())}_{total_residues}_{embed_dim}"
        df_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        emb_path = memmap_dir / f"embeddings_{df_hash}.npy"
        gs_path = memmap_dir / f"gscores_{df_hash}.npy"

        all_embeddings = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(total_residues, embed_dim))
        all_gscores = np.memmap(gs_path, dtype=np.float32, mode="w+", shape=(total_residues,))
    else:
        all_embeddings = np.empty((total_residues, embed_dim), dtype=np.float32)
        all_gscores = np.empty(total_residues, dtype=np.float32)

    idx = 0
    for label in tqdm(df.index, desc="Preparing residue embeddings"):
        esm_path = esm_cache_dir / f"{label}.pt"
        if not esm_path.exists():
            raise FileNotFoundError(
                f"ESM embedding for {label} not found at {esm_path}. Run embedding extraction first.",
            )

        esm_emb = load_esm_embedding(esm_path).numpy()
        sequence = df.loc[label, "sequence"]
        # Skip CLS token (index 0) and take only sequence residues
        esm_emb = esm_emb[1 : 1 + len(sequence)]

        gscores = df.loc[label, "gscores"]

        # Get valid indices (non-NaN g-scores or all if not dropping)
        if drop_missing:
            valid_idx = np.where(~np.isnan(gscores))[0]
        else:
            valid_idx = np.arange(len(gscores))

        n_valid = len(valid_idx)
        if n_valid == 0:
            continue

        # Vectorized assignment: copy ESM embeddings for valid residues
        all_embeddings[idx : idx + n_valid, : config.ESM_EMBED_DIM] = esm_emb[valid_idx]
        # Broadcast ST embedding to all valid residues
        all_embeddings[idx : idx + n_valid, config.ESM_EMBED_DIM :] = st_repr[label]
        # Copy g-scores
        all_gscores[idx : idx + n_valid] = gscores[valid_idx]
        idx += n_valid

    # Flush memmap to disk
    if use_memmap:
        all_embeddings.flush()
        all_gscores.flush()

    return all_embeddings, all_gscores


def get_embeddings_for_sequences(
    sequences: list[str],
    tokenizer: AutoTokenizer,
    esm_model: AutoModel,
    st_model: SentenceTransformer,
) -> list[np.ndarray]:
    """
    Compute combined embeddings for a batch of sequences (for inference).

    Returns a list of arrays, one per sequence, shape (seq_len, 1600).

    Note: Sequences longer than MAX_SEQUENCE_LEN will be truncated.
    Use EnsembleDisorderPredictor for proper handling of long sequences with overlap.
    """
    device = next(esm_model.parameters()).device

    # Tokenize and get ESM embeddings (with truncation for safety)
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_SEQUENCE_LEN + 2,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        esm_outputs = esm_model(**inputs)

    # Get ST embeddings
    st_embeddings = st_model.encode(sequences)

    # Combine for each sequence
    results = []
    for i, seq in enumerate(sequences):
        # ESM: skip CLS token (index 0), take seq_len tokens
        esm_emb = esm_outputs.last_hidden_state[i, 1 : 1 + len(seq), :].cpu().numpy()
        # Tile ST embedding for all residues
        st_emb = np.tile(st_embeddings[i], (len(seq), 1))
        # Concatenate
        results.append(np.concatenate([esm_emb, st_emb], axis=1))

    return results
