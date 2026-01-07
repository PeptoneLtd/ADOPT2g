"""Inference logic for disorder prediction."""

from typing import Iterator

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from . import config
from .models import DisorderMLP

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


class DisorderPredictor:
    """
    Disorder predictor combining ESM residue embeddings with ST sequence embeddings.

    Handles:
    - Chunking for sequences longer than ESM max length
    - Non-standard amino acids (replaced with 'X', output set to NaN)
    - Sigmoid transform on output logits
    """

    def __init__(
        self,
        esm_tokenizer: AutoTokenizer,
        esm_model: AutoModel,
        st_model: SentenceTransformer,
        mlp: DisorderMLP,
        device: str | None = None,
    ):
        self.tokenizer = esm_tokenizer
        self.esm_model = esm_model
        self.st_model = st_model
        self.mlp = mlp

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.esm_model = self.esm_model.to(device)
        self.mlp = self.mlp.to(device)
        self.mlp.eval()

        self._max_seq_len = config.MAX_SEQUENCE_LEN

    def _chunk_sequences(self, sequences: list[str]) -> Iterator[list[str]]:
        """Yield batches of sequences that fit in memory."""
        current_batch = []
        current_len = 0

        for seq in sequences:
            seq_len = len(seq)
            if current_batch and current_len + seq_len >= self._max_seq_len:
                yield current_batch
                current_batch = [seq]
                current_len = seq_len
            else:
                current_batch.append(seq)
                current_len += seq_len

        if current_batch:
            yield current_batch

    def _get_embeddings(self, sequences: list[str]) -> list[np.ndarray]:
        """Compute combined embeddings for a batch of sequences."""
        results = []

        for batch in self._chunk_sequences(sequences):
            # ESM embeddings
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                esm_out = self.esm_model(**inputs)

            # ST embeddings
            st_embs = self.st_model.encode(batch)

            # Combine for each sequence
            for i, seq in enumerate(batch):
                # Skip CLS token, take len(seq) residues
                esm_emb = esm_out.last_hidden_state[i, 1 : 1 + len(seq), :].cpu().numpy()
                st_emb = np.tile(st_embs[i], (len(seq), 1))
                results.append(np.concatenate([esm_emb, st_emb], axis=1).astype(np.float32))

        return results

    def _predict_scores(self, embeddings: list[np.ndarray]) -> list[np.ndarray]:
        """Run MLP on embeddings (model outputs g-scores directly)."""
        results = []

        for emb in embeddings:
            x = torch.from_numpy(emb).to(self.device)
            with torch.no_grad():
                scores = self.mlp(x).cpu().numpy()
            results.append(scores)

        return results

    def _predict_short_sequences(self, sequences: list[str]) -> list[np.ndarray]:
        """Predict scores for sequences within max length."""
        embeddings = self._get_embeddings(sequences)
        return self._predict_scores(embeddings)

    def _predict_long_sequence(self, sequence: str) -> np.ndarray:
        """
        Predict scores for a long sequence using overlapping chunks.

        Splits into chunks of max_seq_len with 50% overlap, averages overlapping regions.
        """
        # Edge case: sequence fits in one chunk (shouldn't happen but handle gracefully)
        if len(sequence) <= self._max_seq_len:
            return self._predict_short_sequences([sequence])[0]

        min_overlap = self._max_seq_len // 2
        num_chunks = int(np.ceil((len(sequence) - min_overlap) / (self._max_seq_len - min_overlap)))
        
        # Ensure at least 2 chunks to avoid division by zero
        num_chunks = max(2, num_chunks)
        overlap = int(np.floor((self._max_seq_len * num_chunks - len(sequence)) / (num_chunks - 1)))

        # Split into chunks
        chunks = []
        for i in range(num_chunks):
            start = i * (self._max_seq_len - overlap)
            end = min(len(sequence), start + self._max_seq_len)
            chunks.append(sequence[start:end])

        # Predict scores for each chunk
        chunk_scores = self._predict_short_sequences(chunks)

        # Combine with averaging in overlap regions
        preds = np.zeros(len(sequence), dtype=float)
        counts = np.zeros(len(sequence), dtype=int)

        for i, scores in enumerate(chunk_scores):
            start = i * (self._max_seq_len - overlap)
            end = min(len(sequence), start + self._max_seq_len)
            preds[start:end] += scores
            counts[start:end] += 1

        return preds / counts

    def predict(self, sequences: list[str]) -> list[list[float]]:
        """
        Predict disorder scores for a list of sequences.

        Non-standard amino acids are replaced with 'X' for prediction,
        and their scores are set to NaN.

        Args:
            sequences: List of amino acid sequences

        Returns:
            List of score arrays, one per sequence
        """
        # Handle non-standard amino acids
        cleaned_sequences = []
        non_std_masks = []

        for seq in sequences:
            seq_arr = np.array(list(seq))
            mask = np.array([aa not in STANDARD_AMINO_ACIDS for aa in seq])
            non_std_masks.append(np.where(mask)[0])

            if mask.any():
                seq_arr[mask] = "X"
                cleaned_sequences.append("".join(seq_arr))
            else:
                cleaned_sequences.append(seq)

        # Separate short and long sequences
        results = [None] * len(sequences)
        short_indices = []
        short_seqs = []

        for i, seq in enumerate(cleaned_sequences):
            if len(seq) <= self._max_seq_len:
                short_indices.append(i)
                short_seqs.append(seq)
            else:
                # Predict long sequence directly
                results[i] = self._predict_long_sequence(seq)

        # Batch predict short sequences
        if short_seqs:
            short_scores = self._predict_short_sequences(short_seqs)
            for i, idx in enumerate(short_indices):
                results[idx] = short_scores[i]

        # Apply NaN mask for non-standard amino acids
        for i, mask in enumerate(non_std_masks):
            if len(mask) > 0:
                results[i][mask] = np.nan

        # Convert to lists
        return [scores.tolist() for scores in results]


def load_predictor(
    st_model_path: str,
    mlp_path: str,
    esm_model_name: str = config.ESM_MODEL_NAME,
    device: str | None = None,
) -> DisorderPredictor:
    """
    Load all components and create a DisorderPredictor.

    Args:
        st_model_path: Path to saved SentenceTransformer model
        mlp_path: Path to saved MLP state dict
        esm_model_name: Name of ESM model on HuggingFace
        device: Device to use (defaults to CUDA if available)

    Returns:
        Ready-to-use DisorderPredictor
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading models to {device}...")

    # Load ESM
    tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
    esm_model = AutoModel.from_pretrained(esm_model_name, torch_dtype=torch.float16)

    # Load ST
    st_model = SentenceTransformer(st_model_path)

    # Load MLP
    mlp = DisorderMLP()
    mlp.load_state_dict(torch.load(mlp_path, weights_only=True, map_location=device))

    return DisorderPredictor(
        esm_tokenizer=tokenizer,
        esm_model=esm_model,
        st_model=st_model,
        mlp=mlp,
        device=device,
    )
