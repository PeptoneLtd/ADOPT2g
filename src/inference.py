"""Inference logic for disorder prediction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from . import config
from .ensemble import EnsembleManifest
from .models import DisorderMLP

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results."""

    gscores: list[float]  # Mean prediction per residue
    gscores_std: list[float]  # Std across members per residue
    mean_gscore: float  # Mean of gscores (nanmean)
    mean_gscore_std: float  # Std across members of per-member mean_gscore


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


class EnsembleDisorderPredictor:
    """
    Ensemble predictor that combines predictions from multiple ST+MLP members.

    Shares a single ESM model across all members for efficiency.
    Returns mean predictions and per-residue uncertainty estimates.
    """

    def __init__(
        self,
        esm_tokenizer: AutoTokenizer,
        esm_model: AutoModel,
        st_models: list[SentenceTransformer],
        mlps: list[DisorderMLP],
        device: str | None = None,
    ):
        self.tokenizer = esm_tokenizer
        self.esm_model = esm_model
        self.st_models = st_models
        self.mlps = mlps
        self.ensemble_size = len(st_models)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.esm_model = self.esm_model.to(device)
        for mlp in self.mlps:
            mlp.to(device)
            mlp.eval()

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

    def _get_esm_embeddings(self, sequences: list[str]) -> list[np.ndarray]:
        """Compute ESM embeddings for a batch of sequences (shared across members)."""
        results = []

        for batch in self._chunk_sequences(sequences):
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                esm_out = self.esm_model(**inputs)

            for i, seq in enumerate(batch):
                # Skip CLS token, take len(seq) residues
                esm_emb = esm_out.last_hidden_state[i, 1 : 1 + len(seq), :].cpu().numpy()
                results.append(esm_emb.astype(np.float32))

        return results

    def _predict_member_scores(
        self,
        member_idx: int,
        esm_embeddings: list[np.ndarray],
        sequences: list[str],
    ) -> list[np.ndarray]:
        """Predict scores for one member given precomputed ESM embeddings."""
        st_model = self.st_models[member_idx]
        mlp = self.mlps[member_idx]

        # Get ST embeddings for all sequences
        st_embs = st_model.encode(sequences)

        results = []
        for i, (esm_emb, seq) in enumerate(zip(esm_embeddings, sequences)):
            # Tile ST embedding for all residues
            st_emb = np.tile(st_embs[i], (len(seq), 1))
            # Concatenate
            combined = np.concatenate([esm_emb, st_emb], axis=1).astype(np.float32)

            # Run MLP
            x = torch.from_numpy(combined).to(self.device)
            with torch.no_grad():
                scores = mlp(x).cpu().numpy()
            results.append(scores)

        return results

    def _predict_short_sequences(self, sequences: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Predict scores for sequences within max length.

        Returns (mean_scores, std_scores) per sequence.
        """
        # Compute ESM embeddings once (shared)
        esm_embeddings = self._get_esm_embeddings(sequences)

        # Get predictions from each member
        all_member_scores = []
        for member_idx in range(self.ensemble_size):
            member_scores = self._predict_member_scores(member_idx, esm_embeddings, sequences)
            all_member_scores.append(member_scores)

        # Aggregate: compute mean and std across members
        mean_scores = []
        std_scores = []

        for seq_idx in range(len(sequences)):
            # Stack member predictions for this sequence: (n_members, seq_len)
            stacked = np.stack([all_member_scores[m][seq_idx] for m in range(self.ensemble_size)])
            mean_scores.append(np.mean(stacked, axis=0))
            std_scores.append(np.std(stacked, axis=0))

        return mean_scores, std_scores

    def _predict_long_sequence(self, sequence: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict scores for a long sequence using overlapping chunks.

        Each member handles chunks independently, then we aggregate across members.
        """
        if len(sequence) <= self._max_seq_len:
            means, stds = self._predict_short_sequences([sequence])
            return means[0], stds[0]

        min_overlap = self._max_seq_len // 2
        num_chunks = int(np.ceil((len(sequence) - min_overlap) / (self._max_seq_len - min_overlap)))
        num_chunks = max(2, num_chunks)
        overlap = int(np.floor((self._max_seq_len * num_chunks - len(sequence)) / (num_chunks - 1)))

        # Split into chunks
        chunks = []
        for i in range(num_chunks):
            start = i * (self._max_seq_len - overlap)
            end = min(len(sequence), start + self._max_seq_len)
            chunks.append(sequence[start:end])

        # Get ESM embeddings for chunks (shared)
        esm_embeddings = self._get_esm_embeddings(chunks)

        # Get predictions from each member for all chunks
        all_member_preds = []  # List of (member, seq_len) arrays

        for member_idx in range(self.ensemble_size):
            # Predict each chunk with this member
            chunk_scores = self._predict_member_scores(member_idx, esm_embeddings, chunks)

            # Combine chunks into full sequence prediction for this member
            preds = np.zeros(len(sequence), dtype=float)
            counts = np.zeros(len(sequence), dtype=int)

            for i, scores in enumerate(chunk_scores):
                start = i * (self._max_seq_len - overlap)
                end = min(len(sequence), start + self._max_seq_len)
                preds[start:end] += scores
                counts[start:end] += 1

            all_member_preds.append(preds / counts)

        # Aggregate across members
        stacked = np.stack(all_member_preds)  # (n_members, seq_len)
        mean_scores = np.mean(stacked, axis=0)
        std_scores = np.std(stacked, axis=0)

        return mean_scores, std_scores

    def predict(self, sequences: list[str]) -> list[EnsemblePrediction]:
        """
        Predict disorder scores with uncertainty for a list of sequences.

        Non-standard amino acids are replaced with 'X' for prediction,
        and their scores are set to NaN.

        Args:
            sequences: List of amino acid sequences

        Returns:
            List of EnsemblePrediction objects with mean, std, and mean_gscore_std
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
        mean_results: list[np.ndarray | None] = [None] * len(sequences)
        std_results: list[np.ndarray | None] = [None] * len(sequences)
        short_indices = []
        short_seqs = []

        for i, seq in enumerate(cleaned_sequences):
            if len(seq) <= self._max_seq_len:
                short_indices.append(i)
                short_seqs.append(seq)
            else:
                mean_results[i], std_results[i] = self._predict_long_sequence(seq)

        # Batch predict short sequences
        if short_seqs:
            short_means, short_stds = self._predict_short_sequences(short_seqs)
            for i, idx in enumerate(short_indices):
                mean_results[idx] = short_means[i]
                std_results[idx] = short_stds[i]

        # Apply NaN mask for non-standard amino acids
        for i, mask in enumerate(non_std_masks):
            if len(mask) > 0:
                mean_results[i][mask] = np.nan
                std_results[i][mask] = np.nan

        # Compute mean_gscore_std: std across members of each member's mean_gscore
        # We need to recompute per-member mean_gscore
        ensemble_predictions = []
        for i, seq in enumerate(cleaned_sequences):
            mean_scores = mean_results[i]
            std_scores = std_results[i]

            # Get per-member mean_gscore for this sequence
            member_mean_gscores = self._get_member_mean_gscores(seq, non_std_masks[i])
            mean_gscore_std = float(np.std(member_mean_gscores))

            ensemble_predictions.append(
                EnsemblePrediction(
                    gscores=mean_scores.tolist(),
                    gscores_std=std_scores.tolist(),
                    mean_gscore=float(np.nanmean(mean_scores)),
                    mean_gscore_std=mean_gscore_std,
                )
            )

        return ensemble_predictions

    def _get_member_mean_gscores(self, sequence: str, non_std_mask: np.ndarray) -> list[float]:
        """Compute mean_gscore for each member for a single sequence."""
        # For short sequences, we can reuse predictions
        # For simplicity, we'll recompute - this is called per-sequence anyway
        if len(sequence) <= self._max_seq_len:
            esm_embeddings = self._get_esm_embeddings([sequence])
            member_means = []
            for member_idx in range(self.ensemble_size):
                scores = self._predict_member_scores(member_idx, esm_embeddings, [sequence])[0]
                if len(non_std_mask) > 0:
                    scores[non_std_mask] = np.nan
                member_means.append(float(np.nanmean(scores)))
            return member_means
        else:
            # For long sequences, use chunk-based prediction per member
            min_overlap = self._max_seq_len // 2
            num_chunks = int(np.ceil((len(sequence) - min_overlap) / (self._max_seq_len - min_overlap)))
            num_chunks = max(2, num_chunks)
            overlap = int(np.floor((self._max_seq_len * num_chunks - len(sequence)) / (num_chunks - 1)))

            chunks = []
            for i in range(num_chunks):
                start = i * (self._max_seq_len - overlap)
                end = min(len(sequence), start + self._max_seq_len)
                chunks.append(sequence[start:end])

            esm_embeddings = self._get_esm_embeddings(chunks)

            member_means = []
            for member_idx in range(self.ensemble_size):
                chunk_scores = self._predict_member_scores(member_idx, esm_embeddings, chunks)

                preds = np.zeros(len(sequence), dtype=float)
                counts = np.zeros(len(sequence), dtype=int)

                for i, scores in enumerate(chunk_scores):
                    start = i * (self._max_seq_len - overlap)
                    end = min(len(sequence), start + self._max_seq_len)
                    preds[start:end] += scores
                    counts[start:end] += 1

                full_scores = preds / counts
                if len(non_std_mask) > 0:
                    full_scores[non_std_mask] = np.nan
                member_means.append(float(np.nanmean(full_scores)))

            return member_means


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


def load_ensemble_predictor(
    ensemble_id_or_manifest: str | Path,
    esm_model_name: str = config.ESM_MODEL_NAME,
    device: str | None = None,
) -> EnsembleDisorderPredictor:
    """
    Load an ensemble predictor from manifest.

    Args:
        ensemble_id_or_manifest: Either an ensemble ID (looks in default location)
            or a path to manifest.json
        esm_model_name: Name of ESM model on HuggingFace
        device: Device to use (defaults to CUDA if available)

    Returns:
        Ready-to-use EnsembleDisorderPredictor
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve manifest path
    manifest_path = Path(ensemble_id_or_manifest)
    if not manifest_path.exists():
        # Try as ensemble ID
        manifest_path = config.ENSEMBLES_PATH / ensemble_id_or_manifest / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Ensemble manifest not found: {manifest_path}. Train an ensemble first with run_pipeline.py"
        )

    print(f"Loading ensemble from {manifest_path}...")
    manifest = EnsembleManifest.load(manifest_path)
    print(f"  Ensemble ID: {manifest.ensemble_id}")
    print(f"  Ensemble size: {manifest.ensemble_size}")

    # Load shared ESM model
    print(f"  Loading ESM model to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
    esm_model = AutoModel.from_pretrained(esm_model_name, torch_dtype=torch.float16)

    # Load all ST models
    print(f"  Loading {manifest.ensemble_size} ST models...")
    st_models = []
    for member in manifest.members:
        st_model = SentenceTransformer(member.st_path)
        st_models.append(st_model)

    # Load all MLP heads
    print(f"  Loading {manifest.ensemble_size} MLP heads...")
    mlps = []
    for member in manifest.members:
        mlp = DisorderMLP()
        # Support both old (raw state_dict) and new (dict with state_dict) formats
        checkpoint = torch.load(member.mlp_path, weights_only=True, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            mlp.load_state_dict(checkpoint["state_dict"])
        else:
            mlp.load_state_dict(checkpoint)
        mlps.append(mlp)

    return EnsembleDisorderPredictor(
        esm_tokenizer=tokenizer,
        esm_model=esm_model,
        st_models=st_models,
        mlps=mlps,
        device=device,
    )
