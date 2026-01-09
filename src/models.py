"""Models, ensemble management, and inference for disorder prediction."""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from . import config

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


# =============================================================================
# MLP Architecture
# =============================================================================


class DisorderMLP(nn.Module):
    """
    Simple MLP for predicting disorder g-scores from concatenated embeddings.

    Input: ESM residue embedding (1280) + ST sequence embedding (320) = 1600
    Output: G-score in [0, 1]
    """

    def __init__(
        self,
        input_dim: int = config.TOTAL_EMBED_DIM,
        hidden_dims: list[int] | None = None,
        dropout: float = config.MLP_DROPOUT,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = config.MLP_HIDDEN_DIMS

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ],
            )
            prev_dim = hidden_dim

        # Output layer with sigmoid for g-score [0, 1]
        layers.extend(
            [
                nn.Linear(prev_dim, 1),
                nn.Sigmoid(),
            ],
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Embeddings of shape (batch_size, input_dim)

        Returns:
            G-scores of shape (batch_size,), values in [0, 1]
        """
        return self.net(x).squeeze(-1)


# =============================================================================
# Ensemble Manifest and Configuration
# =============================================================================


@dataclass
class MemberConfig:
    """Configuration for a single ensemble member."""

    member_id: int
    seed_global: int
    seed_st: int
    seed_mlp: int
    split_seed_st: int
    split_seed_mlp: int
    st_train_frac: float
    mlp_val_frac: float
    st_path: str  # Relative path from manifest directory
    mlp_path: str  # Relative path from manifest directory


@dataclass
class EnsembleManifest:
    """Manifest for an ensemble of disorder predictors."""

    ensemble_id: str
    ensemble_size: int
    base_seed: int
    members: list[MemberConfig] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    calibration_factor: float = 1.0

    def to_dict(self) -> dict:
        """Convert manifest to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EnsembleManifest":
        """Create manifest from dictionary."""
        members = [MemberConfig(**m) for m in d.pop("members", [])]
        return cls(members=members, **d)

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "EnsembleManifest":
        """Load manifest from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Path Management
# =============================================================================


def get_ensemble_dir(ensemble_id: str) -> Path:
    """Get the base directory for an ensemble."""
    return config.MODELS_PATH / ensemble_id


def get_member_dir(ensemble_id: str, member_id: int) -> Path:
    """Get the directory for a specific ensemble member."""
    return get_ensemble_dir(ensemble_id) / f"member_{member_id}"


def derive_member_seeds(base_seed: int, member_id: int) -> dict[str, int]:
    """
    Derive deterministic seeds for a member from base seed.

    Uses different multipliers to ensure seeds are independent.
    """
    return {
        "seed_global": base_seed + member_id * 1000,
        "seed_st": base_seed + member_id * 1000 + 100,
        "seed_mlp": base_seed + member_id * 1000 + 200,
        "split_seed_st": base_seed + member_id * 1000 + 300,
        "split_seed_mlp": base_seed + member_id * 1000 + 400,
    }


def create_config_snapshot() -> dict[str, Any]:
    """Create a snapshot of relevant config values."""
    return {
        "st_base_model": config.ST_BASE_MODEL,
        "st_num_iterations": config.ST_NUM_ITERATIONS,
        "st_batch_size": config.ST_BATCH_SIZE,
        "st_num_epochs": config.ST_NUM_EPOCHS,
        "st_warmup_ratio": config.ST_WARMUP_RATIO,
        "mlp_hidden_dims": config.MLP_HIDDEN_DIMS,
        "mlp_dropout": config.MLP_DROPOUT,
        "mlp_lr": config.MLP_LR,
        "mlp_weight_decay": config.MLP_WEIGHT_DECAY,
        "mlp_batch_size": config.MLP_BATCH_SIZE,
        "esm_model": config.ESM_MODEL_NAME,
        "esm_embed_dim": config.ESM_EMBED_DIM,
        "st_embed_dim": config.ST_EMBED_DIM,
        "rebalance_a": config.REBALANCE_A,
        "rebalance_b": config.REBALANCE_B,
    }


def generate_ensemble_id(base_seed: int, ensemble_size: int) -> str:
    """Generate a deterministic ensemble ID from base seed and size."""
    return f"seed{base_seed}_n{ensemble_size}"


def find_latest_ensemble() -> Path | None:
    """Find the most recently created ensemble manifest."""
    if not config.MODELS_PATH.exists():
        return None

    manifests = list(config.MODELS_PATH.glob("*/manifest.json"))
    if not manifests:
        return None

    # Sort by modification time, most recent first
    manifests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return manifests[0]


# =============================================================================
# Inference
# =============================================================================


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results."""

    gscores: list[float]  # Mean prediction per residue
    gscores_std: list[float]  # Std across members per residue
    mean_gscore: float  # Mean of gscores (nanmean)
    mean_gscore_std: float  # Std across members of per-member mean_gscore
    member_gscores: list[list[float]] | None = None  # Per-member gscores (optional)


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
        calibration_factor: float = 1.0,
    ):
        self.tokenizer = esm_tokenizer
        self.esm_model = esm_model
        self.st_models = st_models
        self.mlps = mlps
        self.ensemble_size = len(st_models)
        self.calibration_factor = calibration_factor

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

    def _predict_short_sequences(
        self, sequences: list[str],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[list[np.ndarray]]]:
        """
        Predict scores for sequences within max length.

        Returns:
            mean_scores: List of mean predictions per sequence
            std_scores: List of std predictions per sequence
            all_member_scores: List of [member_0_scores, member_1_scores, ...] per sequence
        """
        # Compute ESM embeddings once (shared)
        esm_embeddings = self._get_esm_embeddings(sequences)

        # Get predictions from each member: all_member_scores[member_idx][seq_idx] = scores
        all_member_scores_by_member = []
        for member_idx in range(self.ensemble_size):
            member_scores = self._predict_member_scores(member_idx, esm_embeddings, sequences)
            all_member_scores_by_member.append(member_scores)

        # Reorganize to per-sequence and aggregate
        mean_scores = []
        std_scores = []
        all_member_scores = []  # per-sequence: list of member score arrays

        for seq_idx in range(len(sequences)):
            # Collect member predictions for this sequence
            seq_member_scores = [all_member_scores_by_member[m][seq_idx] for m in range(self.ensemble_size)]
            all_member_scores.append(seq_member_scores)

            # Stack and compute stats: (n_members, seq_len)
            stacked = np.stack(seq_member_scores)
            mean_scores.append(np.mean(stacked, axis=0))
            std_scores.append(np.std(stacked, axis=0))

        return mean_scores, std_scores, all_member_scores

    def _predict_long_sequence(self, sequence: str) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """
        Predict scores for a long sequence using overlapping chunks.

        Each member handles chunks independently, then we aggregate across members.

        Returns:
            mean_scores: Mean prediction per residue
            std_scores: Std prediction per residue
            member_scores: List of per-member full-sequence predictions
        """
        if len(sequence) <= self._max_seq_len:
            means, stds, all_member = self._predict_short_sequences([sequence])
            return means[0], stds[0], all_member[0]

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
        all_member_preds = []  # List of full-sequence arrays per member

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

        return mean_scores, std_scores, all_member_preds

    def predict(self, sequences: list[str], include_member_gscores: bool = False) -> list[EnsemblePrediction]:
        """
        Predict disorder scores with uncertainty for a list of sequences.

        Non-standard amino acids are replaced with 'X' for prediction,
        and their scores are set to NaN.

        Args:
            sequences: List of amino acid sequences
            include_member_gscores: If True, include per-member gscores in the result

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
        member_scores_results: list[list[np.ndarray] | None] = [None] * len(sequences)
        short_indices = []
        short_seqs = []

        for i, seq in enumerate(cleaned_sequences):
            if len(seq) <= self._max_seq_len:
                short_indices.append(i)
                short_seqs.append(seq)
            else:
                mean_results[i], std_results[i], member_scores_results[i] = self._predict_long_sequence(seq)

        # Batch predict short sequences
        if short_seqs:
            short_means, short_stds, short_member_scores = self._predict_short_sequences(short_seqs)
            for i, idx in enumerate(short_indices):
                mean_results[idx] = short_means[i]
                std_results[idx] = short_stds[i]
                member_scores_results[idx] = short_member_scores[i]

        # Apply NaN mask and compute final predictions
        ensemble_predictions = []
        for i in range(len(sequences)):
            mask = non_std_masks[i]
            mean_scores = mean_results[i]
            std_scores = std_results[i]
            member_scores = member_scores_results[i]

            # Apply NaN mask
            if len(mask) > 0:
                mean_scores[mask] = np.nan
                std_scores[mask] = np.nan
                for m_scores in member_scores:
                    m_scores[mask] = np.nan

            # Compute uncertainty metrics from already-computed member scores
            member_mean_gscores = [float(np.nanmean(m_scores)) for m_scores in member_scores]
            mean_gscore_std = float(np.std(member_mean_gscores))

            # Apply calibration factor to uncertainty estimates
            calibrated_std_scores = std_scores * self.calibration_factor
            calibrated_mean_gscore_std = mean_gscore_std * self.calibration_factor

            ensemble_predictions.append(
                EnsemblePrediction(
                    gscores=mean_scores.tolist(),
                    gscores_std=calibrated_std_scores.tolist(),
                    mean_gscore=float(np.nanmean(mean_scores)),
                    mean_gscore_std=calibrated_mean_gscore_std,
                    member_gscores=[m.tolist() for m in member_scores] if include_member_gscores else None,
                ),
            )

        return ensemble_predictions


def load_ensemble_predictor(
    ensemble_id_or_manifest: str | Path,
    esm_model_name: str = config.ESM_MODEL_NAME,
    device: str | None = None,
    calibration_factor: float | None = None,
) -> EnsembleDisorderPredictor:
    """
    Load an ensemble predictor from manifest.

    Args:
        ensemble_id_or_manifest: Either an ensemble ID (looks in default location)
            or a path to manifest.json
        esm_model_name: Name of ESM model on HuggingFace
        device: Device to use (defaults to CUDA if available)
        calibration_factor: Override the calibration factor from manifest.
            If None, uses the value from manifest. Set to 1.0 to disable calibration.

    Returns:
        Ready-to-use EnsembleDisorderPredictor
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve manifest path
    manifest_path = Path(ensemble_id_or_manifest)
    if not manifest_path.exists():
        # Try as ensemble ID
        manifest_path = config.MODELS_PATH / ensemble_id_or_manifest / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Ensemble manifest not found: {manifest_path}. Train an ensemble first with run_pipeline.py",
        )

    print(f"Loading ensemble from {manifest_path}...")
    manifest = EnsembleManifest.load(manifest_path)
    manifest_dir = manifest_path.parent
    print(f"  Ensemble ID: {manifest.ensemble_id}")
    print(f"  Ensemble size: {manifest.ensemble_size}")

    # Determine calibration factor
    if calibration_factor is None:
        calibration_factor = manifest.calibration_factor
    print(f"  Calibration factor: {calibration_factor:.4f}")

    # Load shared ESM model
    print(f"  Loading ESM model to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
    esm_model = AutoModel.from_pretrained(esm_model_name, dtype=torch.float16)

    # Load all ST models (resolve relative paths from manifest directory)
    print(f"  Loading {manifest.ensemble_size} ST models...")
    st_models = []
    for member in manifest.members:
        st_path = manifest_dir / member.st_path
        st_model = SentenceTransformer(str(st_path))
        st_models.append(st_model)

    # Load all MLP heads (resolve relative paths from manifest directory)
    print(f"  Loading {manifest.ensemble_size} MLP heads...")
    mlps = []
    for member in manifest.members:
        mlp_path = manifest_dir / member.mlp_path
        mlp = DisorderMLP()
        # Support both old (raw state_dict) and new (dict with state_dict) formats
        checkpoint = torch.load(mlp_path, weights_only=True, map_location=device)
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
        calibration_factor=calibration_factor,
    )
