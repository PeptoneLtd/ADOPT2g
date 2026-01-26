"""Ensemble inference and prediction for disorder prediction."""

import gc
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, logit
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from . import config
from .embeddings import load_esm_model
from .ensemble import EnsembleManifest
from .mlp import DisorderMLP

logger = logging.getLogger(__name__)

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def estimate_max_residues_per_batch(usage_fraction: float = 0.75) -> int:
    """
    Estimate max residues per GPU batch based on available memory.

    Args:
        usage_fraction: Fraction of available GPU memory to use (default: 0.75)

    Returns:
        Estimated maximum number of residues that can fit in a batch
    """
    if torch.cuda.is_available():
        free_mem_gb = torch.cuda.mem_get_info()[0] / 1e9
        # Reserve ~3GB for models, use ~usage_fraction of remaining for safety
        usable_gb = max(0.5, (free_mem_gb - 3.0) * usage_fraction)
        # Empirical: ~5KB per residue for ESM2-650M fp16 forward pass
        return int(usable_gb * 1e9 / 5_000)
    return 10_000  # default for CPU


# =============================================================================
# Inference
# =============================================================================


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results."""

    gscores: list[float]  # Mean prediction per residue (sigmoid of mean logit)
    gscores_lower: list[float]  # Lower bound per residue (from logit space uncertainty)
    gscores_upper: list[float]  # Upper bound per residue (from logit space uncertainty)
    mean_gscore: float  # Mean of gscores (nanmean)
    mean_gscore_lower: float  # Lower bound for mean_gscore
    mean_gscore_upper: float  # Upper bound for mean_gscore
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
        uncertainty_calibration_factor: float = 1.0,
        temperature_scaling_factor: float = 1.0,
    ) -> None:
        self.tokenizer = esm_tokenizer
        self.esm_model = esm_model
        self.st_models = st_models
        self.mlps = mlps
        self.ensemble_size = len(st_models)
        self.uncertainty_calibration_factor = uncertainty_calibration_factor
        self.temperature_scaling_factor = temperature_scaling_factor

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
        use_amp = self.device == "cuda"

        for batch in self._chunk_sequences(sequences):
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
                esm_out = self.esm_model(**inputs)

            for i, seq in enumerate(batch):
                # Skip CLS token, take len(seq) residues
                esm_emb = esm_out.last_hidden_state[i, 1 : 1 + len(seq), :].cpu().numpy()
                results.append(esm_emb.astype(np.float32))

        return results

    def _predict_member_logits(
        self,
        member_idx: int,
        esm_embeddings: list[np.ndarray],
        sequences: list[str],
    ) -> list[np.ndarray]:
        """Predict logits for one member given precomputed ESM embeddings."""
        st_model = self.st_models[member_idx]
        mlp = self.mlps[member_idx]
        use_amp = self.device == "cuda"

        # Get ST embeddings for all sequences
        st_embs = st_model.encode(sequences, show_progress_bar=False)

        results = []
        for i, (esm_emb, seq) in enumerate(zip(esm_embeddings, sequences, strict=True)):
            # Tile ST embedding for all residues
            st_emb = np.tile(st_embs[i], (len(seq), 1))
            # Concatenate
            combined = np.concatenate([esm_emb, st_emb], axis=1).astype(np.float32)

            # Run MLP with mixed precision, get logits (before sigmoid)
            x = torch.from_numpy(combined).to(self.device)
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
                logits = mlp.forward_logits(x).cpu().numpy()
            results.append(logits)

        return results

    def _predict_short_sequences(
        self,
        sequences: list[str],
        on_esm_start: Callable[[], None] | None = None,
        on_esm_done: Callable[[], None] | None = None,
        on_member_done: Callable[[int], None] | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[list[np.ndarray]]]:
        """
        Predict logits for sequences within max length.

        Args:
            sequences: List of amino acid sequences
            on_esm_start: Callback called before ESM embeddings computation
            on_esm_done: Callback called after ESM embeddings are computed
            on_member_done: Callback called after each ensemble member is processed (receives member_idx)

        Returns:
            mean_logits: List of mean logits per sequence
            std_logits: List of std logits per sequence
            all_member_logits: List of [member_0_logits, member_1_logits, ...] per sequence
        """
        # Compute ESM embeddings once (shared)
        if on_esm_start:
            on_esm_start()
        esm_embeddings = self._get_esm_embeddings(sequences)
        if on_esm_done:
            on_esm_done()

        # Get predictions from each member: all_member_logits[member_idx][seq_idx] = logits
        all_member_logits_by_member = []
        for member_idx in range(self.ensemble_size):
            member_logits = self._predict_member_logits(member_idx, esm_embeddings, sequences)
            all_member_logits_by_member.append(member_logits)
            if on_member_done:
                on_member_done(member_idx)

        # Reorganize to per-sequence and aggregate
        mean_logits = []
        std_logits = []
        all_member_logits = []  # per-sequence: list of member logit arrays

        for seq_idx in range(len(sequences)):
            # Collect member predictions for this sequence
            seq_member_logits = [all_member_logits_by_member[m][seq_idx] for m in range(self.ensemble_size)]
            all_member_logits.append(seq_member_logits)

            # Stack and compute stats in logit space: (n_members, seq_len)
            stacked = np.stack(seq_member_logits)
            mean_logits.append(np.mean(stacked, axis=0))
            std_logits.append(np.std(stacked, axis=0))

        return mean_logits, std_logits, all_member_logits

    def _predict_long_sequence(
        self,
        sequence: str,
        on_esm_start: Callable[[], None] | None = None,
        on_esm_done: Callable[[], None] | None = None,
        on_member_done: Callable[[int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """
        Predict logits for a long sequence using overlapping chunks.

        Each member handles chunks independently, then we aggregate across members.

        Args:
            sequence: Amino acid sequence longer than max_seq_len
            on_esm_start: Callback called before ESM embeddings computation
            on_esm_done: Callback called after ESM embeddings are computed
            on_member_done: Callback called after each ensemble member is processed (receives member_idx)

        Returns:
            mean_logits: Mean logits per residue
            std_logits: Std logits per residue
            member_logits: List of per-member full-sequence logits
        """
        seq_len = len(sequence)

        if seq_len <= self._max_seq_len:
            means, stds, all_member = self._predict_short_sequences(
                [sequence],
                on_esm_start=on_esm_start,
                on_esm_done=on_esm_done,
                on_member_done=on_member_done,
            )
            return means[0], stds[0], all_member[0]

        min_overlap = self._max_seq_len // 2
        num_chunks = int(np.ceil((seq_len - min_overlap) / (self._max_seq_len - min_overlap)))
        num_chunks = max(2, num_chunks)
        overlap = int(np.floor((self._max_seq_len * num_chunks - seq_len) / (num_chunks - 1)))

        # Split into chunks
        chunks = []
        for i in range(num_chunks):
            start = i * (self._max_seq_len - overlap)
            end = min(seq_len, start + self._max_seq_len)
            chunks.append(sequence[start:end])

        # Get ESM embeddings for chunks (shared)
        if on_esm_start:
            on_esm_start()
        esm_embeddings = self._get_esm_embeddings(chunks)
        if on_esm_done:
            on_esm_done()

        # Get logits from each member for all chunks
        all_member_logits = []  # List of full-sequence arrays per member

        for member_idx in range(self.ensemble_size):
            # Predict each chunk with this member
            chunk_logits = self._predict_member_logits(member_idx, esm_embeddings, chunks)

            # Combine chunks into full sequence logits for this member
            logits = np.zeros(seq_len, dtype=float)
            counts = np.zeros(seq_len, dtype=int)

            for i, chunk_log in enumerate(chunk_logits):
                start = i * (self._max_seq_len - overlap)
                end = min(seq_len, start + self._max_seq_len)
                logits[start:end] += chunk_log
                counts[start:end] += 1

            all_member_logits.append(logits / counts)

            if on_member_done:
                on_member_done(member_idx)

        # Aggregate across members in logit space
        stacked = np.stack(all_member_logits)  # (n_members, seq_len)
        mean_logits = np.mean(stacked, axis=0)
        std_logits = np.std(stacked, axis=0)

        return mean_logits, std_logits, all_member_logits

    def _predict_sequences(
        self,
        sequences: list[str],
        include_member_gscores: bool = False,
        on_esm_start: Callable[[], None] | None = None,
        on_esm_done: Callable[[], None] | None = None,
        on_member_done: Callable[[int], None] | None = None,
    ) -> list[EnsemblePrediction]:
        """
        Internal method to predict disorder scores for a list of sequences.

        Does not perform memory-aware batching. Use `predict()` for the public API.

        Predictions are computed in logit space (before sigmoid), where mean and std
        are calculated. Calibration is applied in logit space, then sigmoid is applied
        to get final scores and asymmetric confidence bounds.

        Non-standard amino acids are replaced with 'X' for prediction,
        and their scores are set to NaN.

        Args:
            sequences: List of amino acid sequences
            include_member_gscores: If True, include per-member gscores in the result
            on_esm_start: Callback called before ESM embeddings computation
            on_esm_done: Callback called after ESM embeddings are computed
            on_member_done: Callback called after each ensemble member is processed (receives member_idx)
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
        mean_logit_results: list[np.ndarray | None] = [None] * len(sequences)
        std_logit_results: list[np.ndarray | None] = [None] * len(sequences)
        member_logits_results: list[list[np.ndarray] | None] = [None] * len(sequences)
        short_indices = []
        short_seqs = []

        for i, seq in enumerate(cleaned_sequences):
            if len(seq) <= self._max_seq_len:
                short_indices.append(i)
                short_seqs.append(seq)
            else:
                mean_logit_results[i], std_logit_results[i], member_logits_results[i] = self._predict_long_sequence(
                    seq,
                    on_esm_start=on_esm_start,
                    on_esm_done=on_esm_done,
                    on_member_done=on_member_done,
                )

        # Batch predict short sequences
        if short_seqs:
            short_means, short_stds, short_member_logits = self._predict_short_sequences(
                short_seqs,
                on_esm_start=on_esm_start,
                on_esm_done=on_esm_done,
                on_member_done=on_member_done,
            )
            for i, idx in enumerate(short_indices):
                mean_logit_results[idx] = short_means[i]
                std_logit_results[idx] = short_stds[i]
                member_logits_results[idx] = short_member_logits[i]

        # Apply NaN mask and compute final predictions
        ensemble_predictions = []
        for i, seq in enumerate(sequences):
            mask = non_std_masks[i]
            mean_logits = mean_logit_results[i]
            std_logits = std_logit_results[i]
            member_logits = member_logits_results[i]

            # Apply NaN mask in logit space
            if len(mask) > 0:
                mean_logits[mask] = np.nan
                std_logits[mask] = np.nan
                for m_logits in member_logits:
                    m_logits[mask] = np.nan

            # Apply temperature scaling to logits
            T = self.temperature_scaling_factor
            scaled_mean_logits = mean_logits / T
            scaled_std_logits = std_logits / T

            # Apply uncertainty calibration factor in logit space
            calibrated_std_logits = scaled_std_logits * self.uncertainty_calibration_factor

            # Compute per-residue gscores and bounds via sigmoid
            gscores = expit(scaled_mean_logits)
            gscores_lower = expit(scaled_mean_logits - calibrated_std_logits)
            gscores_upper = expit(scaled_mean_logits + calibrated_std_logits)

            # Compute sequence-level mean_gscore and bounds
            # Each member's mean logit (across residues), then aggregate
            member_mean_logits = np.array([float(np.nanmean(m_logits)) for m_logits in member_logits])
            mean_of_mean_logits = np.mean(member_mean_logits) / T
            std_of_mean_logits = np.std(member_mean_logits) / T
            calibrated_std_of_mean_logits = std_of_mean_logits * self.uncertainty_calibration_factor

            mean_gscore = float(expit(np.array([mean_of_mean_logits]))[0])
            mean_gscore_lower = float(expit(np.array([mean_of_mean_logits - calibrated_std_of_mean_logits]))[0])
            mean_gscore_upper = float(expit(np.array([mean_of_mean_logits + calibrated_std_of_mean_logits]))[0])

            # Convert member logits to gscores if requested
            member_gscores = None
            if include_member_gscores:
                member_gscores = [expit(m_logits).tolist() for m_logits in member_logits]

            ensemble_predictions.append(
                EnsemblePrediction(
                    gscores=gscores.tolist(),
                    gscores_lower=gscores_lower.tolist(),
                    gscores_upper=gscores_upper.tolist(),
                    mean_gscore=mean_gscore,
                    mean_gscore_lower=mean_gscore_lower,
                    mean_gscore_upper=mean_gscore_upper,
                    member_gscores=member_gscores,
                ),
            )

        return ensemble_predictions

    def predict(
        self,
        sequences: list[str],
        include_member_gscores: bool = False,
        max_residues_per_batch: int | None = None,
        on_esm_start: Callable[[], None] | None = None,
        on_esm_done: Callable[[], None] | None = None,
        on_member_done: Callable[[int], None] | None = None,
    ) -> list[EnsemblePrediction]:
        """
        Predict disorder scores with uncertainty for a list of sequences.

        Automatically batches sequences to fit in GPU memory.

        Predictions are computed in logit space (before sigmoid), where mean and std
        are calculated. Calibration is applied in logit space, then sigmoid is applied
        to get final scores and asymmetric confidence bounds.

        Non-standard amino acids are replaced with 'X' for prediction,
        and their scores are set to NaN.

        Args:
            sequences: List of amino acid sequences
            include_member_gscores: If True, include per-member gscores in the result
            max_residues_per_batch: Maximum residues per batch (auto-detected if None)
            on_esm_start: Callback called before ESM embeddings computation
            on_esm_done: Callback called after ESM embeddings are computed
            on_member_done: Callback called after each ensemble member is processed (receives member_idx)

        Returns:
            List of EnsemblePrediction objects with mean, lower/upper bounds
        """
        if max_residues_per_batch is None:
            max_residues_per_batch = estimate_max_residues_per_batch()

        all_predictions: list[EnsemblePrediction] = []
        batch: list[str] = []
        batch_residues = 0

        if isinstance(sequences, str):
            sequences = [sequences]

        for seq in sequences:
            seq_len = len(seq)

            if batch and batch_residues + seq_len > max_residues_per_batch:
                # Process current batch
                all_predictions.extend(
                    self._predict_sequences(
                        batch,
                        include_member_gscores=include_member_gscores,
                        on_esm_start=on_esm_start,
                        on_esm_done=on_esm_done,
                        on_member_done=on_member_done,
                    ),
                )
                batch = []
                batch_residues = 0

            batch.append(seq)
            batch_residues += seq_len

        # Process final batch
        if batch:
            all_predictions.extend(
                self._predict_sequences(
                    batch,
                    include_member_gscores=include_member_gscores,
                    on_esm_start=on_esm_start,
                    on_esm_done=on_esm_done,
                    on_member_done=on_member_done,
                ),
            )

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return all_predictions

    def compute_calibration_factors(
        self,
        sequences: list[str],
        gscores: list[np.ndarray],
    ) -> tuple[float, float]:
        """
        Compute calibration factors for uncertainty estimates and temperature scaling.

        Computes two factors:
        1. Temperature scaling factor (T): Learned by minimizing Bernoulli NLL with soft labels.
           Logits are divided by T before applying sigmoid.
        2. Uncertainty calibration factor: Standard deviation of normalized residuals
           computed in logit space after temperature scaling:
           uncertainty_factor = std((logit(true) - mean_logit/T) / (std_logit/T))

        Args:
            sequences: List of amino acid sequences
            gscores: List of numpy arrays with ground truth g-scores per residue

        Returns:
            Tuple of (uncertainty_calibration_factor, temperature_scaling_factor)
        """
        assert len(sequences) == len(gscores), f"Got {len(sequences)} sequences but {len(gscores)} gscores"

        # Ensure calibration factors are 1.0 for calculation
        old_uncertainty_factor = self.uncertainty_calibration_factor
        old_temperature_factor = self.temperature_scaling_factor
        if old_uncertainty_factor != 1.0 or old_temperature_factor != 1.0:
            logger.warning(
                f"Computing calibration factors while current factors are "
                f"uncertainty={old_uncertainty_factor:.4f}, temperature={old_temperature_factor:.4f}. "
                "Temporarily setting both to 1.0 for calculation.",
            )
            self.uncertainty_calibration_factor = 1.0
            self.temperature_scaling_factor = 1.0

        try:
            # Get predictions using current calibration settings (both = 1.0)
            predictions = self.predict(sequences)

            # Collect all residue-level data for temperature scaling
            all_mean_logits = []
            all_true_gscores = []
            all_std_logits = []

            for true_gscore, pred in zip(gscores, predictions, strict=True):
                # Since calibration=1.0: gscores = sigmoid(mean_logit)
                #                        gscores_lower = sigmoid(mean_logit - std_logit)
                mean_logit = logit(pred.gscores)
                std_logit = mean_logit - logit(pred.gscores_lower)

                # Valid mask for temperature scaling (need valid true gscores in [0,1])
                valid_mask = (
                    np.isfinite(mean_logit)
                    & np.isfinite(std_logit)
                    & (std_logit > 1e-8)
                    & np.isfinite(true_gscore)
                    & (true_gscore >= 0.0)
                    & (true_gscore <= 1.0)
                )
                if valid_mask.sum() > 0:
                    all_mean_logits.extend(mean_logit[valid_mask])
                    all_true_gscores.extend(true_gscore[valid_mask])
                    all_std_logits.extend(std_logit[valid_mask])

            all_mean_logits = np.array(all_mean_logits)
            all_true_gscores = np.array(all_true_gscores)
            all_std_logits = np.array(all_std_logits)

            # Fit temperature scaling by minimizing Bernoulli NLL
            # BCE(sigmoid(logits/T), y) with soft labels y
            temperature_scaling_factor = self._fit_temperature_scaling(all_mean_logits, all_true_gscores)

            # Compute uncertainty calibration factor after temperature scaling
            true_logits = logit(all_true_gscores)
            mask = np.isfinite(true_logits)  # ingore gscores equal to 0 or 1
            true_logits = true_logits[mask]
            scaled_mean_logits = all_mean_logits[mask] / temperature_scaling_factor
            scaled_std_logits = all_std_logits[mask] / temperature_scaling_factor

            residuals_normalized = (true_logits - scaled_mean_logits) / scaled_std_logits
            uncertainty_calibration_factor = float(np.std(residuals_normalized))

        finally:
            self.uncertainty_calibration_factor = old_uncertainty_factor
            self.temperature_scaling_factor = old_temperature_factor

        return temperature_scaling_factor, uncertainty_calibration_factor

    def _fit_temperature_scaling(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        lr: float = 1,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> float:
        """
        Fit temperature scaling parameter by minimizing Bernoulli NLL with soft labels.

        Uses quasi-Newton (LBFGS) on log(T) to ensure T > 0.

        Args:
            logits: Raw model logits
            targets: Soft labels (ground truth gscores in [0, 1])
            lr: Learning rate (step size, 1.0 is standard for LBFGS with line search)
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Optimal temperature T
        """
        if len(logits) == 0:
            logger.warning("No valid data for temperature scaling. Returning T=1.0")
            return 1.0

        logits_t = torch.tensor(logits, dtype=torch.float32)
        targets_t = torch.tensor(targets, dtype=torch.float32)

        # Parameterize as log(T) to ensure T > 0
        log_T = torch.tensor(0.0, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [log_T],
            lr=lr,
            max_iter=max_iter,
            tolerance_grad=tol,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            T = torch.exp(log_T)
            scaled_logits = logits_t / T
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                scaled_logits,
                targets_t,
                reduction="mean",
            )
            if torch.isnan(loss):
                return torch.tensor(float("inf"))
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception as e:
            logger.warning(
                f"Temperature scaling optimization failed: {e}. Falling back to T=1.0",
            )
            return 1.0

        temperature = float(torch.exp(log_T).item())
        if np.isnan(temperature):
            logger.warning("Temperature scaling resulted in NaN. Falling back to 1.0.")
            return 1.0

        logger.info(f"  Temperature scaling fitted: T = {temperature:.4f}")
        return temperature


def load_ensemble_predictor(
    ensemble_id_or_manifest: str | Path,
    device: str | None = None,
    use_calibration: bool = True,
) -> EnsembleDisorderPredictor:
    """
    Load an ensemble predictor from manifest.

    Args:
        ensemble_id_or_manifest: Either an ensemble ID (looks in default location)
            or a path to manifest.json
        device: Device to use (defaults to CUDA if available)
        use_calibration: Whether to use calibration factors from manifest.
            If False, both uncertainty and temperature scaling factors are set to 1.0.

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

    logger.info(f"Loading ensemble from {manifest_path}...")
    manifest = EnsembleManifest.load(manifest_path)
    manifest_dir = manifest_path.parent
    logger.info(f"  Ensemble ID: {manifest.ensemble_id}")
    logger.info(f"  Ensemble size: {manifest.ensemble_size}")

    config_snapshot = manifest.config_snapshot or {}
    esm_model_name = config_snapshot.get("esm_model", config.ESM_MODEL_NAME)

    # Determine calibration factors
    if use_calibration:
        uncertainty_calibration_factor = manifest.uncertainty_calibration_factor
        temperature_scaling_factor = manifest.temperature_scaling_factor
        logger.info(f"  Temperature scaling factor: {temperature_scaling_factor:.4f}")
        logger.info(f"  Uncertainty calibration factor: {uncertainty_calibration_factor:.4f}")
    else:
        uncertainty_calibration_factor = 1.0
        temperature_scaling_factor = 1.0
        logger.info("  Calibration disabled (using raw predictions)")

    # Load shared ESM model with Flash Attention if available
    logger.info(f"  Loading ESM model to {device}...")
    tokenizer, esm_model = load_esm_model(
        model_name=esm_model_name,
        device=device,
        use_flash_attention=(device == "cuda"),
    )

    # Load all ST models (resolve relative paths from manifest directory)
    logger.info(f"  Loading {manifest.ensemble_size} ST models...")
    st_models = []
    for member in manifest.members:
        st_path = manifest_dir / member.st_path
        st_model = SentenceTransformer(str(st_path))
        st_models.append(st_model)

    # Load all MLP heads (resolve relative paths from manifest directory)
    # Prefer manifest-configured architecture to avoid shape mismatches.
    mlp_hidden_dims = config_snapshot.get("mlp_hidden_dims", config.MLP_HIDDEN_DIMS)
    mlp_dropout = config_snapshot.get("mlp_dropout", config.MLP_DROPOUT)
    mlp_input_dim = config_snapshot.get("esm_embed_dim", config.ESM_EMBED_DIM) + config_snapshot.get(
        "st_embed_dim",
        config.ST_EMBED_DIM,
    )

    logger.info(
        "  Loading %d MLP heads (hidden_dims=%s, dropout=%.3f)...",
        manifest.ensemble_size,
        mlp_hidden_dims,
        mlp_dropout,
    )
    mlps = []
    for member in manifest.members:
        mlp_path = manifest_dir / member.mlp_path
        mlp = DisorderMLP(input_dim=mlp_input_dim, hidden_dims=mlp_hidden_dims, dropout=mlp_dropout)
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
        uncertainty_calibration_factor=uncertainty_calibration_factor,
        temperature_scaling_factor=temperature_scaling_factor,
    )
