#!/usr/bin/env python3
"""
CatBoost baseline for disorder prediction using ESM embeddings.

This provides a strong baseline comparison for the ensemble MLP approach.
Uses RMSEWithUncertainty loss for built-in uncertainty estimation.

Usage:
    # Train a new model
    adopt2g-catboost train --output models/catboost

    # Predict with trained model
    adopt2g-catboost predict --input proteins.fasta --output results.csv --model models/catboost
"""

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import torch
from catboost import CatBoostRegressor, Pool
from tqdm import tqdm
from transformers import AutoModel, PreTrainedTokenizerBase

from adopt2g import config
from adopt2g.data import (
    CsvIterator,
    FastaIterator,
    compute_rebalancing_weights,
    load_trizod_dataset,
    split_train_test,
    split_train_val,
)
from adopt2g.embeddings import extract_esm_embeddings, load_esm_embedding, load_esm_model
from adopt2g.inference import estimate_max_residues_per_batch
from adopt2g.utils import setup_logging

logger = logging.getLogger(__name__)

# CatBoost hyperparameters (stronger defaults)
DEFAULT_ITERATIONS = 3000
DEFAULT_DEPTH = 8
DEFAULT_LEARNING_RATE = 0.045
DEFAULT_L2_LEAF_REG = 2.5
DEFAULT_BAGGING_TEMPERATURE = 0.5
DEFAULT_RANDOM_STRENGTH = 2.0


def prepare_catboost_data(
    df: pd.DataFrame,
    esm_cache_dir: Path,
    drop_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare ESM embeddings and g-scores for CatBoost training.

    Args:
        df: DataFrame with sequences and gscores
        esm_cache_dir: Directory with cached ESM embeddings
        drop_missing: Whether to drop residues with NaN g-scores

    Returns:
        (embeddings, g_scores) arrays
    """
    # Count total residues for pre-allocation
    total_residues = 0
    for label in df.index:
        gscores = df.loc[label, "gscores"]
        if drop_missing:
            total_residues += np.sum(~np.isnan(gscores))
        else:
            total_residues += len(gscores)

    feature_dim = config.ESM_EMBED_DIM
    logger.info(f"Total residues: {total_residues:,}, feature dim: {feature_dim}")

    # Pre-allocate arrays
    all_embeddings = np.empty((total_residues, feature_dim), dtype=np.float32)
    all_gscores = np.empty(total_residues, dtype=np.float32)

    idx = 0
    for label in tqdm(df.index, desc="Preparing ESM features"):
        esm_path = esm_cache_dir / f"{label}.pt"
        if not esm_path.exists():
            raise FileNotFoundError(
                f"ESM embedding for {label} not found at {esm_path}. Run embedding extraction first.",
            )

        esm_emb = load_esm_embedding(esm_path).numpy()
        sequence = df.loc[label, "sequence"]
        # Skip CLS token (index 0) and take only sequence residues
        # ESM may truncate long sequences, so use min of actual length
        actual_len = min(len(sequence), esm_emb.shape[0] - 1)
        esm_emb = esm_emb[1 : 1 + actual_len].astype(np.float32)

        gscores = df.loc[label, "gscores"][:actual_len]  # Truncate to match

        # Get valid indices (within truncated range)
        valid_idx = np.where(~np.isnan(gscores))[0] if drop_missing else np.arange(len(gscores))

        n_valid = len(valid_idx)
        if n_valid == 0:
            continue

        all_embeddings[idx : idx + n_valid] = esm_emb[valid_idx]
        all_gscores[idx : idx + n_valid] = gscores[valid_idx]
        idx += n_valid

    return all_embeddings[:idx], all_gscores[:idx]


def train_catboost_model(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    train_weights: np.ndarray | None = None,
    test_embeddings: np.ndarray | None = None,
    test_targets: np.ndarray | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    depth: int = DEFAULT_DEPTH,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    l2_leaf_reg: float = DEFAULT_L2_LEAF_REG,
    output_dir: Path | None = None,
    use_uncertainty: bool = True,
) -> CatBoostRegressor:
    """
    Train a CatBoost model for disorder prediction.

    Args:
        train_embeddings: Training embeddings (N, feature_dim)
        train_targets: Training g-scores (N,)
        train_weights: Optional sample weights for rebalancing
        test_embeddings: Optional test embeddings for evaluation
        test_targets: Optional test g-scores for evaluation
        iterations: Number of boosting iterations
        depth: Tree depth
        learning_rate: Learning rate
        l2_leaf_reg: L2 regularization coefficient
        output_dir: Directory to save model and metadata
        use_uncertainty: Use RMSEWithUncertainty for uncertainty estimation

    Returns:
        Trained CatBoostRegressor
    """
    logger.info("Training CatBoost model...")
    logger.info(f"  Training samples: {len(train_targets):,}, features: {train_embeddings.shape[1]}")
    logger.info(f"  Iterations: {iterations}, Depth: {depth}, LR: {learning_rate}")
    logger.info(f"  Uncertainty estimation: {use_uncertainty}")
    logger.info(f"  Sample weights: {train_weights is not None}")

    # Configure loss function
    loss_function = "RMSEWithUncertainty" if use_uncertainty else "RMSE"

    model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        loss_function=loss_function,
        random_seed=config.DEFAULT_SEED,
        verbose=100,
        task_type="GPU" if torch.cuda.is_available() else "CPU",
        # Additional regularization for robustness
        bagging_temperature=0.5,
        random_strength=1.0,
        border_count=254,
    )

    # Create pools with optional sample weights
    train_pool = Pool(train_embeddings, train_targets, weight=train_weights)
    eval_pool = Pool(test_embeddings, test_targets) if test_embeddings is not None else None

    # Train with early stopping
    model.fit(
        train_pool,
        eval_set=eval_pool,
        use_best_model=(eval_pool is not None),
        early_stopping_rounds=200 if eval_pool is not None else None,
    )

    # Evaluate on test set
    if test_embeddings is not None and test_targets is not None:
        preds = model.predict(test_embeddings)
        pred_means = preds[:, 0] if use_uncertainty else preds

        mse = np.mean((pred_means - test_targets) ** 2)
        mae = np.mean(np.abs(pred_means - test_targets))
        spearman = scipy.stats.spearmanr(test_targets, pred_means).correlation
        pearson = scipy.stats.pearsonr(test_targets, pred_means)[0]

        logger.info(f"Test MSE: {mse:.6f}")
        logger.info(f"Test MAE: {mae:.6f}")
        logger.info(f"Test Spearman: {spearman:.4f}")
        logger.info(f"Test Pearson: {pearson:.4f}")

        test_metrics = {
            "mse_test": float(mse),
            "mae_test": float(mae),
            "spearman_test": float(spearman),
            "pearson_test": float(pearson),
        }
    else:
        test_metrics = {}

    # Save model
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "catboost_esm.cbm"
        model.save_model(str(model_path))
        logger.info(f"Saved model to {model_path}")

        # Save metadata
        metadata = {
            "esm_model_name": config.ESM_MODEL_NAME,
            "esm_cache_dir": str(config.CACHE_PATH / "esm2"),
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "use_uncertainty": use_uncertainty,
            "loss_function": loss_function,
            "feature_dim": int(train_embeddings.shape[1]),
            **test_metrics,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {output_dir / 'metadata.json'}")

    return model


def load_catboost_model(model_dir: Path) -> tuple[CatBoostRegressor, dict]:
    """Load a trained CatBoost model and its metadata."""
    model_path = model_dir / "catboost_esm.cbm"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = CatBoostRegressor()
    model.load_model(str(model_path))
    logger.info(f"Loaded model from {model_path}")

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        logger.warning(f"Metadata not found at {metadata_path}, using defaults")
        metadata = {"use_uncertainty": True}

    return model, metadata


class CatBoostPredictor:
    """
    CatBoost predictor for disorder scores with uncertainty estimation.
    """

    def __init__(
        self,
        model: CatBoostRegressor,
        tokenizer: PreTrainedTokenizerBase,
        esm_model: AutoModel,
        use_uncertainty: bool = True,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.esm_model = esm_model
        self.use_uncertainty = use_uncertainty

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.esm_model = self.esm_model.to(device)
        self._max_seq_len = config.MAX_SEQUENCE_LEN

    def _get_esm_embeddings(self, sequences: list[str]) -> list[np.ndarray]:
        """Compute ESM embeddings for sequences."""
        results = []
        use_amp = self.device == "cuda"

        # Process in batches respecting max length
        batch: list[str] = []
        for seq in sequences:
            if batch and sum(len(s) for s in batch) + len(seq) >= self._max_seq_len:
                # Process batch
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
                    esm_out = self.esm_model(**inputs)
                for i, s in enumerate(batch):
                    esm_emb = esm_out.last_hidden_state[i, 1 : 1 + len(s), :].cpu().numpy()
                    results.append(esm_emb.astype(np.float32))
                batch = []
            batch.append(seq)

        # Process remaining batch
        if batch:
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
                esm_out = self.esm_model(**inputs)
            for i, s in enumerate(batch):
                esm_emb = esm_out.last_hidden_state[i, 1 : 1 + len(s), :].cpu().numpy()
                results.append(esm_emb.astype(np.float32))

        return results

    def _predict_sequence(self, esm_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict g-scores for a single sequence.

        Returns:
            (gscores, uncertainties) arrays
        """
        preds = self.model.predict(esm_emb)

        if self.use_uncertainty and preds.ndim == 2:
            gscores = preds[:, 0]
            variances = np.maximum(preds[:, 1], 0)
            uncertainties = np.sqrt(variances)
        else:
            gscores = preds if preds.ndim == 1 else preds.flatten()
            uncertainties = np.zeros_like(gscores)

        # Clip gscores to [0, 1] range
        gscores = np.clip(gscores, 0, 1)

        return gscores, uncertainties

    def predict(
        self,
        sequences: list[str],
        show_progress: bool = False,
        max_residues_per_batch: int | None = None,
    ) -> list[dict]:
        """
        Predict disorder scores with uncertainty for sequences.

        Returns list of dicts matching the EnsemblePrediction format.
        """
        if max_residues_per_batch is None:
            max_residues_per_batch = estimate_max_residues_per_batch()

        if isinstance(sequences, str):
            sequences = [sequences]

        results = []
        batch_seqs: list[str] = []
        batch_residues = 0

        seq_iter = tqdm(sequences, desc="Predicting", disable=not show_progress)

        for seq in seq_iter:
            if batch_seqs and batch_residues + len(seq) > max_residues_per_batch:
                # Process batch
                esm_embeddings = self._get_esm_embeddings(batch_seqs)
                for emb in esm_embeddings:
                    gscores, uncerts = self._predict_sequence(emb)
                    results.append(self._format_prediction(gscores, uncerts))
                batch_seqs = []
                batch_residues = 0

            batch_seqs.append(seq)
            batch_residues += len(seq)

        # Process final batch
        if batch_seqs:
            esm_embeddings = self._get_esm_embeddings(batch_seqs)
            for emb in esm_embeddings:
                gscores, uncerts = self._predict_sequence(emb)
                results.append(self._format_prediction(gscores, uncerts))

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def _format_prediction(
        self,
        gscores: np.ndarray,
        uncertainties: np.ndarray,
    ) -> dict:
        """Format prediction as a dict matching EnsemblePrediction."""
        gscores_lower = np.clip(gscores - uncertainties, 0, 1)
        gscores_upper = np.clip(gscores + uncertainties, 0, 1)

        mean_gscore = float(np.nanmean(gscores))
        mean_uncert = float(np.nanmean(uncertainties) / np.sqrt(len(gscores)))
        mean_gscore_lower = max(0, mean_gscore - mean_uncert)
        mean_gscore_upper = min(1, mean_gscore + mean_uncert)

        return {
            "gscores": gscores.tolist(),
            "gscores_lower": gscores_lower.tolist(),
            "gscores_upper": gscores_upper.tolist(),
            "mean_gscore": mean_gscore,
            "mean_gscore_lower": mean_gscore_lower,
            "mean_gscore_upper": mean_gscore_upper,
        }


def load_catboost_predictor(
    model_dir: Path,
    device: str | None = None,
) -> CatBoostPredictor:
    """
    Load a CatBoost predictor ready for inference.

    Args:
        model_dir: Directory containing catboost_esm.cbm and metadata.json
        device: Device for ESM model

    Returns:
        Ready-to-use CatBoostPredictor
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, metadata = load_catboost_model(model_dir)
    use_uncertainty = metadata.get("use_uncertainty", True)

    logger.info(f"Loading ESM model to {device}...")
    tokenizer, esm_model = load_esm_model(
        model_name=metadata.get("esm_model_name", config.ESM_MODEL_NAME),
        device=device,
        use_flash_attention=(device == "cuda"),
    )

    return CatBoostPredictor(
        model=model,
        tokenizer=tokenizer,
        esm_model=esm_model,
        use_uncertainty=use_uncertainty,
        device=device,
    )


def run_train(args: argparse.Namespace) -> CatBoostRegressor:
    """Run training mode."""
    # Load dataset
    logger.info("Loading TriZOD dataset...")
    df = load_trizod_dataset()
    df_train_full, df_test = split_train_test(df)
    # Split training data into train and validation (test is held out)
    df_train, df_val = split_train_val(df_train_full, val_frac=0.1, random_state=config.DEFAULT_SEED)
    logger.info(f"Train sequences: {len(df_train)}, Val sequences: {len(df_val)}, Test sequences: {len(df_test)}")

    # Ensure ESM embeddings are cached
    esm_cache_dir = config.CACHE_PATH / "esm2"
    if not esm_cache_dir.exists() or len(list(esm_cache_dir.glob("*.pt"))) < len(df):
        logger.info("Extracting ESM embeddings (this may take a while)...")
        tokenizer, esm_model = load_esm_model()
        extract_esm_embeddings(df, esm_cache_dir, tokenizer, esm_model)
        del tokenizer, esm_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Prepare training data
    logger.info("Preparing training data...")
    train_embeddings, train_targets = prepare_catboost_data(df_train, esm_cache_dir)

    # Compute rebalancing weights to address class imbalance
    if args.no_weights:
        logger.info("Using uniform weights (--no-weights)")
        train_weights = None
    else:
        logger.info("Computing rebalancing weights...")
        train_weights = compute_rebalancing_weights(train_targets)

    # Prepare validation data (for early stopping)
    logger.info("Preparing validation data...")
    val_embeddings, val_targets = prepare_catboost_data(df_val, esm_cache_dir)

    # Train model (using validation set for early stopping, test set is held out)
    model = train_catboost_model(
        train_embeddings=train_embeddings,
        train_targets=train_targets,
        train_weights=train_weights,
        test_embeddings=val_embeddings,
        test_targets=val_targets,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        output_dir=args.output,
        use_uncertainty=not args.no_uncertainty,
    )

    # Final evaluation on held-out test set
    logger.info("Evaluating on held-out test set...")
    test_embeddings, test_targets = prepare_catboost_data(df_test, esm_cache_dir)
    use_uncertainty = not args.no_uncertainty
    preds = model.predict(test_embeddings)
    pred_means = preds[:, 0] if use_uncertainty else preds

    mse = np.mean((pred_means - test_targets) ** 2)
    mae = np.mean(np.abs(pred_means - test_targets))
    spearman = scipy.stats.spearmanr(test_targets, pred_means).correlation
    pearson = scipy.stats.pearsonr(test_targets, pred_means)[0]

    logger.info(f"Held-out Test MSE: {mse:.6f}")
    logger.info(f"Held-out Test MAE: {mae:.6f}")
    logger.info(f"Held-out Test Spearman: {spearman:.4f}")
    logger.info(f"Held-out Test Pearson: {pearson:.4f}")

    # Update metadata with held-out test metrics
    if args.output is not None:
        metadata_path = args.output / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            metadata["mse_test_heldout"] = float(mse)
            metadata["mae_test_heldout"] = float(mae)
            metadata["spearman_test_heldout"] = float(spearman)
            metadata["pearson_test_heldout"] = float(pearson)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Updated metadata with held-out test metrics")

    return model


def run_predict(args: argparse.Namespace) -> None:
    """Run prediction mode."""
    # Load predictor
    predictor = load_catboost_predictor(args.model)
    logger.info(f"Running CatBoost prediction (uncertainty: {predictor.use_uncertainty})")

    # Load input data in chunks
    seq_iter: CsvIterator | FastaIterator
    if args.input.suffix.lower() == ".csv":
        try:
            label_col, seq_col = args.csv_cols.split(",")
            seq_iter = CsvIterator(
                args.input,
                chunk_size=args.chunk_size,
                label_col=label_col,
                sequence_col=seq_col,
            )
            logger.info(f"Loading input from CSV: {args.input} (cols: {label_col}, {seq_col})")
        except Exception as e:
            logger.warning(f"Failed to load as CSV, falling back to FASTA: {e}")
            seq_iter = FastaIterator(args.input, chunk_size=args.chunk_size)
            logger.info(f"Loading input from FASTA: {args.input}")
    else:
        seq_iter = FastaIterator(args.input, chunk_size=args.chunk_size)
        logger.info(f"Loading input from FASTA: {args.input}")

    n_chunks = len(seq_iter)
    n_digits = len(str(n_chunks))
    logger.info(f"Processing {n_chunks} chunks")

    for chunk_no, records in enumerate(tqdm(seq_iter, total=n_chunks)):
        chunk_file = Path(f"{args.output}.{chunk_no:0{n_digits}d}")

        if chunk_file.exists():
            logger.info(f"Chunk {chunk_no} already exists, skipping")
            continue

        headers, sequences = zip(*records, strict=True)

        # Predict
        predictions = predictor.predict(list(sequences), show_progress=True)

        # Save chunk results
        results = []
        for header, seq, pred in zip(headers, sequences, predictions, strict=True):
            result = {
                "label": header,
                "sequence": seq,
                "mean_gscore": pred["mean_gscore"],
                "mean_gscore_lower": pred["mean_gscore_lower"],
                "mean_gscore_upper": pred["mean_gscore_upper"],
                "gscores": json.dumps(pred["gscores"]),
                "gscores_lower": json.dumps(pred["gscores_lower"]),
                "gscores_upper": json.dumps(pred["gscores_upper"]),
            }
            results.append(result)

        pd.DataFrame(results).to_csv(chunk_file, index=False)

    # Combine chunks
    logger.info("Combining chunks...")
    chunks = [pd.read_csv(Path(f"{args.output}.{i:0{n_digits}d}")) for i in range(n_chunks)]
    pd.concat(chunks, ignore_index=True).to_csv(args.output, index=False)

    # Clean up chunk files
    for i in range(n_chunks):
        Path(f"{args.output}.{i:0{n_digits}d}").unlink()

    logger.info(f"Results saved to {args.output}")
    cols = "label, sequence, mean_gscore, mean_gscore_lower, mean_gscore_upper, gscores, gscores_lower, gscores_upper"
    logger.info(f"Columns: {cols}")
    logger.info("  - mean_gscore_lower/upper: confidence bounds for mean_gscore")
    logger.info("  - gscores_lower/upper: per-residue confidence bounds (JSON arrays)")
    logger.info("(Use json.loads() to parse JSON array columns)")


# =============================================================================
# Hyperparameter Search
# =============================================================================

# Search space for random hyperparameter search
SEARCH_SPACE = {
    "iterations": (1000, 8000),
    "depth": (6, 12),
    "learning_rate": (0.01, 0.1),  # log-uniform
    "l2_leaf_reg": (1.0, 10.0),  # log-uniform
    "bagging_temperature": (0.0, 1.0),
    "random_strength": (0.5, 10.0),
}


def sample_hyperparameters(rng: np.random.Generator) -> dict:
    """Sample random hyperparameters from search space."""
    return {
        "iterations": int(rng.integers(SEARCH_SPACE["iterations"][0], SEARCH_SPACE["iterations"][1])),
        "depth": int(rng.integers(SEARCH_SPACE["depth"][0], SEARCH_SPACE["depth"][1] + 1)),
        "learning_rate": float(
            np.exp(
                rng.uniform(
                    np.log(SEARCH_SPACE["learning_rate"][0]),
                    np.log(SEARCH_SPACE["learning_rate"][1]),
                ),
            ),
        ),
        "l2_leaf_reg": float(
            np.exp(
                rng.uniform(
                    np.log(SEARCH_SPACE["l2_leaf_reg"][0]),
                    np.log(SEARCH_SPACE["l2_leaf_reg"][1]),
                ),
            ),
        ),
        "bagging_temperature": float(rng.uniform(*SEARCH_SPACE["bagging_temperature"])),
        "random_strength": float(rng.uniform(*SEARCH_SPACE["random_strength"])),
    }


def run_trial(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    train_weights: np.ndarray,
    test_embeddings: np.ndarray,
    test_targets: np.ndarray,
    hyperparams: dict,
    use_uncertainty: bool = True,
) -> dict:
    """Run a single hyperparameter trial and return metrics."""
    loss_function = "RMSEWithUncertainty" if use_uncertainty else "RMSE"

    model = CatBoostRegressor(
        iterations=hyperparams["iterations"],
        depth=hyperparams["depth"],
        learning_rate=hyperparams["learning_rate"],
        l2_leaf_reg=hyperparams["l2_leaf_reg"],
        loss_function=loss_function,
        random_seed=config.DEFAULT_SEED,
        verbose=0,  # Suppress output during search
        task_type="GPU" if torch.cuda.is_available() else "CPU",
        bagging_temperature=hyperparams["bagging_temperature"],
        random_strength=hyperparams["random_strength"],
        border_count=254,
    )

    train_pool = Pool(train_embeddings, train_targets, weight=train_weights)
    eval_pool = Pool(test_embeddings, test_targets)

    model.fit(
        train_pool,
        eval_set=eval_pool,
        use_best_model=True,
        early_stopping_rounds=100,
    )

    # Evaluate
    preds = model.predict(test_embeddings)
    pred_means = preds[:, 0] if use_uncertainty else preds

    mse = float(np.mean((pred_means - test_targets) ** 2))
    mae = float(np.mean(np.abs(pred_means - test_targets)))
    spearman = float(scipy.stats.spearmanr(test_targets, pred_means).correlation)
    pearson = float(scipy.stats.pearsonr(test_targets, pred_means)[0])

    return {
        "hyperparams": hyperparams,
        "mse": mse,
        "mae": mae,
        "spearman": spearman,
        "pearson": pearson,
        "model": model,
    }


def run_search(args: argparse.Namespace) -> None:
    """Run random hyperparameter search."""
    # Load dataset
    logger.info("Loading TriZOD dataset...")
    df = load_trizod_dataset()
    df_train_full, df_test = split_train_test(df)
    # Split training data into train and validation (test is held out)
    df_train, df_val = split_train_val(df_train_full, val_frac=0.1, random_state=config.DEFAULT_SEED)
    logger.info(f"Train sequences: {len(df_train)}, Val sequences: {len(df_val)}, Test sequences: {len(df_test)}")

    # Ensure ESM embeddings are cached
    esm_cache_dir = config.CACHE_PATH / "esm2"
    if not esm_cache_dir.exists() or len(list(esm_cache_dir.glob("*.pt"))) < len(df):
        logger.info("Extracting ESM embeddings (this may take a while)...")
        tokenizer, esm_model = load_esm_model()
        extract_esm_embeddings(df, esm_cache_dir, tokenizer, esm_model)
        del tokenizer, esm_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Prepare data
    logger.info("Preparing training data...")
    train_embeddings, train_targets = prepare_catboost_data(df_train, esm_cache_dir)

    if args.no_weights:
        logger.info("Using uniform weights (--no-weights)")
        train_weights = None
    else:
        logger.info("Computing rebalancing weights...")
        train_weights = compute_rebalancing_weights(train_targets)

    # Prepare validation data (for early stopping and model selection during search)
    logger.info("Preparing validation data...")
    val_embeddings, val_targets = prepare_catboost_data(df_val, esm_cache_dir)

    # Run random search (using validation set for early stopping and model selection)
    logger.info(f"Starting random hyperparameter search ({args.n_trials} trials)...")
    rng = np.random.default_rng(args.seed)

    best_trial = None
    all_results = []

    for trial_idx in range(args.n_trials):
        hyperparams = sample_hyperparameters(rng)
        logger.info(f"\nTrial {trial_idx + 1}/{args.n_trials}")
        logger.info(
            f"  iterations={hyperparams['iterations']}, depth={hyperparams['depth']}, "
            f"lr={hyperparams['learning_rate']:.4f}, l2={hyperparams['l2_leaf_reg']:.2f}",
        )

        try:
            result = run_trial(
                train_embeddings,
                train_targets,
                train_weights,
                val_embeddings,
                val_targets,
                hyperparams,
                use_uncertainty=not args.no_uncertainty,
            )

            logger.info(
                f"  Val MSE={result['mse']:.6f}, MAE={result['mae']:.4f}, "
                f"Spearman={result['spearman']:.4f}, Pearson={result['pearson']:.4f}",
            )

            all_results.append(
                {
                    "trial": trial_idx + 1,
                    **hyperparams,
                    "mse_val": result["mse"],
                    "mae_val": result["mae"],
                    "spearman_val": result["spearman"],
                    "pearson_val": result["pearson"],
                },
            )

            if best_trial is None or result["pearson"] > best_trial["pearson"]:
                best_trial = result
                logger.info("  *** New best! ***")

        except Exception as e:
            logger.warning(f"  Trial failed: {e}")
            continue

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)

    # Save all trial results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output / "search_results.csv", index=False)
    logger.info(f"\nSaved all trial results to {args.output / 'search_results.csv'}")

    # Save best model and evaluate on held-out test set
    if best_trial is not None:
        logger.info("\n" + "=" * 60)
        logger.info("Best hyperparameters:")
        for k, v in best_trial["hyperparams"].items():
            logger.info(f"  {k}: {v}")
        logger.info("\nBest validation metrics:")
        logger.info(f"  Val MSE: {best_trial['mse']:.6f}")
        logger.info(f"  Val MAE: {best_trial['mae']:.4f}")
        logger.info(f"  Val Spearman: {best_trial['spearman']:.4f}")
        logger.info(f"  Val Pearson: {best_trial['pearson']:.4f}")

        # Evaluate best model on held-out test set
        logger.info("\nEvaluating on held-out test set...")
        test_embeddings, test_targets = prepare_catboost_data(df_test, esm_cache_dir)
        use_uncertainty = not args.no_uncertainty
        preds = best_trial["model"].predict(test_embeddings)
        pred_means = preds[:, 0] if use_uncertainty else preds

        mse_test = float(np.mean((pred_means - test_targets) ** 2))
        mae_test = float(np.mean(np.abs(pred_means - test_targets)))
        spearman_test = float(scipy.stats.spearmanr(test_targets, pred_means).correlation)
        pearson_test = float(scipy.stats.pearsonr(test_targets, pred_means)[0])

        logger.info(f"  Held-out Test MSE: {mse_test:.6f}")
        logger.info(f"  Held-out Test MAE: {mae_test:.4f}")
        logger.info(f"  Held-out Test Spearman: {spearman_test:.4f}")
        logger.info(f"  Held-out Test Pearson: {pearson_test:.4f}")

        # Save best model
        model_path = args.output / "catboost_esm.cbm"
        best_trial["model"].save_model(str(model_path))
        logger.info(f"\nSaved best model to {model_path}")

        # Save metadata with both validation and held-out test metrics
        metadata = {
            "esm_model_name": config.ESM_MODEL_NAME,
            "esm_cache_dir": str(config.CACHE_PATH / "esm2"),
            "use_uncertainty": not args.no_uncertainty,
            "loss_function": "RMSEWithUncertainty" if not args.no_uncertainty else "RMSE",
            "feature_dim": int(train_embeddings.shape[1]),
            "n_trials": int(args.n_trials),
            "best_hyperparams": best_trial["hyperparams"],
            # Validation metrics (used for model selection)
            "mse_val": best_trial["mse"],
            "mae_val": best_trial["mae"],
            "spearman_val": best_trial["spearman"],
            "pearson_val": best_trial["pearson"],
            # Held-out test metrics (final evaluation)
            "mse_test_heldout": mse_test,
            "mae_test_heldout": mae_test,
            "spearman_test_heldout": spearman_test,
            "pearson_test_heldout": pearson_test,
        }
        with open(args.output / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {args.output / 'metadata.json'}")
    else:
        logger.error("No successful trials!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CatBoost baseline for disorder prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  adopt2g-catboost train --output models/catboost

  # Random hyperparameter search (20 trials)
  adopt2g-catboost search --output models/catboost_search --n-trials 20

  # Predict with trained model
  adopt2g-catboost predict --input proteins.fasta --output results.csv --model models/catboost
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a CatBoost model")
    train_parser.add_argument(
        "--output",
        type=Path,
        default=config.MODELS_PATH / "catboost",
        help="Output directory for model (default: models/catboost)",
    )
    train_parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of boosting iterations (default: {DEFAULT_ITERATIONS})",
    )
    train_parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help=f"Tree depth (default: {DEFAULT_DEPTH})",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    train_parser.add_argument(
        "--l2-leaf-reg",
        type=float,
        default=DEFAULT_L2_LEAF_REG,
        help=f"L2 regularization (default: {DEFAULT_L2_LEAF_REG})",
    )
    train_parser.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="Disable uncertainty estimation (use plain RMSE loss)",
    )
    train_parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Disable rebalancing weights (use uniform weights)",
    )

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Predict with trained model")
    predict_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input FASTA or CSV file",
    )
    predict_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file",
    )
    predict_parser.add_argument(
        "--model",
        type=Path,
        default=config.MODELS_PATH / "catboost",
        help="Model directory (default: models/catboost)",
    )
    predict_parser.add_argument(
        "--csv-cols",
        type=str,
        default="label,sequence",
        help="Column names for label and sequence in CSV input (default: 'label,sequence')",
    )
    predict_parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Sequences per chunk",
    )

    # Search subcommand
    search_parser = subparsers.add_parser("search", help="Random hyperparameter search")
    search_parser.add_argument(
        "--output",
        type=Path,
        default=config.MODELS_PATH / "catboost",
        help="Output directory for search results and best model",
    )
    search_parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of random trials (default: 20)",
    )
    search_parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        help=f"Random seed for sampling (default: {config.DEFAULT_SEED})",
    )
    search_parser.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="Disable uncertainty estimation (use plain RMSE loss)",
    )
    search_parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Disable rebalancing weights (use uniform weights)",
    )

    args = parser.parse_args()
    setup_logging()

    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "search":
        run_search(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
