#!/usr/bin/env python3
"""
Full training pipeline for the disorder predictor.

Trains an ensemble of models with different seeds and train/val splits
for improved predictions with uncertainty estimates.

Steps per ensemble member:
1. Train SentenceTransformer (BiEncoder) on sequence pairs
2. Extract ESM embeddings for train/test sets (shared across ensemble)
3. Prepare combined embeddings
4. Train MLP prediction head
5. Evaluate on test set
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import wandb
from adopt2g import config
from adopt2g.data import (
    compute_rebalancing_weights,
    load_st_classes,
    load_trizod_dataset,
    sample_st_training_data,
    split_train_test,
    split_train_val,
)
from adopt2g.embeddings import extract_esm_embeddings, load_esm_model, prepare_residue_embeddings
from adopt2g.ensemble import (
    EnsembleManifest,
    MemberConfig,
    create_config_snapshot,
    derive_member_seeds,
    generate_ensemble_id,
    get_ensemble_dir,
    get_member_dir,
)
from adopt2g.inference import load_ensemble_predictor
from adopt2g.training import evaluate_model, set_seed, train_mlp, train_sentence_transformer


def compute_calibration_factor(
    manifest_path: Path,
    df_data: pd.DataFrame,
    esm_cache_path: Path,
) -> float:
    """
    Compute calibration factor for uncertainty estimates.

    The calibration factor is the standard deviation of normalized residuals:
    calibration_factor = std((true - pred) / pred_std)

    If uncertainties are well-calibrated, this should be ~1.0.
    If uncertainties are under-estimated, this will be > 1.0.

    Args:
        manifest_path: Path to ensemble manifest
        df_data: DataFrame with sequences and gscores to calibrate on
        esm_cache_path: Path to cached ESM embeddings

    Returns:
        Calibration factor (float)
    """
    # Load ensemble predictor without calibration (factor=1.0)
    predictor = load_ensemble_predictor(manifest_path, calibration_factor=1.0)

    # Collect all residue-level predictions and targets
    all_residuals_normalized = []

    for _label, row in df_data.iterrows():
        sequence = row["sequence"]
        true_gscores = np.array(row["gscores"])

        # Get ensemble prediction
        pred = predictor.predict([sequence])[0]
        pred_gscores = np.array(pred.gscores)
        pred_std = np.array(pred.gscores_std)

        # Compute normalized residuals, masking where std is 0 or NaN
        valid_mask = (pred_std > 1e-8) & ~np.isnan(pred_std) & ~np.isnan(true_gscores)
        if valid_mask.sum() > 0:
            residuals = (true_gscores[valid_mask] - pred_gscores[valid_mask]) / pred_std[valid_mask]
            all_residuals_normalized.extend(residuals)

    # Calibration factor is the std of normalized residuals
    all_residuals_normalized = np.array(all_residuals_normalized)
    calibration_factor = float(np.std(all_residuals_normalized))

    return calibration_factor


def train_single_member(
    member_id: int,
    seeds: dict[str, int],
    st_classes_df: pd.DataFrame,
    df_train_pool: pd.DataFrame,
    df_test: pd.DataFrame,
    esm_cache_path: Path,
    st_path: Path,
    mlp_path: Path,
    args: argparse.Namespace,
    ensemble_id: str | None = None,
) -> dict:
    """
    Train a single ensemble member.

    Returns dict with member metrics.
    """
    print(f"\n{'=' * 60}")
    print(f"Training ensemble member {member_id}")
    print(f"{'=' * 60}")

    # Set global seed for this member
    set_seed(seeds["seed_global"])

    # Initialize W&B run for this member
    if not args.no_wandb:
        run_name = f"member_{member_id}"
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            group=ensemble_id,
            name=run_name,
            reinit=True,
            config={
                # Member info
                "member_id": member_id,
                "ensemble_id": ensemble_id,
                **seeds,
                # ST config
                "st_base_model": config.ST_BASE_MODEL,
                "st_num_iterations": config.ST_NUM_ITERATIONS,
                "st_batch_size": config.ST_BATCH_SIZE,
                "st_num_epochs": config.ST_NUM_EPOCHS,
                "st_warmup_ratio": config.ST_WARMUP_RATIO,
                "st_train_frac": args.st_train_frac,
                # MLP config
                "mlp_hidden_dims": config.MLP_HIDDEN_DIMS,
                "mlp_dropout": config.MLP_DROPOUT,
                "mlp_lr": config.MLP_LR,
                "mlp_weight_decay": config.MLP_WEIGHT_DECAY,
                "mlp_epochs": args.epochs,
                "mlp_batch_size": config.MLP_BATCH_SIZE,
                "mlp_val_frac": args.mlp_val_frac,
                # Embeddings
                "esm_model": config.ESM_MODEL_NAME,
                "esm_embed_dim": config.ESM_EMBED_DIM,
                "st_embed_dim": config.ST_EMBED_DIM,
                # Rebalancing
                "rebalance_a": config.REBALANCE_A,
                "rebalance_b": config.REBALANCE_B,
            },
        )

    # Step 1: Train or load SentenceTransformer
    if st_path.exists() and not args.overwrite_st:
        print(f"Loading existing ST model from {st_path}")
        st_model = SentenceTransformer(str(st_path))
    else:
        print("Training SentenceTransformer...")
        # Use member-specific seed for ST sampling
        x_train_st, y_train_st = sample_st_training_data(
            st_classes_df,
            train_frac=args.st_train_frac,
            random_state=seeds["split_seed_st"],
        )
        # Set seed again before pair generation (uses np.random)
        set_seed(seeds["seed_st"])
        st_model = train_sentence_transformer(
            sequences=x_train_st,
            classes=y_train_st,
            output_path=st_path,
        )

    # Step 2: Split training data into train/val for this member
    df_train_member, df_val_member = split_train_val(
        df_train_pool,
        val_frac=args.mlp_val_frac,
        random_state=seeds["split_seed_mlp"],
    )

    print(f"Member {member_id} split:")
    print(f"  Train: {len(df_train_member)} sequences")
    print(f"  Val: {len(df_val_member)} sequences")
    print(f"  Test: {len(df_test)} sequences (fixed)")

    # Log dataset sizes
    if wandb.run is not None:
        wandb.config.update(
            {
                "train_sequences": len(df_train_member),
                "val_sequences": len(df_val_member),
                "test_sequences": len(df_test),
            },
        )

    # Step 3: Prepare embeddings
    print("Preparing training embeddings...")
    X_train, y_train = prepare_residue_embeddings(df_train_member, esm_cache_path, st_model)

    print("Preparing validation embeddings...")
    X_val, y_val = prepare_residue_embeddings(df_val_member, esm_cache_path, st_model)

    print("Preparing test embeddings...")
    X_test, y_test = prepare_residue_embeddings(df_test, esm_cache_path, st_model)

    print(f"  Train: {len(y_train):,} residues")
    print(f"  Val: {len(y_val):,} residues")
    print(f"  Test: {len(y_test):,} residues")

    if wandb.run is not None:
        wandb.config.update(
            {
                "train_residues": len(y_train),
                "val_residues": len(y_val),
                "test_residues": len(y_test),
            },
        )

    # Step 4: Compute rebalancing weights
    print("Computing rebalancing weights...")
    weights = compute_rebalancing_weights(y_train)

    # Step 5: Train MLP
    print("Training MLP...")
    set_seed(seeds["seed_mlp"])
    mlp = train_mlp(
        train_embeddings=X_train,
        train_targets=y_train,
        train_weights=weights,
        val_embeddings=X_val,
        val_targets=y_val,
        epochs=args.epochs,
        output_path=mlp_path,
    )

    # Step 6: Evaluate on test set
    print("\nEvaluation on test set:")
    metrics = evaluate_model(mlp, X_test, y_test)
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Spearman correlation: {metrics['spearman_corr']:.4f}")

    # Log final metrics with rich visualizations
    if wandb.run is not None:
        preds = metrics["predictions"]
        targets = metrics["targets"]
        residuals = preds - targets

        # Basic metrics
        log_dict = {
            "test/mse": metrics["mse"],
            "test/mae": metrics["mae"],
            "test/spearman_corr": metrics["spearman_corr"],
            "test/pearson_r": metrics["pearson_r"],
            "test/kendall_tau": metrics["kendall_tau"],
        }

        # Histograms
        log_dict["test/predictions_hist"] = wandb.Histogram(preds)
        log_dict["test/targets_hist"] = wandb.Histogram(targets)
        log_dict["test/residuals_hist"] = wandb.Histogram(residuals)

        # Sample points for scatter plot (max 10000 to keep dashboard responsive)
        n_samples = min(10000, len(preds))
        sample_idx = np.random.choice(len(preds), n_samples, replace=False)
        scatter_data = [[t, p] for t, p in zip(targets[sample_idx], preds[sample_idx])]
        scatter_table = wandb.Table(data=scatter_data, columns=["target", "prediction"])
        log_dict["test/pred_vs_target"] = wandb.plot.scatter(
            scatter_table,
            "target",
            "prediction",
            title=f"Test Set: Prediction vs Target (Member {member_id})",
        )

        # Performance summary bar chart
        summary_data = [
            ["Spearman", metrics["spearman_corr"]],
            ["Pearson", metrics["pearson_r"]],
            ["Kendall", metrics["kendall_tau"]],
        ]
        summary_table = wandb.Table(data=summary_data, columns=["metric", "value"])
        log_dict["test/correlation_summary"] = wandb.plot.bar(
            summary_table,
            "metric",
            "value",
            title="Test Set Correlation Metrics",
        )

        wandb.log(log_dict)
        wandb.save(str(mlp_path))
        wandb.finish()

    print(f"\nMember {member_id} complete:")
    print(f"  ST: {st_path}")
    print(f"  MLP: {mlp_path}")

    return {
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "spearman_corr": metrics["spearman_corr"],
        "pearson_r": metrics["pearson_r"],
        "kendall_tau": metrics["kendall_tau"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train disorder predictor ensemble")

    # Ensemble args
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=config.DEFAULT_ENSEMBLE_SIZE,
        help=f"Number of ensemble members (default: {config.DEFAULT_ENSEMBLE_SIZE})",
    )
    parser.add_argument(
        "--ensemble-id",
        type=str,
        default=None,
        help="Ensemble ID (default: derived from seed and size)",
    )
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="Base seed")

    # Split fractions
    parser.add_argument(
        "--mlp-val-frac",
        type=float,
        default=config.DEFAULT_MLP_VAL_FRAC,
        help=f"Fraction of training data for validation (default: {config.DEFAULT_MLP_VAL_FRAC})",
    )
    parser.add_argument(
        "--st-train-frac",
        type=float,
        default=config.DEFAULT_ST_TRAIN_FRAC,
        help=f"Fraction of ST training data to use per member (default: {config.DEFAULT_ST_TRAIN_FRAC})",
    )

    # Existing args
    parser.add_argument("--overwrite-st", action="store_true", help="Retrain ST even if exists")
    parser.add_argument("--overwrite-esm", action="store_true", help="Re-extract ESM embeddings")
    parser.add_argument("--epochs", type=int, default=config.MLP_EPOCHS, help="MLP epochs")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip uncertainty calibration factor computation (saves time)",
    )

    args = parser.parse_args()

    # Generate ensemble ID if not provided
    ensemble_id = args.ensemble_id or generate_ensemble_id(args.seed, args.ensemble_size)

    print(f"Training ensemble: {ensemble_id}")
    print(f"  Base seed: {args.seed}")
    print(f"  Ensemble size: {args.ensemble_size}")
    print(f"  MLP val fraction: {args.mlp_val_frac}")
    print(f"  ST train fraction: {args.st_train_frac}")

    # Setup paths
    esm_cache_path = config.CACHE_PATH / "esm2"
    config.CACHE_PATH.mkdir(parents=True, exist_ok=True)

    # Get ensemble directory
    ensemble_dir = get_ensemble_dir(ensemble_id)
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ensemble_dir / "manifest.json"

    # Load datasets (shared across all members)
    print("\nLoading datasets...")
    trizod_df = load_trizod_dataset()
    st_classes_df = load_st_classes()

    df_train_pool, df_test = split_train_test(trizod_df)
    print(f"Train pool: {len(df_train_pool)} sequences")
    print(f"Test set: {len(df_test)} sequences (fixed)")

    # Extract ESM embeddings once (shared cache)
    def all_embeddings_cached(df: pd.DataFrame, cache_dir: Path) -> bool:
        return all((cache_dir / f"{label}.pt").exists() for label in df.index)

    all_cached = all_embeddings_cached(trizod_df, esm_cache_path)

    if all_cached and not args.overwrite_esm:
        print("Skipping ESM extraction (all embeddings cached)")
    else:
        print("\nLoading ESM model...")
        tokenizer, esm_model = load_esm_model()

        print("Extracting ESM embeddings...")
        extract_esm_embeddings(trizod_df, esm_cache_path, tokenizer, esm_model, overwrite=args.overwrite_esm)

        # Free GPU memory
        del esm_model
        torch.cuda.empty_cache()

    # Initialize manifest for ensemble
    manifest = EnsembleManifest(
        ensemble_id=ensemble_id,
        ensemble_size=args.ensemble_size,
        base_seed=args.seed,
        config_snapshot=create_config_snapshot(),
    )

    # Train each member
    all_metrics = []

    for member_id in range(args.ensemble_size):
        # Get member-specific seeds
        seeds = derive_member_seeds(args.seed, member_id)

        # Get paths for this member
        member_dir = get_member_dir(ensemble_id, member_id)
        member_dir.mkdir(parents=True, exist_ok=True)
        st_path = member_dir / "st"
        mlp_path = member_dir / "mlp.pt"

        # Train member
        metrics = train_single_member(
            member_id=member_id,
            seeds=seeds,
            st_classes_df=st_classes_df,
            df_train_pool=df_train_pool,
            df_test=df_test,
            esm_cache_path=esm_cache_path,
            st_path=st_path,
            mlp_path=mlp_path,
            args=args,
            ensemble_id=ensemble_id,
        )
        all_metrics.append(metrics)

        # Update manifest with relative paths (relative to ensemble directory)
        member_config = MemberConfig(
            member_id=member_id,
            st_train_frac=args.st_train_frac,
            mlp_val_frac=args.mlp_val_frac,
            st_path=str(st_path.relative_to(ensemble_dir)),
            mlp_path=str(mlp_path.relative_to(ensemble_dir)),
            **seeds,
        )
        manifest.members.append(member_config)
        manifest.save(manifest_path)
        print(f"  Manifest updated: {manifest_path}")

    # Print ensemble summary
    print(f"\n{'=' * 60}")
    print("Ensemble Training Complete!")
    print(f"{'=' * 60}")
    print(f"Ensemble ID: {ensemble_id}")
    print(f"Manifest: {manifest_path}")
    print("\nPer-member test metrics:")

    for i, m in enumerate(all_metrics):
        print(f"  Member {i}: MSE={m['mse']:.4f}, Spearman={m['spearman_corr']:.4f}")

    # Compute ensemble summary statistics
    mse_vals = [m["mse"] for m in all_metrics]
    corr_vals = [m["spearman_corr"] for m in all_metrics]
    print("\nEnsemble summary:")
    print(f"  MSE: mean={np.mean(mse_vals):.4f}, std={np.std(mse_vals):.4f}")
    print(f"  Spearman: mean={np.mean(corr_vals):.4f}, std={np.std(corr_vals):.4f}")

    # Compute calibration factor on training data
    if args.skip_calibration:
        calibration_factor = 1.0
        print("\nSkipping calibration factor computation (--skip-calibration)")
        print(f"  Using default calibration factor: {calibration_factor:.4f}")
    else:
        print("\nComputing uncertainty calibration factor on training data...")
        calibration_factor = compute_calibration_factor(
            manifest_path=manifest_path,
            df_data=df_train_pool,
            esm_cache_path=esm_cache_path,
        )
        print(f"  Calibration factor: {calibration_factor:.4f}")

        # Update manifest with calibration factor
        manifest.calibration_factor = calibration_factor
        manifest.save(manifest_path)
        print("  Manifest updated with calibration factor")

    # Log ensemble summary to W&B
    if not args.no_wandb:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            group=ensemble_id,
            name="ensemble_summary",
            reinit=True,
            config={
                "ensemble_id": ensemble_id,
                "ensemble_size": args.ensemble_size,
                "base_seed": args.seed,
            },
        )

        # Extract all metric arrays
        mse_vals = np.array([m["mse"] for m in all_metrics])
        mae_vals = np.array([m["mae"] for m in all_metrics])
        spearman_vals = np.array([m["spearman_corr"] for m in all_metrics])
        pearson_vals = np.array([m["pearson_r"] for m in all_metrics])
        kendall_vals = np.array([m["kendall_tau"] for m in all_metrics])

        log_dict = {
            # Summary statistics
            "ensemble/mse_mean": np.mean(mse_vals),
            "ensemble/mse_std": np.std(mse_vals),
            "ensemble/mae_mean": np.mean(mae_vals),
            "ensemble/mae_std": np.std(mae_vals),
            "ensemble/spearman_mean": np.mean(spearman_vals),
            "ensemble/spearman_std": np.std(spearman_vals),
            "ensemble/pearson_mean": np.mean(pearson_vals),
            "ensemble/pearson_std": np.std(pearson_vals),
            "ensemble/kendall_mean": np.mean(kendall_vals),
            "ensemble/kendall_std": np.std(kendall_vals),
            # Calibration
            "ensemble/calibration_factor": calibration_factor,
        }

        # Create a table with all member metrics
        member_table_data = [
            [i, m["mse"], m["mae"], m["spearman_corr"], m["pearson_r"], m["kendall_tau"]]
            for i, m in enumerate(all_metrics)
        ]
        member_table = wandb.Table(
            data=member_table_data,
            columns=["member_id", "mse", "mae", "spearman", "pearson", "kendall"],
        )
        log_dict["ensemble/member_metrics"] = member_table

        # Bar chart for correlation metrics across members
        corr_chart_data = []
        for i in range(args.ensemble_size):
            corr_chart_data.append([f"Member {i}", "Spearman", spearman_vals[i]])
            corr_chart_data.append([f"Member {i}", "Pearson", pearson_vals[i]])
            corr_chart_data.append([f"Member {i}", "Kendall", kendall_vals[i]])
        corr_table = wandb.Table(
            data=corr_chart_data,
            columns=["member", "metric", "value"],
        )
        log_dict["ensemble/correlation_by_member"] = wandb.plot.bar(
            corr_table,
            "member",
            "value",
            title="Correlation Metrics by Member",
        )

        # Bar chart for error metrics across members
        error_chart_data = []
        for i in range(args.ensemble_size):
            error_chart_data.append([f"Member {i}", "MSE", mse_vals[i]])
            error_chart_data.append([f"Member {i}", "MAE", mae_vals[i]])
        error_table = wandb.Table(
            data=error_chart_data,
            columns=["member", "metric", "value"],
        )
        log_dict["ensemble/error_by_member"] = wandb.plot.bar(
            error_table,
            "member",
            "value",
            title="Error Metrics by Member",
        )

        # Summary bar chart showing mean metrics with labels
        summary_data = [
            ["Spearman (mean)", np.mean(spearman_vals)],
            ["Pearson (mean)", np.mean(pearson_vals)],
            ["Kendall (mean)", np.mean(kendall_vals)],
        ]
        summary_table = wandb.Table(data=summary_data, columns=["metric", "value"])
        log_dict["ensemble/mean_correlations"] = wandb.plot.bar(
            summary_table,
            "metric",
            "value",
            title="Ensemble Mean Correlations",
        )

        wandb.log(log_dict)
        wandb.save(str(manifest_path))
        wandb.finish()


if __name__ == "__main__":
    main()
