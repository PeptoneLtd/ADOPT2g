#!/usr/bin/env python3
"""
Full training pipeline for the disorder predictor.

Supports training an ensemble of models with different seeds and train/val splits
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
from src import config
from src.data import (
    compute_rebalancing_weights,
    load_st_classes,
    load_trizod_dataset,
    sample_st_training_data,
    split_train_test,
    split_train_val,
)
from src.embeddings import extract_esm_embeddings, load_esm_model, prepare_residue_embeddings
from src.ensemble import (
    EnsembleManifest,
    MemberConfig,
    create_config_snapshot,
    derive_member_seeds,
    generate_ensemble_id,
    get_ensemble_paths,
    get_member_paths,
)
from src.training import evaluate_model, set_seed, train_mlp, train_sentence_transformer


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
        run_name = f"member_{member_id}" if ensemble_id else f"seed_{seeds['seed_global']}"
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
            }
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
            }
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

    # Log final metrics
    if wandb.run is not None:
        wandb.log(
            {
                "test/mse": metrics["mse"],
                "test/mae": metrics["mae"],
                "test/spearman_corr": metrics["spearman_corr"],
                "test/pearson_r": metrics["pearson_r"],
                "test/kendall_tau": metrics["kendall_tau"],
            }
        )
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
        help="Number of ensemble members (default: 5)",
    )
    parser.add_argument(
        "--ensemble-id", type=str, default=None, help="Ensemble ID (default: derived from seed and size)"
    )
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="Base seed")

    # Split fractions
    parser.add_argument(
        "--mlp-val-frac",
        type=float,
        default=config.DEFAULT_MLP_VAL_FRAC,
        help="Fraction of training data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--st-train-frac",
        type=float,
        default=config.DEFAULT_ST_TRAIN_FRAC,
        help="Fraction of ST training data to use per member (default: 0.9)",
    )

    # Existing args
    parser.add_argument("--overwrite-st", action="store_true", help="Retrain ST even if exists")
    parser.add_argument("--overwrite-esm", action="store_true", help="Re-extract ESM embeddings")
    parser.add_argument("--epochs", type=int, default=config.MLP_EPOCHS, help="MLP epochs")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    args = parser.parse_args()

    # Generate ensemble ID if not provided
    ensemble_id = args.ensemble_id or generate_ensemble_id(args.seed, args.ensemble_size)
    is_ensemble = args.ensemble_size > 1

    print(f"Training {'ensemble' if is_ensemble else 'single model'}: {ensemble_id}")
    print(f"  Base seed: {args.seed}")
    print(f"  Ensemble size: {args.ensemble_size}")
    print(f"  MLP val fraction: {args.mlp_val_frac}")
    print(f"  ST train fraction: {args.st_train_frac}")

    # Setup paths
    esm_cache_path = config.DATA_PATH / "esm2"
    config.DATA_PATH.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

    # Get ensemble paths
    if is_ensemble:
        ensemble_paths = get_ensemble_paths(ensemble_id)
        ensemble_paths["base"].mkdir(parents=True, exist_ok=True)
        ensemble_paths["st_base"].mkdir(parents=True, exist_ok=True)
        ensemble_paths["mlp_base"].mkdir(parents=True, exist_ok=True)

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
    if is_ensemble:
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
        if is_ensemble:
            member_paths = get_member_paths(ensemble_id, member_id)
            st_path = member_paths["st"]
            mlp_path = member_paths["mlp"]
        else:
            # Legacy single-model paths
            st_path = config.DATA_PATH / "st" / str(args.seed)
            mlp_path = config.CHECKPOINTS_PATH / f"mlp_{args.seed}.pt"

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
            ensemble_id=ensemble_id if is_ensemble else None,
        )
        all_metrics.append(metrics)

        # Update manifest
        if is_ensemble:
            member_config = MemberConfig(
                member_id=member_id,
                st_train_frac=args.st_train_frac,
                mlp_val_frac=args.mlp_val_frac,
                st_path=str(st_path),
                mlp_path=str(mlp_path),
                **seeds,
            )
            manifest.members.append(member_config)
            manifest.save(ensemble_paths["manifest"])
            print(f"  Manifest updated: {ensemble_paths['manifest']}")

    # Print ensemble summary
    if is_ensemble:
        print(f"\n{'=' * 60}")
        print("Ensemble Training Complete!")
        print(f"{'=' * 60}")
        print(f"Ensemble ID: {ensemble_id}")
        print(f"Manifest: {ensemble_paths['manifest']}")
        print("\nPer-member test metrics:")

        for i, m in enumerate(all_metrics):
            print(f"  Member {i}: MSE={m['mse']:.4f}, Spearman={m['spearman_corr']:.4f}")

        # Compute ensemble summary statistics
        mse_vals = [m["mse"] for m in all_metrics]
        corr_vals = [m["spearman_corr"] for m in all_metrics]
        print("\nEnsemble summary:")
        print(f"  MSE: mean={np.mean(mse_vals):.4f}, std={np.std(mse_vals):.4f}")
        print(f"  Spearman: mean={np.mean(corr_vals):.4f}, std={np.std(corr_vals):.4f}")

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
            wandb.log(
                {
                    "ensemble/mse_mean": np.mean(mse_vals),
                    "ensemble/mse_std": np.std(mse_vals),
                    "ensemble/spearman_mean": np.mean(corr_vals),
                    "ensemble/spearman_std": np.std(corr_vals),
                }
            )
            wandb.save(str(ensemble_paths["manifest"]))
            wandb.finish()
    else:
        print("\nTraining complete!")
        print(f"  ST: {st_path}")
        print(f"  MLP: {mlp_path}")


if __name__ == "__main__":
    main()
