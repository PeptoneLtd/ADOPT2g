#!/usr/bin/env python3
"""
Full training pipeline for the disorder predictor.

Steps:
1. Train SentenceTransformer (BiEncoder) on sequence pairs
2. Extract ESM embeddings for train/test sets
3. Prepare combined embeddings
4. Train MLP prediction head
5. Evaluate on test set
"""

import argparse

from sentence_transformers import SentenceTransformer

import wandb
from src import config
from src.data import (
    compute_rebalancing_weights,
    load_st_classes,
    load_trizod_dataset,
    prepare_st_training_data,
    split_train_test,
)
from src.embeddings import extract_esm_embeddings, load_esm_model, prepare_residue_embeddings
from src.training import evaluate_model, set_seed, train_mlp, train_sentence_transformer


def main():
    parser = argparse.ArgumentParser(description="Train disorder predictor")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--overwrite-st", action="store_true", help="Retrain ST even if model exists")
    parser.add_argument("--overwrite-esm", action="store_true", help="Re-extract ESM embeddings even if cached")
    parser.add_argument("--epochs", type=int, default=config.MLP_EPOCHS, help="MLP training epochs")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()

    set_seed(args.seed)

    # Initialize Weights & Biases
    if not args.no_wandb:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            config={
                # General
                "seed": args.seed,
                # Sentence Transformer
                "st_base_model": config.ST_BASE_MODEL,
                "st_num_iterations": config.ST_NUM_ITERATIONS,
                "st_batch_size": config.ST_BATCH_SIZE,
                "st_num_epochs": config.ST_NUM_EPOCHS,
                "st_warmup_ratio": config.ST_WARMUP_RATIO,
                # MLP
                "mlp_hidden_dims": config.MLP_HIDDEN_DIMS,
                "mlp_dropout": config.MLP_DROPOUT,
                "mlp_lr": config.MLP_LR,
                "mlp_weight_decay": config.MLP_WEIGHT_DECAY,
                "mlp_epochs": args.epochs,
                "mlp_batch_size": config.MLP_BATCH_SIZE,
                # Embeddings
                "esm_model": config.ESM_MODEL_NAME,
                "esm_embed_dim": config.ESM_EMBED_DIM,
                "st_embed_dim": config.ST_EMBED_DIM,
                # Rebalancing
                "rebalance_a": config.REBALANCE_A,
                "rebalance_b": config.REBALANCE_B,
            },
        )

    # Paths
    st_model_path = config.DATA_PATH / "st" / str(args.seed)
    mlp_path = config.CHECKPOINTS_PATH / f"mlp_{args.seed}.pt"
    esm_cache_path = config.DATA_PATH / "esm2"

    # Create directories
    config.DATA_PATH.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    trizod_df = load_trizod_dataset()
    st_classes_df = load_st_classes()

    df_train, df_test = split_train_test(trizod_df)
    print(f"Train: {len(df_train)} sequences, Test: {len(df_test)} sequences")

    # Log dataset sizes to wandb
    if wandb.run is not None:
        wandb.config.update(
            {
                "train_sequences": len(df_train),
                "test_sequences": len(df_test),
            }
        )

    # Step 1: Train or load SentenceTransformer
    if st_model_path.exists() and not args.overwrite_st:
        print(f"Loading existing ST model from {st_model_path}")
        st_model = SentenceTransformer(str(st_model_path))
    else:
        print("Training SentenceTransformer...")
        x_train_st, y_train_st = prepare_st_training_data(st_classes_df, random_state=args.seed)
        st_model = train_sentence_transformer(
            sequences=x_train_st,
            classes=y_train_st,
            output_path=st_model_path,
        )

    # Step 2: Extract ESM embeddings (skip if all cached)
    def all_embeddings_cached(df, cache_dir):
        return all((cache_dir / f"{label}.pt").exists() for label in df.index)

    all_cached = all_embeddings_cached(trizod_df, esm_cache_path)

    if all_cached and not args.overwrite_esm:
        print("Skipping ESM extraction (all embeddings cached)")
    else:
        print("Loading ESM model...")
        tokenizer, esm_model = load_esm_model()

        print("Extracting ESM embeddings...")
        extract_esm_embeddings(trizod_df, esm_cache_path, tokenizer, esm_model, overwrite=args.overwrite_esm)

        # Free GPU memory
        del esm_model
        import torch

        torch.cuda.empty_cache()

    # Step 3: Prepare combined embeddings
    print("Preparing training embeddings...")
    X_train, y_train = prepare_residue_embeddings(df_train, esm_cache_path, st_model)

    print("Preparing test embeddings...")
    X_test, y_test = prepare_residue_embeddings(df_test, esm_cache_path, st_model)

    print(f"Train: {len(y_train)} residues, Test: {len(y_test)} residues")

    # Log residue counts to wandb
    if wandb.run is not None:
        wandb.config.update(
            {
                "train_residues": len(y_train),
                "test_residues": len(y_test),
            }
        )

    # Step 4: Compute rebalancing weights
    print("Computing rebalancing weights...")
    weights = compute_rebalancing_weights(y_train)

    # Step 5: Train MLP (model outputs g-scores directly via sigmoid)
    print("Training MLP...")
    mlp = train_mlp(
        train_embeddings=X_train,
        train_targets=y_train,
        train_weights=weights,
        val_embeddings=X_test,
        val_targets=y_test,
        epochs=args.epochs,
        output_path=mlp_path,
    )

    # Step 6: Final evaluation
    print("\nFinal evaluation on test set:")
    metrics = evaluate_model(mlp, X_test, y_test)
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Spearman correlation: {metrics['spearman_corr']:.4f}")

    # Log final metrics to wandb
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
        # Log model artifacts
        wandb.save(str(mlp_path))
        wandb.finish()

    print(f"\nTraining complete! Models saved to:")
    print(f"  ST: {st_model_path}")
    print(f"  MLP: {mlp_path}")


if __name__ == "__main__":
    main()
