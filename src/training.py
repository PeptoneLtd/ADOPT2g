"""Training functions for the disorder predictor."""

import random
from pathlib import Path

import numpy as np
import scipy.stats
import torch
import wandb
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .data import ResidueDataset, generate_sequence_pairs
from .models import DisorderMLP


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_sentence_transformer(
    sequences: np.ndarray,
    classes: np.ndarray,
    base_model: str = config.ST_BASE_MODEL,
    num_iterations: int = config.ST_NUM_ITERATIONS,
    batch_size: int = config.ST_BATCH_SIZE,
    num_epochs: int = config.ST_NUM_EPOCHS,
    warmup_ratio: float = config.ST_WARMUP_RATIO,
    output_path: Path | None = None,
) -> SentenceTransformer:
    """
    Train a SentenceTransformer model using contrastive learning.

    Uses cosine similarity loss on positive/negative sequence pairs.
    """
    model = SentenceTransformer(base_model)
    print(f"SentenceTransformer device: {model.device}")

    # Generate training pairs
    train_examples = generate_sequence_pairs(
        sequences.tolist(),
        classes,
        num_iterations=num_iterations,
    )

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size, num_workers=4)
    train_loss = losses.CosineSimilarityLoss(model)
    warmup_steps = int(np.ceil(len(train_dataloader) * num_epochs * warmup_ratio))

    print(f"Training sentence transformer for {num_epochs} epochs (batch_size={batch_size})...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        use_amp=True,  # Mixed precision for faster training
    )

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        model.save(str(output_path))
        print(f"Saved sentence transformer to {output_path}")

    return model


def train_mlp(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    train_weights: np.ndarray | None = None,
    val_embeddings: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
    hidden_dims: list[int] = None,
    dropout: float = config.MLP_DROPOUT,
    lr: float = config.MLP_LR,
    weight_decay: float = config.MLP_WEIGHT_DECAY,
    epochs: int = config.MLP_EPOCHS,
    batch_size: int = config.MLP_BATCH_SIZE,
    device: str | None = None,
    output_path: Path | None = None,
) -> DisorderMLP:
    """
    Train the MLP prediction head.

    Uses weighted MSE loss on g-scores (values in [0, 1]).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if hidden_dims is None:
        hidden_dims = config.MLP_HIDDEN_DIMS

    # Create datasets
    # Note: num_workers=0 avoids fork issues with memory-mapped arrays
    # For regular arrays, num_workers=4 would be faster
    is_memmap = hasattr(train_embeddings, 'flush')  # memmaps have flush method
    num_workers = 0 if is_memmap else 4
    
    train_dataset = ResidueDataset(train_embeddings, train_targets, train_weights)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda" and num_workers > 0),
    )

    if val_embeddings is not None and val_targets is not None:
        val_dataset = ResidueDataset(val_embeddings, val_targets)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            num_workers=num_workers, pin_memory=(device == "cuda" and num_workers > 0),
        )
    else:
        val_loader = None

    # Initialize model
    model = DisorderMLP(
        input_dim=train_embeddings.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    print(f"MLP training on: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_emb, batch_target, batch_weight in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False
        ):
            emb, target, weight = batch_emb.to(device), batch_target.to(device), batch_weight.to(device)

            optimizer.zero_grad()
            pred = model(emb)

            # Weighted MSE loss
            loss = (weight * (pred - target) ** 2).mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(emb)

        train_loss /= len(train_dataset)
        scheduler.step()

        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_preds, all_targets = [], []

            with torch.no_grad():
                for batch_emb, batch_target, _ in val_loader:
                    emb, target = batch_emb.to(device), batch_target.to(device)
                    pred = model(emb)
                    val_loss += ((pred - target) ** 2).sum().item()
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

            val_loss /= len(val_dataset)
            corr = scipy.stats.spearmanr(all_targets, all_preds).correlation

            print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, corr={corr:.4f}")

            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/spearman_corr": corr,
                    "learning_rate": current_lr,
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
        else:
            print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}")

            # Log to wandb (training only)
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "learning_rate": current_lr,
                })

    # Restore best model if validation was used
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        print(f"Saved MLP to {output_path}")

    return model


def evaluate_model(
    model: DisorderMLP,
    embeddings: np.ndarray,
    targets: np.ndarray,
    device: str | None = None,
) -> dict:
    """Evaluate model on test data."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model = model.to(device)

    # Use single worker for memmap compatibility
    is_memmap = hasattr(embeddings, 'flush')
    dataset = ResidueDataset(embeddings, targets)
    loader = DataLoader(dataset, batch_size=config.MLP_BATCH_SIZE, num_workers=0 if is_memmap else 4)

    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_emb, batch_target, _ in loader:
            emb = batch_emb.to(device)
            target = batch_target
            pred = model(emb)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    spearman_corr = scipy.stats.spearmanr(all_targets, all_preds).correlation
    pearson_r = scipy.stats.pearsonr(all_targets, all_preds)[0]
    kendall_tau = scipy.stats.kendalltau(all_targets, all_preds).correlation

    return {
        "mse": mse,
        "mae": mae,
        "spearman_corr": spearman_corr,
        "pearson_r": pearson_r,
        "kendall_tau": kendall_tau,
        "predictions": all_preds,
        "targets": all_targets,
    }
