"""Ensemble manifest, configuration, and path management."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

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
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    temperature_scaling_factor: float = 1.0
    uncertainty_calibration_factor: float = 1.0

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
