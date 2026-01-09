#!/usr/bin/env python3
"""
Predict disorder scores from FASTA file.

Supports both ensemble predictions (default) with uncertainty estimates
and legacy single-model predictions.

Usage:
    # Ensemble mode (default) - outputs mean predictions + uncertainty
    python predict.py --fasta input.fasta --output results.csv --ensemble-id seed883_n5

    # Single model mode (legacy)
    python predict.py --fasta input.fasta --output results.csv --no-ensemble --seed 883
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config
from src.data import FastaIterator
from src.ensemble import find_latest_ensemble
from src.inference import load_ensemble_predictor, load_predictor


def estimate_max_residues_per_batch(usage_fraction: float = 0.75) -> int:
    """Estimate max residues per GPU batch based on available memory."""
    if torch.cuda.is_available():
        free_mem_gb = torch.cuda.mem_get_info()[0] / 1e9
        # Reserve ~3GB for models, use ~usage_fraction of remaining for safety
        usable_gb = max(0.5, (free_mem_gb - 3.0) * usage_fraction)
        # Empirical: ~5MB per residue for ESM2-650M fp16 forward pass
        return int(usable_gb * 1e9 / 5_000_000)
    return 10_000  # default for CPU


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict disorder from FASTA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use latest trained ensemble
  python predict.py --fasta proteins.fasta --output results.csv

  # Use specific ensemble
  python predict.py --fasta proteins.fasta --output results.csv --ensemble-id seed883_n5

  # Save per-model gscores for each ensemble member
  python predict.py --fasta proteins.fasta --output results.csv --save-member-gscores

  # Legacy single-model mode
  python predict.py --fasta proteins.fasta --output results.csv --no-ensemble --seed 883
""",
    )
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA file")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file")

    # Ensemble options (default mode)
    parser.add_argument(
        "--ensemble-id",
        type=str,
        default=None,
        help="Ensemble ID to load (default: latest trained ensemble)",
    )
    parser.add_argument(
        "--ensemble-manifest",
        type=Path,
        default=None,
        help="Path to ensemble manifest.json (alternative to --ensemble-id)",
    )

    # Single model options (legacy mode)
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Use single model instead of ensemble (legacy mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        help="Model seed for single-model mode",
    )

    # Processing options
    parser.add_argument("--chunk-size", type=int, default=1000, help="Sequences per chunk")
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable uncertainty calibration (use raw ensemble std)",
    )
    parser.add_argument(
        "--save-member-gscores",
        action="store_true",
        help="Save per-model gscores for each ensemble member (ensemble mode only)",
    )

    args = parser.parse_args()

    # Determine mode and load predictor
    use_ensemble = not args.no_ensemble

    if use_ensemble:
        # Ensemble mode
        if args.ensemble_manifest:
            manifest_path = args.ensemble_manifest
        elif args.ensemble_id:
            manifest_path = args.ensemble_id
        else:
            # Find latest ensemble
            latest = find_latest_ensemble()
            if latest is None:
                print("No trained ensemble found. Train one with run_pipeline.py")
                print("Or use --no-ensemble --seed <seed> for single-model mode")
                return
            manifest_path = latest

        # Load predictor with or without calibration
        calibration_factor = 1.0 if args.no_calibration else None
        predictor = load_ensemble_predictor(manifest_path, calibration_factor=calibration_factor)
        print("Running in ensemble mode (predictions include uncertainty)")
        if args.no_calibration:
            print("  Uncertainty calibration disabled (using raw ensemble std)")
        if args.save_member_gscores:
            print("  Saving per-member gscores")
    else:
        # Legacy single-model mode
        st_path = config.DATA_PATH / "st" / str(args.seed)
        mlp_path = config.CHECKPOINTS_PATH / f"mlp_{args.seed}.pt"

        if not st_path.exists():
            raise FileNotFoundError(f"ST model not found at {st_path}. Run training first.")
        if not mlp_path.exists():
            raise FileNotFoundError(f"MLP model not found at {mlp_path}. Run training first.")

        predictor = load_predictor(str(st_path), str(mlp_path))
        print("Running in single-model mode (no uncertainty estimates)")

    # Process FASTA in chunks
    fasta_iter = FastaIterator(args.fasta, chunk_size=args.chunk_size)
    n_chunks = len(fasta_iter)
    n_digits = len(str(n_chunks))
    max_residues = estimate_max_residues_per_batch()

    print(f"Processing {n_chunks} chunks, max {max_residues:_} residues per batch")

    for chunk_no, records in enumerate(tqdm(fasta_iter, total=n_chunks)):
        chunk_file = Path(f"{args.output}.{chunk_no:0{n_digits}d}")

        if chunk_file.exists():
            print(f"Chunk {chunk_no} already exists, skipping")
            continue

        headers, sequences = zip(*records)

        # Process in batches that fit in GPU memory
        all_predictions: list = []
        batch: list[str] = []
        batch_residues = 0

        for seq in tqdm(sequences, desc="Processing", leave=False):
            seq_len = len(seq)

            if batch and batch_residues + seq_len > max_residues:
                # Process current batch
                if use_ensemble:
                    all_predictions.extend(predictor.predict(batch, include_member_gscores=args.save_member_gscores))
                else:
                    all_predictions.extend(predictor.predict(batch))
                batch = []
                batch_residues = 0
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            batch.append(seq)
            batch_residues += seq_len

        # Process final batch
        if batch:
            if use_ensemble:
                all_predictions.extend(predictor.predict(batch, include_member_gscores=args.save_member_gscores))
            else:
                all_predictions.extend(predictor.predict(batch))

        # Save chunk results
        results = []
        for header, seq, pred in zip(headers, sequences, all_predictions):
            if use_ensemble:
                # Ensemble mode: pred is EnsemblePrediction
                result = {
                    "label": header,
                    "sequence": seq,
                    "mean_gscore": pred.mean_gscore,
                    "mean_gscore_std": pred.mean_gscore_std,
                    "gscores": json.dumps(pred.gscores),
                    "gscores_std": json.dumps(pred.gscores_std),
                }
                if args.save_member_gscores and pred.member_gscores is not None:
                    result["member_gscores"] = json.dumps(pred.member_gscores)
                results.append(result)
            else:
                # Single mode: pred is list[float]
                results.append(
                    {
                        "label": header,
                        "sequence": seq,
                        "mean_gscore": float(np.nanmean(pred)),
                        "gscores": json.dumps(pred),
                    }
                )

        pd.DataFrame(results).to_csv(chunk_file, index=False)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Combine chunks
    print("Combining chunks...")
    chunks = [pd.read_csv(Path(f"{args.output}.{i:0{n_digits}d}")) for i in range(n_chunks)]
    pd.concat(chunks, ignore_index=True).to_csv(args.output, index=False)

    # Clean up chunk files
    for i in range(n_chunks):
        Path(f"{args.output}.{i:0{n_digits}d}").unlink()

    print(f"Results saved to {args.output}")
    if use_ensemble:
        cols = "label, sequence, mean_gscore, mean_gscore_std, gscores, gscores_std"
        if args.save_member_gscores:
            cols += ", member_gscores"
        print(f"Columns: {cols}")
        print("  - mean_gscore_std: calibrated std across ensemble members of per-member mean_gscore")
        print("  - gscores_std: calibrated per-residue std across ensemble members (JSON array)")
        if args.save_member_gscores:
            print("  - member_gscores: per-residue gscores for each ensemble member (JSON 2D array)")
    else:
        print("Columns: label, sequence, mean_gscore, gscores")
    print("(Use json.loads() to parse JSON array columns)")


if __name__ == "__main__":
    main()
