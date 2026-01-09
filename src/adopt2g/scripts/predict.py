#!/usr/bin/env python3
"""
Predict disorder scores from FASTA file using an ensemble of models.

The ensemble provides mean predictions and per-residue uncertainty estimates.

Usage:
    # Use latest trained ensemble
    adopt2g-predict --fasta input.fasta --output results.csv

    # Use specific ensemble
    adopt2g-predict --fasta input.fasta --output results.csv --ensemble-id seed883_n5
"""

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from adopt2g.data import FastaIterator
from adopt2g.ensemble import find_latest_ensemble
from adopt2g.inference import load_ensemble_predictor


def estimate_max_residues_per_batch(usage_fraction: float = 0.75) -> int:
    """Estimate max residues per GPU batch based on available memory."""
    if torch.cuda.is_available():
        free_mem_gb = torch.cuda.mem_get_info()[0] / 1e9
        # Reserve ~3GB for models, use ~usage_fraction of remaining for safety
        usable_gb = max(0.5, (free_mem_gb - 3.0) * usage_fraction)
        # Empirical: ~5KB per residue for ESM2-650M fp16 forward pass
        return int(usable_gb * 1e9 / 5_000)
    return 10_000  # default for CPU


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict disorder from FASTA using an ensemble of models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use latest trained ensemble
  adopt2g-predict --fasta proteins.fasta --output results.csv

  # Use specific ensemble
  adopt2g-predict --fasta proteins.fasta --output results.csv --ensemble-id seed883_n5

  # Save per-model gscores for each ensemble member
  adopt2g-predict --fasta proteins.fasta --output results.csv --save-member-gscores
""",
    )
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA file")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file")

    # Ensemble options
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

    # Processing options
    parser.add_argument("--chunk-size", type=int, default=10_000, help="Sequences per chunk")
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable uncertainty calibration (use raw ensemble std)",
    )
    parser.add_argument(
        "--save-member-gscores",
        action="store_true",
        help="Save per-model gscores for each ensemble member",
    )

    args = parser.parse_args()

    # Determine manifest path
    if args.ensemble_manifest:
        manifest_path = args.ensemble_manifest
    elif args.ensemble_id:
        manifest_path = args.ensemble_id
    else:
        # Find latest ensemble
        latest = find_latest_ensemble()
        if latest is None:
            print("No trained ensemble found. Train one with adopt2g-train")
            return
        manifest_path = latest

    # Load predictor with or without calibration
    calibration_factor = 1.0 if args.no_calibration else None
    predictor = load_ensemble_predictor(manifest_path, calibration_factor=calibration_factor)
    print("Running ensemble prediction (includes uncertainty estimates)")
    if args.no_calibration:
        print("  Uncertainty calibration disabled (using raw ensemble std)")
    if args.save_member_gscores:
        print("  Saving per-member gscores")

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
                all_predictions.extend(predictor.predict(batch, include_member_gscores=args.save_member_gscores))
                batch = []
                batch_residues = 0
                # Note: gc.collect() and empty_cache() removed here for performance
                # Memory cleanup happens between chunks instead

            batch.append(seq)
            batch_residues += seq_len

        # Process final batch
        if batch:
            all_predictions.extend(predictor.predict(batch, include_member_gscores=args.save_member_gscores))

        # Save chunk results
        results = []
        for header, seq, pred in zip(headers, sequences, all_predictions):
            # pred is EnsemblePrediction
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
    cols = "label, sequence, mean_gscore, mean_gscore_std, gscores, gscores_std"
    if args.save_member_gscores:
        cols += ", member_gscores"
    print(f"Columns: {cols}")
    calibrated_str = "calibrated " if not args.no_calibration else ""
    print(f"  - mean_gscore_std: {calibrated_str}std across ensemble members of per-member mean_gscore")
    print(f"  - gscores_std: {calibrated_str}per-residue std across ensemble members (JSON array)")
    if args.save_member_gscores:
        print("  - member_gscores: per-residue gscores for each ensemble member (JSON 2D array)")
    print("(Use json.loads() to parse JSON array columns)")


if __name__ == "__main__":
    main()
