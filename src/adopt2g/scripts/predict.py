#!/usr/bin/env python3
"""
Predict disorder scores from FASTA file using an ensemble of models.

The ensemble provides mean predictions and per-residue uncertainty estimates.

Usage:
    # Use latest trained ensemble
    adopt2g-predict --input input.fasta --output results.csv

    # Use specific ensemble
    adopt2g-predict --input input.fasta --output results.csv --ensemble-id seed883_n5
"""

import argparse
import gc
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from adopt2g.data import CsvIterator, FastaIterator
from adopt2g.ensemble import find_latest_ensemble
from adopt2g.inference import load_ensemble_predictor
from adopt2g.utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict disorder from FASTA using an ensemble of models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use latest trained ensemble
  adopt2g-predict --input proteins.fasta --output results.csv

  # Use specific ensemble
  adopt2g-predict --input proteins.fasta --output results.csv --ensemble-id seed883_n5

  # Save per-model gscores for each ensemble member
  adopt2g-predict --input proteins.fasta --output results.csv --save-member-gscores
""",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input FASTA or CSV file")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file")
    parser.add_argument(
        "--csv-cols",
        type=str,
        default="label,sequence",
        help="Column names for label and sequence in CSV input (default: 'label,sequence')",
    )

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
    setup_logging()

    # Determine manifest path
    if args.ensemble_manifest:
        manifest_path = args.ensemble_manifest
    elif args.ensemble_id:
        manifest_path = args.ensemble_id
    else:
        # Find latest ensemble
        latest = find_latest_ensemble()
        if latest is None:
            logger.error("No trained ensemble found. Train one with adopt2g-train")
            return
        manifest_path = latest

    # Load predictor with or without calibration
    calibration_factor = 1.0 if args.no_calibration else None
    predictor = load_ensemble_predictor(manifest_path, calibration_factor=calibration_factor)
    logger.info("Running ensemble prediction (includes uncertainty estimates)")
    if args.no_calibration:
        logger.info("  Uncertainty calibration disabled (using raw ensemble std)")
    if args.save_member_gscores:
        logger.info("  Saving per-member gscores")

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

        # Predict with automatic memory-aware batching
        all_predictions = predictor.predict(
            list(sequences),
            include_member_gscores=args.save_member_gscores,
            show_progress=True,
        )

        # Save chunk results
        results = []
        for header, seq, pred in zip(headers, sequences, all_predictions, strict=True):
            # pred is EnsemblePrediction
            result = {
                "label": header,
                "sequence": seq,
                "mean_gscore": pred.mean_gscore,
                "mean_gscore_lower": pred.mean_gscore_lower,
                "mean_gscore_upper": pred.mean_gscore_upper,
                "gscores": json.dumps(pred.gscores),
                "gscores_lower": json.dumps(pred.gscores_lower),
                "gscores_upper": json.dumps(pred.gscores_upper),
            }
            if args.save_member_gscores and pred.member_gscores is not None:
                result["member_gscores"] = json.dumps(pred.member_gscores)
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
    if args.save_member_gscores:
        cols += ", member_gscores"
    logger.info(f"Columns: {cols}")
    calibrated_str = "calibrated " if not args.no_calibration else ""
    logger.info(f"  - mean_gscore_lower/upper: {calibrated_str}confidence bounds for mean_gscore (asymmetric)")
    logger.info(f"  - gscores_lower/upper: {calibrated_str}per-residue confidence bounds (JSON arrays, asymmetric)")
    if args.save_member_gscores:
        logger.info("  - member_gscores: per-residue gscores for each ensemble member (JSON 2D array)")
    logger.info("(Use json.loads() to parse JSON array columns)")


if __name__ == "__main__":
    main()
