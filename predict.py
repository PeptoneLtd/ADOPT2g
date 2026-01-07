#!/usr/bin/env python3
"""
Predict disorder scores from FASTA file.

Usage:
    python predict.py --fasta input.fasta --output results.csv
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config
from src.data import FastaIterator
from src.inference import load_predictor


def estimate_max_residues_per_batch() -> int:
    """Estimate max residues per GPU batch based on available memory."""
    if torch.cuda.is_available():
        return int(np.round(torch.cuda.mem_get_info()[0] / 60_000_000))
    return 10000  # Conservative default for CPU


def count_residues(sequences: list[str]) -> int:
    """Count total residues in a list of sequences."""
    return sum(len(s) for s in sequences)


def main():
    parser = argparse.ArgumentParser(description="Predict disorder from FASTA")
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA file")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="Model seed")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Sequences per chunk")
    args = parser.parse_args()
    
    # Load predictor
    st_path = config.DATA_PATH / "st" / str(args.seed)
    mlp_path = config.CHECKPOINTS_PATH / f"mlp_{args.seed}.pt"
    
    if not st_path.exists():
        raise FileNotFoundError(f"ST model not found at {st_path}. Run training first.")
    if not mlp_path.exists():
        raise FileNotFoundError(f"MLP model not found at {mlp_path}. Run training first.")
    
    predictor = load_predictor(str(st_path), str(mlp_path))
    
    # Process FASTA in chunks
    fasta_iter = FastaIterator(args.fasta, chunk_size=args.chunk_size)
    n_chunks = len(fasta_iter)
    n_digits = len(str(n_chunks))
    max_residues = estimate_max_residues_per_batch()
    
    print(f"Processing {n_chunks} chunks, max {max_residues} residues per batch")
    
    for chunk_no, records in enumerate(tqdm(fasta_iter, total=n_chunks)):
        chunk_file = Path(f"{args.output}.{chunk_no:0{n_digits}d}")
        
        if chunk_file.exists():
            print(f"Chunk {chunk_no} already exists, skipping")
            continue
        
        headers, sequences = zip(*records)
        
        # Process in batches that fit in GPU memory
        all_scores = []
        batch = []
        batch_residues = 0
        
        for seq in tqdm(sequences, desc="Processing", leave=False):
            seq_len = len(seq)
            
            if batch and batch_residues + seq_len > max_residues:
                # Process current batch
                all_scores.extend(predictor.predict(batch))
                batch = []
                batch_residues = 0
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            batch.append(seq)
            batch_residues += seq_len
        
        # Process final batch
        if batch:
            all_scores.extend(predictor.predict(batch))
        
        # Save chunk results
        import json
        results = []
        for header, seq, scores in zip(headers, sequences, all_scores):
            results.append({
                "label": header,
                "sequence": seq,
                "mean_gscore": np.nanmean(scores),
                "gscores": json.dumps(scores),
            })
        
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
    print("(Use json.loads() to parse the 'gscores' column)")


if __name__ == "__main__":
    main()

