import argparse
import gc
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# sys.path.insert(1, '/home/mambauser/adopt2')
sys.path.insert(1, os.path.abspath("../../"))

import torch

from adopt_2.models import inference


def create_parser():
    parser = argparse.ArgumentParser(description="Disorder prediction with ADOPT2 from FASTA file")

    parser.add_argument(
        "--fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "--output_file",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
    )

    return parser


def estimate_max_residues_per_batch():
    return int(np.round(torch.cuda.mem_get_info()[0] / 60_000_000))


class FastaChunkIterator:
    def __init__(self, filename, chunk_size=1000):
        self.filename = filename
        self.chunk_size = chunk_size
        self._length = None  # for lazy calculation

    def __iter__(self):
        self.file = open(self.filename, "r")
        self._header = None
        self._sequence = []
        self.done = False
        return self

    def __next__(self):
        chunk = []
        if hasattr(self, "done") and self.done:
            self.file.close()
            raise StopIteration

        for line in self.file:
            if line.startswith(">"):
                if self._header:
                    chunk.append((self._header, "".join(self._sequence)))
                    if len(chunk) == self.chunk_size:
                        self._header = line[1:].strip()
                        self._sequence = []
                        break
                self._header = line[1:].strip()
                self._sequence = []
            else:
                self._sequence.append(line.strip())
        else:
            # end of file
            if self._header:
                chunk.append((self._header, "".join(self._sequence)))
            self.done = True
            if not chunk:
                self.file.close()
                raise StopIteration
        return chunk

    def __len__(self):
        if self._length is not None:
            return self._length
        # Count number of records (lines starting with '>')
        with open(self.filename, "r") as f:
            n_records = sum(1 for line in f if line.startswith(">"))
        self._length = int(np.ceil(n_records / self.chunk_size))
        return self._length


adopt2 = inference.AdoptTwo(inference.st_model, inference.sess)


def n_resid(batch):
    n = 0
    for s in batch:
        n += len(s)
    return n


def run(args):
    chunks = FastaChunkIterator(str(args.fasta_file), chunk_size=args.chunk_size)
    n_chunks = len(chunks)
    print(f"chunks: {n_chunks}")
    n_digits = len(str(n_chunks))
    max_residues_per_gpu_batch = estimate_max_residues_per_batch()
    print(f"estimated max residues per batch: {max_residues_per_gpu_batch}")
    for chunk_no, ff in enumerate(tqdm(chunks)):
        if os.path.isfile(f"{args.output_file}.{chunk_no:>0{n_digits}d}"):
            print(f"{args.output_file}.{chunk_no:>0{n_digits}d} already exists, skipping chunk")
            continue
        tags, seqs = list(zip(*ff))
        disorder_scores = []
        current_batch = []
        current_batch_res_count = 0
        for seq in tqdm(seqs, desc="gpu batch", leave=False):
            if current_batch_res_count + len(seq) <= max_residues_per_gpu_batch:
                current_batch.append(seq)
                current_batch_res_count += len(seq)
            elif len(seq) <= max_residues_per_gpu_batch:
                assert n_resid(current_batch) <= max_residues_per_gpu_batch
                disorder_scores += adopt2.predict_disorder(current_batch)
                current_batch = [seq]
                current_batch_res_count = len(seq)
                gc.collect()
                torch.cuda.empty_cache()
            else:
                assert n_resid(current_batch) <= max_residues_per_gpu_batch
                disorder_scores += adopt2.predict_disorder(current_batch)
                gc.collect()
                torch.cuda.empty_cache()
                disorder_scores += adopt2.predict_disorder([seq])
                current_batch = []
                current_batch_res_count = 0
                gc.collect()
                torch.cuda.empty_cache()
        if len(current_batch) > 0:
            assert n_resid(current_batch) <= max_residues_per_gpu_batch
            disorder_scores += adopt2.predict_disorder(current_batch)
            gc.collect()
            torch.cuda.empty_cache()
        assert len(seqs) == len(tags)
        assert len(seqs) == len(disorder_scores)
        res_dict = {
            i: (tag, seq, np.nanmean(scores), str(scores).replace("nan", "NaN"))
            for i, (tag, seq, scores) in enumerate(zip(tags, seqs, disorder_scores))
        }
        pd.DataFrame.from_dict(
            res_dict,
            orient="index",
            columns=["label", "sequence", "mean_gscore_adopt2", "gscores_adopt2"],
        ).to_csv(f"{args.output_file}.{chunk_no:>0{n_digits}d}", index=False)
    print("combining batches")
    pd.concat(
        [pd.read_csv(f"{args.output_file}.{i:>0{n_digits}d}") for i in range(n_chunks)],
        ignore_index=True,
    ).to_csv(args.output_file, index=False)
    for i in range(n_chunks):
        os.remove(f"{args.output_file}.{i:>0{n_digits}d}")
    print("disorder predictions are now saved")
    print("(you can use json.loads to parse the 'gscores_adopt2' array)")


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
