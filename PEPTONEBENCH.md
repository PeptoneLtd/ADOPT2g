# ADOPT2g and PeptoneBench

[PeptoneBench](https://github.com/PeptoneLtd/peptonebench) uses **ADOPT2g v1.0**. This branch
holds v2.0, a newer version that gives different predictions. To reproduce or extend PeptoneBench,
use v1.0. For other work, use v2.0.

## What PeptoneBench takes from ADOPT2g

Only PeptoneDB-SAXS contains ADOPT2g predictions, in the columns `mean_gscore_adopt2` and
`gscores_adopt2`. They were used to choose the proteins in the dataset, by sampling across bins of
predicted mean G-score, and to place the proteins on the disorder axis of the SAXS figures. The
benchmark scores themselves do not use them.

The G-scores in PeptoneDB-CS and PeptoneDB-Integrative are not predictions. They are computed with
TriZOD from the measured chemical shifts.

## Getting v1.0

| What | Where |
|---|---|
| Code | The [`peptonebench`](https://github.com/PeptoneLtd/ADOPT2g/tree/peptonebench) branch, tag [`v1.0`](https://github.com/PeptoneLtd/ADOPT2g/tree/v1.0) |
| Weights and training data | The asset of the [v1.0 release](https://github.com/PeptoneLtd/ADOPT2g/releases/tag/v1.0) |

Running v1.0 again reproduces the PeptoneBench values closely, but not exactly. On PeptoneDB-SAXS
the per-residue Pearson correlation is 0.9994, and the mean G-score of each protein changes by at
most 0.03. The small differences come from ESM-2 running in 8-bit, where results depend slightly on
which sequences share a batch and vary slightly between runs.

## How v1.0 and v2.0 differ

Both versions predict TriZOD G-scores from the same inputs: a per-residue ESM-2 650M embedding
joined with a per-sequence embedding from a fine-tuned `esm2_t6_8M`.

| | v1.0 (PeptoneBench) | v2.0 (this branch) |
|---|---|---|
| **Trained** | 2025-07-15 | 2026-11-10 |
| **Training data** | The TriZOD 2024-05-09 release with the `tolerant` filter: 6,943 training and 348 test entries | TriZOD G-scores recomputed from a full BMRB download on 2025-12-12, clustered with MMseqs2 and cleaned against DisProt 2025_12. The test split is stratified by disorder and excludes sequences that overlap CheZOD |
| **Sequence-embedding classes** | From the mean and spread of each sequence's G-scores | From how well TriZOD and DisProt agree within each sequence cluster |
| **Head** | Linear ElasticNet fitted to logit(G), exported to ONNX | MLP with hidden layers of 64 and 32, trained with binary cross-entropy |
| **Models** | One | An ensemble of 7, with calibrated per-residue uncertainty |
| **ESM-2 precision** | 8-bit | 16-bit |
| **How to run it** | `adopt_2/examples/disorder_prediction.py` | `adopt2g-predict` |

On the 439 PeptoneDB-SAXS sequences, v2.0 agrees with v1.0 at a per-residue Pearson correlation of
0.962, with a mean absolute difference of 0.046. Few of the differences are systematic: v2.0 is
0.02 higher on average where v1.0 predicts order (G < 0.1), and 0.02 lower where v1.0 predicts
strong disorder (G > 0.9). The mean G-score of 13 of the 439 proteins changes side of 0.5. The two
versions are close but not interchangeable, so use v1.0 to reproduce PeptoneBench.

The SAXS set has no measured G-scores, so this comparison shows how the versions differ, not which
is more accurate. The v2.0 test split cannot settle that either: 109 of its 158 test
sequences are in the v1.0 training data.

## Timeline

| Date | Event |
|---|---|
| 2024-05-09 | TriZOD dataset release used to train v1.0 |
| 2025-07-15 | v1.0 trained |
| 2025-10-13 | PeptoneDB datasets published in PeptoneBench, with v1.0 predictions in PeptoneDB-SAXS |
| 2025-10-19 | PeptoneBench preprint ([10.1101/2025.10.18.680935](https://doi.org/10.1101/2025.10.18.680935)) |
| 2025-12-12 | TriZOD G-scores recomputed from BMRB for v2.0 |
| 2026-01-19 | v2.0 training data finalised |
| 2026-11-10 | v2.0 trained, and v1.0 and v2.0 released |
