# ADOPT2g

Updated version of the [ADOPT](https://github.com/PeptoneLtd/ADOPT) disorder predictor, based on the TriZOD G-score.
This branch is ADOPT2g v2.0. [PeptoneBench](https://github.com/PeptoneLtd/peptonebench) used the earlier v1.0, which is
kept on the [`peptonebench`](https://github.com/PeptoneLtd/ADOPT2g/tree/peptonebench) branch. See
[PEPTONEBENCH.md](PEPTONEBENCH.md) for how the two versions differ.

Overview of most relevant implementation details:

### Data

- **Dataset (TriZOD + DisProt, deduplicated by clustering)**: We combine an updated TriZOD export with a recent DisProt release. DisProt's disorder content is sanity-checked and recomputed from consensus annotations. The union set is clustered with MMseqs2 to deduplicate and identify shared groups across sources (`datasets/create_datasets.ipynb`).
- **Cleaning nonstandard edge cases**: For shared clusters where DisProt indicates strong disorder but TriZOD suggests low disorder, the corresponding TriZOD entries are removed (treating DisProt as more reliable).
- **Train/test split (stratified on disorder)**: TriZOD test examples are selected with a fixed seed and stratified by disorder level so ordered/disordered examples are balanced; the cleaned TriZOD dataset is saved with per-residue g-scores serialized as JSON.
- **Imbalance handling (TriZOD reweighting)**: MLP training uses a weighted BCE loss and per-sample weights computed from a saturating logistic curve to rebalance ordered vs disordered regions (`src/adopt2g/data.py`, `src/adopt2g/config.py`, `src/adopt2g/training.py`).
- **SentenceTransformer training data (sequence-level supervision)**: We build `datasets/st_classes.csv` on cluster representatives from DisProt+TriZOD, label each representative into ordered/disordered/uncertain classes based on within-cluster agreement and mean disorder. The SentenceTransformer is trained with contrastive learning using cosine similarity on positive/negative pairs (`src/adopt2g/data.py`, `src/adopt2g/training.py`).

### Model

- **Model architecture (ESM2 + MLP head with bounded outputs)**: We concatenate a residue-level ESM2 embedding with a sequence-level embedding from the trained SentenceTransformer base, and predict g-scores with an MLP head ending in a sigmoid so outputs stay in \([0, 1]\) (`src/adopt2g/mlp.py`, `src/adopt2g/config.py`).
- **Other notable training details**: SentenceTransformer (see also [ADOPT2](https://github.com/PeptoneLtd/contrastive-finetuning-plms)) training uses AMP, and the MLP dataloaders handle both in-memory arrays and memory-mapped embeddings for large datasets (`src/adopt2g/training.py`).
- **Ensemble predictions and calibration**: Predictions are made using an ensemble of multiple models (trained with different random seeds). The ensemble provides mean predictions and per-residue uncertainty estimates (standard deviation across ensemble members). Calibration includes temperature scaling of the mean predictions and uncertainty rescaling to better align confidence with observed error (`src/adopt2g/scripts/run_pipeline.py`, `src/adopt2g/inference.py`).

## Installation

### Using uv

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,train]"
```

For inference only, `uv pip install -e .` is enough. The `train` extra adds Weights & Biases logging,
and `dev` adds Jupyter, plotting and linting tools.

### Trained models

The trained v2.0 ensemble is published as an asset of the
[latest release](https://github.com/PeptoneLtd/ADOPT2g/releases/latest). Download it, check it and
unpack it at the repository root:

```bash
curl -LO https://github.com/PeptoneLtd/ADOPT2g/releases/latest/download/adopt2g-models.tar.gz
curl -LO https://github.com/PeptoneLtd/ADOPT2g/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf adopt2g-models.tar.gz
```

This creates `models/<ensemble_id>/`, which `adopt2g-predict` picks up by default. The models are
released under the MIT licence, and the archive includes the licence and third-party notices. On first
use, ESM-2 650M (about 2.6 GB) is downloaded from Hugging Face.

### Optional: Flash Attention

Flash Attention 2 can speed up ESM-2 inference on supported GPUs. It is not installed by default,
because it is compiled against your PyTorch and CUDA versions. Install it separately:

```bash
uv pip install flash-attn --no-build-isolation
```

ADOPT2g uses it automatically when it is available on a GPU, and otherwise falls back to PyTorch's
built-in attention.

## Training

```bash
adopt2g-train
```

Or run the script directly:

```bash
python src/adopt2g/scripts/run_pipeline.py
```

## Inference

```bash
adopt2g-predict --input my.fasta --output results.csv
```

Or run the script directly:

```bash
python src/adopt2g/scripts/predict.py --input my.fasta --output results.csv
```

Use `--no-calibration` to output raw ensemble standard deviation as uncertainty.

## CatBoost Baseline

A CatBoost-based baseline is provided for comparison with the main ensemble approach. It uses ESM-2 embeddings and provides built-in uncertainty estimation using the `RMSEWithUncertainty` loss.

- **Train**: `adopt2g-catboost train --output models/catboost`
- **Search**: `adopt2g-catboost search --output models/catboost_search --n-trials 20`
- **Predict**: `adopt2g-catboost predict --input proteins.fasta --output results.csv --model models/catboost`

See also `notebooks/plots_train.ipynb` and `notebooks/plots_test.ipynb`.

## Citation

If you use ADOPT2g in your research, please cite:

```bibtex
@article{peptone2025,
  title     = {Advancing Protein Ensemble Predictions Across the Order-Disorder Continuum},
  author    = {Invernizzi, Michele and Bottaro, Sandro and Streit, Julian O and Trentini, Bruno and Venanzi, Niccolo AE and Reidenbach, Danny and Lee, Youhan and Dallago, Christian and Sirelkhatim, Hassan and Jing, Bowen and Airoldi, Fabio and Lindorff-Larsen, Kresten and Fisicaro, Carlo and Tamiola, Kamil},
  year      = 2025,
  journal   = {bioRxiv},
  publisher = {Cold Spring Harbor Laboratory},
  doi       = {10.1101/2025.10.18.680935},
  url       = {https://www.biorxiv.org/content/early/2025/10/19/2025.10.18.680935}
}
```

## License

The code and trained models are released under the MIT licence, see [LICENSE](LICENSE).
The derived datasets in `datasets/` are shared under CC BY 4.0.

ADOPT2g builds on data from BMRB, TriZOD and DisProt, and on the ESM-2 protein language models.
See [THIRD_PARTY.md](THIRD_PARTY.md) for their licences, credits and citations.
