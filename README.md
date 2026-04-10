# ADOPT-2g

Updated version of ADOPT disorder predictor, based on the TriZOD G-score.

Overview of most relevant implementation details:

### Data
- **Dataset (TriZOD + DisProt, deduplicated by clustering)**: We combine an updated TriZOD export with a recent DisProt release. DisProt's disorder content is sanity-checked and recomputed from consensus annotations. The union set is clustered with MMseqs2 to deduplicate and identify shared groups across sources (`datasets/finalize_datasets.ipynb`).
- **Cleaning nonstandard edge cases**: For shared clusters where DisProt indicates strong disorder but TriZOD suggests low disorder, the corresponding TriZOD entries are removed (treating DisProt as more reliable).
- **Train/test split (stratified on disorder)**: TriZOD test examples are selected with a fixed seed and stratified by disorder level so ordered/disordered examples are balanced; the cleaned TriZOD dataset is saved with per-residue g-scores serialized as JSON.
- **Imbalance handling (TriZOD reweighting)**: MLP training uses a weighted BCE loss and per-sample weights computed from a saturating logistic curve to rebalance ordered vs disordered regions (`src/adopt2g/data.py`, `src/adopt2g/config.py`, `src/adopt2g/training.py`).
- **SentenceTransformer training data (sequence-level supervision)**: We build `datasets/st_classes.csv` on cluster representatives from DisProt+TriZOD, label each representative into ordered/disordered/uncertain classes based on within-cluster agreement and mean disorder. The SentenceTransformer is trained with contrastive learning using cosine similarity on positive/negative pairs (`src/adopt2g/data.py`, `src/adopt2g/training.py`).
### Model
- **Model architecture (ESM2 + MLP head with bounded outputs)**: We concatenate a residue-level ESM2 embedding with a sequence-level embedding from the trained SentenceTransformer base, and predict g-scores with an MLP head ending in a sigmoid so outputs stay in \([0, 1]\) (`src/adopt2g/mlp.py`, `src/adopt2g/config.py`).
- **Other notable training details**: SentenceTransformer training uses AMP, and the MLP dataloaders handle both in-memory arrays and memory-mapped embeddings for large datasets (`src/adopt2g/training.py`).
- **Ensemble predictions and calibration**: Predictions are made using an ensemble of multiple models (trained with different random seeds). The ensemble provides mean predictions and per-residue uncertainty estimates (standard deviation across ensemble members). Calibration includes temperature scaling of the mean predictions and uncertainty rescaling to better align confidence with observed error (`src/adopt2g/scripts/run_pipeline.py`, `src/adopt2g/inference.py`).

## Installation

### Using uv

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130  # for DGX-spark
uv pip install -e ".[dev,train]"
```

### Optional: Flash Attention

Flash Attention can speed up ESM inference on supported GPUs (currently not DGX-spark):

```bash
uv pip install flash-attn --no-build-isolation
```

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
