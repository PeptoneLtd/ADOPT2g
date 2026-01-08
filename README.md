# ADOPT 2g

Updated version of ADOPT disorder predictor, based on the TriZOD G-score.

Overview of most relevant implementation details:

- **Dataset (TriZOD + DisProt, deduplicated by clustering)**: We use an updated TriZOD export (`trizod-2025_12_12/tolerant-dataset_new.csv`) and combine it with DisProt (release `2025_12`, downloaded from DisProt). DisProt’s `disorder_content` is sanity-checked and fixed by recomputing it from the “Structural state” consensus annotations. The union DisProt+TriZOD sequences are clustered with MMseqs2 `easy-cluster` (coverage threshold \(c=0.8\)) to deduplicate and to identify shared groups across sources (`datasets/finalize_datasets.ipynb`).
- **Cleaning nonstandard edge cases**: For shared clusters where DisProt is strongly disordered but TriZOD reports low mean g-score (e.g. `disorder_content > 0.8` and `mean_gscore < 0.5`), the corresponding TriZOD entries are removed (treating DisProt as more reliable).
- **Train/test split (stratified on disorder)**: TriZOD test examples are selected with a fixed seed and stratified by `mean_gscore >= 0.5` so ordered/disordered examples are balanced in the test set; the cleaned TriZOD dataset is saved as `datasets/cleaned_trizod_dataset.csv` (with `gscores` serialized as JSON).
- **SentenceTransformer training data (sequence-level supervision)**: We build `datasets/st_classes.csv` on *cluster representatives* from DisProt+TriZOD, label each representative into 3 classes (`0=ordered`, `1=disordered`, `2=middle/uncertain`) using thresholds on within-cluster disagreement (`std_disorder_threshold=0.25`) and mean disorder (`mean_disorder_threshold=0.5`). The SentenceTransformer is trained with contrastive learning using cosine similarity on positive/negative pairs, sampling a balanced number per class (`src/data.py`, `src/training.py`).
- **Model architecture (ESM2 + MLP head with bounded outputs)**: We concatenate a residue-level ESM2 embedding (`facebook/esm2_t33_650M_UR50D`, 1280-d) with a sequence-level embedding from the trained SentenceTransformer base (`facebook/esm2_t6_8M_UR50D`, 320-d) and predict g-scores with an MLP head (`[512, 128]`, ReLU + dropout) ending in a **sigmoid** so outputs stay in \([0, 1]\) (`src/models.py`, `src/config.py`).
- **Imbalance handling (TriZOD reweighting)**: MLP training uses a **weighted MSE** loss and per-sample weights computed with a saturating logistic curve \(w=\min(1, a / (1 + e^{-b(y-0.5)}))\) (defaults `a=16.229`, `b=10.670` determined in `datasets/visualize.ipynb`) to rebalance ordered vs disordered regions (`src/data.py`, `src/config.py`, `src/training.py`).
- **Other notable training details**: SentenceTransformer training uses AMP (`use_amp=True`), and the MLP dataloaders handle both in-memory arrays and memory-mapped embeddings (switching `num_workers` accordingly) for large datasets.
- **Ensemble predictions and uncertainty**: By default, predictions are made using an ensemble of multiple models (trained with different random seeds). The ensemble provides mean predictions and per-residue uncertainty estimates (standard deviation across ensemble members). Uncertainty estimates are calibrated on training data to better reflect prediction confidence (`run_pipeline.py`, `src/inference.py`).

## Training
```
python run_pipeline.py
```

## Inference
```
python predict.py --fasta my.fasta --out test.csv
```
Use `--no-calibration` to output vanilla std as uncertainties. 
See also `data/plots.ipynb`
