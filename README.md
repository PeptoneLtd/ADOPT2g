# ADOPT2g v1.0

This branch holds ADOPT2g v1.0, the version used to compute the predicted G-scores in
[PeptoneBench](https://github.com/PeptoneLtd/peptonebench). It is kept so those values can be
reproduced. For new work, use ADOPT2g v2.0 on the
[main branch](https://github.com/PeptoneLtd/ADOPT2g).

ADOPT2g v1.0 predicts a per-residue [TriZOD](https://github.com/MarkusHaak/trizod) G-score, from 0
(ordered) to 1 (disordered). It keeps the design of the original ADOPT2, but is trained on TriZOD
G-scores instead of CheZOD Z-scores:

- **Residue embedding:** ESM-2 650M (`facebook/esm2_t33_650M_UR50D`), 1,280 values per residue,
  computed with the model loaded in 8-bit.
- **Sequence embedding:** `facebook/esm2_t6_8M_UR50D`, fine-tuned with a contrastive loss to
  separate ordered, disordered and uncertain sequences, 320 values per sequence.
- **Head:** a linear ElasticNet model on the 1,600 joined values, fitted to logit(G) and exported to
  ONNX. A sigmoid turns its output into a G-score.

The model was trained on 2025-07-15 on the TriZOD 2024-05-09 release with the `tolerant` filter:
the 6,943 entries of its rest set for training and the 348 entries of its test set for testing.
Its seed is 57, which is why the weights are called `57`.

[PEPTONEBENCH.md](https://github.com/PeptoneLtd/ADOPT2g/blob/main/PEPTONEBENCH.md) on the main
branch explains how v1.0 and v2.0 differ.

## Installation

`env.yaml` is the environment the model was built and served with: Python 3.10, PyTorch 1.13.1
with CUDA 11.6, transformers 4.31.0, sentence-transformers 2.2.2, bitsandbytes 0.41.2 and
onnxruntime 1.15.1.

```bash
conda env create -n adopt2g-v1 -f env.yaml
conda activate adopt2g-v1
```

Prediction also works with current versions. `requirements.txt` pins the versions it was tested
with: Python 3.12, PyTorch 2.15, transformers 5.22, sentence-transformers 6.2, bitsandbytes 0.50,
onnxruntime 1.31 and all their dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Linux, the pinned PyTorch 2.15 uses CUDA 13, which needs an NVIDIA driver of version 580 or
newer.

Prediction needs a CUDA GPU, because ESM-2 is loaded in 8-bit with bitsandbytes. The head runs on
the CPU. onnxruntime warns that `TensorrtExecutionProvider` is not available, and this warning can
be ignored.

## Model weights

The weights and the training data are published as an asset of the
[v1.0 release](https://github.com/PeptoneLtd/ADOPT2g/releases/tag/v1.0). Download them, check them
and unpack them at the repository root:

```bash
curl -LO https://github.com/PeptoneLtd/ADOPT2g/releases/download/v1.0/adopt2g-v1.0-models.tar.gz
curl -LO https://github.com/PeptoneLtd/ADOPT2g/releases/download/v1.0/SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf adopt2g-v1.0-models.tar.gz
```

This creates:

- `data/st/57/`, the fine-tuned sentence encoder
- `data/onnx/57.onnx`, the prediction head
- `data/trizod_2024-05-09/train.json` and `test.json`, the training and test data
- `data/LICENSE` and `data/THIRD_PARTY.md`

On first use, ESM-2 650M (about 2.6 GB) is downloaded from Hugging Face.

## Prediction

Run the script from `adopt_2/examples/`, because the code finds `data/` relative to the working
directory:

```bash
cd adopt_2/examples
python disorder_prediction.py --fasta_file my.fasta --output_file my_gscores.csv
```

The output has the columns `label`, `sequence`, `mean_gscore_adopt2` and `gscores_adopt2`, the
same columns as `PeptoneDB-SAXS.csv` in PeptoneBench. `gscores_adopt2` is a list of per-residue
scores that `json.loads` can read. Residues other than the 20 standard amino acids get `NaN`.
Sequences longer than 1,024 residues are split into chunks that overlap by about half, and the
scores are averaged where chunks overlap.

### Reproducing PeptoneBench

PeptoneBench's values were computed with these weights, with ESM-2 in 8-bit, in batches of 16.
In 8-bit, the handling of outlier values depends on which sequences share a batch, so a rerun does
not give exactly the same numbers.

On the 439 sequences of PeptoneDB-SAXS, running `disorder_prediction.py` with the released weights
and the versions in `requirements.txt` agrees with the published values as follows:

- per-residue Pearson correlation 0.9994 and mean absolute difference 0.006
- per-protein mean G-score: mean absolute difference 0.002
- no protein's mean G-score moves to the other side of 0.5

## Training

1. Compute the ESM-2 embeddings. As written, this runs on the CPU in fp32 and writes
   `data/esm2/train/` and `data/esm2/test/`:

   ```bash
   cd data
   python esm2.py
   ```

2. Train the sentence encoder and the head:

   ```bash
   cd adopt_2/examples
   python model_search.py
   ```

   This writes `data/st/57/` and `data/onnx/57.onnx`, replacing the downloaded weights. Training
   additionally needs scikit-learn and skl2onnx.

`model_search.py` is the script that produced the released weights. Training was not rerun for this
release, and other library versions may give slightly different weights.

## Changes from the original code

The code is Peptone's original code for this model, with one later commit so that it runs with
current library versions:

- ESM-2 is loaded with `add_pooling_layer=False` and a `BitsAndBytesConfig`, which current
  transformers needs for 8-bit loading. The pooling layer was never used.
- The tokenised sequences are moved to the GPU explicitly. Older transformers did this
  automatically.
- The maximum chunk length is read from `config.max_position_embeddings`, which gives the same
  value as before.
- `model_search.py` loads the saved embeddings with `weights_only=False`, which current PyTorch
  needs.

None of these changes the predictions.

## Licence

The code and the trained model are released under the MIT licence, see [LICENSE](LICENSE). The
training data are shared under CC BY 4.0. ADOPT2g v1.0 builds on TriZOD, BMRB and ESM-2. See
[THIRD_PARTY.md](THIRD_PARTY.md) for their licences, credits and citations.

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
