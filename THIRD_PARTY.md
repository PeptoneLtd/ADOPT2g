# Third-party data, models and tools

The ADOPT2g code and its trained models are released under the MIT licence in [LICENSE](LICENSE).
They build on the public resources listed here. Please credit them when you use or redistribute
ADOPT2g, its datasets or its models. ADOPT2g v1.0, on the `peptonebench` branch, has its own
`THIRD_PARTY.md`.

## Datasets

The two files in `datasets/` are derived from public data:

| File | Derived from |
|---|---|
| `cleaned_trizod_dataset.csv` | TriZOD G-scores computed from BMRB chemical shifts |
| `st_classes.csv` | the same TriZOD entries, plus DisProt sequences and consensus disorder annotations |

These derived files are shared under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
If you redistribute them, keep the attributions below. How the files were built is in
`datasets/create_datasets.ipynb`.

### BMRB

The chemical shifts come from the [Biological Magnetic Resonance Data Bank](https://bmrb.io). BMRB
is a core archive of the wwPDB, and its data are distributed under the wwPDB data policy as CC0 1.0.
Attribution is not required but is requested.

> Hoch JC, et al. Biological Magnetic Resonance Data Bank. *Nucleic Acids Res.* 2023;51(D1):D368–D376.
> [doi:10.1093/nar/gkac1050](https://doi.org/10.1093/nar/gkac1050)

### TriZOD

Per-residue G-scores were computed with [TriZOD](https://github.com/MarkusHaak/trizod), with its
`tolerant` filter defaults, from a full BMRB download. The output is dated 2025-12-12. The run used
the fork at [invemichele-peptone/trizod](https://github.com/invemichele-peptone/trizod). Entries were then
filtered, deduplicated by clustering and cleaned against DisProt, as described in the README.

TriZOD is licensed under AGPL-3.0. Its code is not included in, imported by, or required by this
repository. Only the scores it produced are used.

> Haak M, Senoner T, Schlensok J, Wagemann D, Rost B. TriZOD Dataset 2024-05-09. figshare, 2024.
> [doi:10.6084/m9.figshare.25792035.v1](https://doi.org/10.6084/m9.figshare.25792035.v1)

> Schlensok J, Wagemann D, Senoner T, Haak M, Rost B. UdonPred: untangling protein intrinsic
> disorder prediction. *bioRxiv.* 2026.
> [doi:10.64898/2026.01.26.701679](https://doi.org/10.64898/2026.01.26.701679)

### DisProt

Sequences and consensus structural-state annotations come from [DisProt](https://disprot.org),
release 2025_12 (`DisProt-release_2025_12-with_ambiguous_evidences.json`), licensed under
CC BY 4.0. DisProt's sequences originate from UniProtKB.

Changes made: entries were filtered by length, disorder content was recomputed from the consensus
regions, sequences were clustered together with the TriZOD entries, and each cluster representative
was given an ordered, disordered or uncertain class label.

> Nugnes MV, et al. DisProt in 2026: enhancing intrinsically disordered proteins accessibility,
> deposition, and annotation. *Nucleic Acids Res.* 2026;54(D1):D383–D392.
> [doi:10.1093/nar/gkaf1175](https://doi.org/10.1093/nar/gkaf1175)

### CheZOD, used but not redistributed

The CheZOD 1325 sequence set (`CheZOD1325.tar.gz` from
[protein-nmr/CheZOD](https://github.com/protein-nmr/CheZOD)) was used only to keep sequences that
overlap with it out of the test split. No CheZOD data is included in this repository.

> Dass R, Mulder FAA, Nielsen JT. ODiNPred: comprehensive prediction of protein order and disorder.
> *Sci Rep.* 2020;10:14780. [doi:10.1038/s41598-020-71716-1](https://doi.org/10.1038/s41598-020-71716-1)

## Pretrained models

### ESM-2

- `facebook/esm2_t33_650M_UR50D` provides the residue embeddings. It is downloaded from Hugging Face
  at run time and is not redistributed here.
- The sentence encoder of each ensemble member (`member_*/st/`) is fine-tuned from
  `facebook/esm2_t6_8M_UR50D`. Anyone redistributing those weights must keep the notice below.

> Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language
> model. *Science.* 2023;379(6637):1123–1130.
> [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)

```text
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Tools used to build the datasets

- **MMseqs2** clustered the sequences. It is licensed under GPL-3.0 and was run from its Docker image,
  so none of its code is included here.

  > Steinegger M, Söding J. MMseqs2 enables sensitive protein sequence searching for the analysis
  > of massive data sets. *Nat Biotechnol.* 2017;35(11):1026–1028.
  > [doi:10.1038/nbt.3988](https://doi.org/10.1038/nbt.3988)

- **TriZOD** computed the G-scores, as described above.
