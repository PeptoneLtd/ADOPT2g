# Third-party data, models and tools

The ADOPT2g v1.0 code and its trained model are released under the MIT licence in
[LICENSE](LICENSE). They build on the public resources listed here. Please credit them when you
use or redistribute ADOPT2g v1.0, its data or its model.

## Datasets

The v1.0 release asset contains two data files:

| File | What it is |
|---|---|
| `data/trizod_2024-05-09/train.json` | The 6,943 entries of the TriZOD 2024-05-09 `tolerant` rest set |
| `data/trizod_2024-05-09/test.json` | The 348 entries of the TriZOD 2024-05-09 test set |

Each entry has an ID, a sequence and per-residue G-scores and Z-scores, taken from the TriZOD
release. The files are shared under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
the licence of the TriZOD dataset. If you redistribute them, keep the attributions below.

### TriZOD

> Haak M, Senoner T, Schlensok J, Wagemann D, Rost B. TriZOD Dataset 2024-05-09. figshare, 2024.
> [doi:10.6084/m9.figshare.25792035.v1](https://doi.org/10.6084/m9.figshare.25792035.v1)

The [TriZOD](https://github.com/MarkusHaak/trizod) software is licensed under AGPL-3.0. Its code is
not included in, imported by, or required by this branch. Only the published scores are used.

### BMRB

TriZOD computes its scores from chemical shifts in the
[Biological Magnetic Resonance Data Bank](https://bmrb.io). BMRB is a core archive of the wwPDB,
and its data are distributed under the wwPDB data policy as CC0 1.0. Attribution is not required
but is requested.

> Hoch JC, et al. Biological Magnetic Resonance Data Bank. *Nucleic Acids Res.* 2023;51(D1):D368–D376.
> [doi:10.1093/nar/gkac1050](https://doi.org/10.1093/nar/gkac1050)

## Pretrained models

### ESM-2

- `facebook/esm2_t33_650M_UR50D` provides the residue embeddings. It is downloaded from Hugging Face
  at run time and is not redistributed here.
- The sentence encoder in `data/st/57/` is fine-tuned from `facebook/esm2_t6_8M_UR50D`. Anyone
  redistributing those weights must keep the notice below.

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
