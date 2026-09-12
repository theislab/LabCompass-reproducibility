---
license: cc-by-4.0
tags:
  - single-cell
  - flow-cytometry
  - spectral-flow-cytometry
  - haematopoiesis
  - experimental-design
size_categories:
  - 10M<n<100M
---

# LabCompass — Spectral Flow Cytometry haematopoiesis dataset

Measurements underlying **LabCompass**, a method for generative modeling of experimental design in
single-cell data. This dataset contains Spectral Flow Cytometry (SFC) profiles of *in vitro*
haematopoietic differentiation cultures, collected over successive rounds of a closed-loop
experimental design cycle.

Each round — a **loop** — proposes new culture protocols, runs them at the bench, and measures the
resulting cells. The measurements from each loop are published here as a separate file.

- **Code and full reproduction pipeline:** <https://github.com/theislab/LabCompass>
- **Wet-lab experiments and measurements:** Göttgens Lab
- **License:** CC-BY-4.0

## ⚠️ These files are per-loop, not cumulative

`loops/loop3.h5ad` contains **only the cells measured in loop 3** — not loops 0–3 together. Models in
the paper are trained on the *accumulated* data, so a loop's training set is the concatenation of
every loop up to and including it:

```
dataset(N) = concat(dataset(N-1), loopN)
```

Concatenating them yourself is a few lines of `anndata`, but the exact chain matters (one loop
introduces new protocol axes that must be zero-filled on the earlier data — see below). The
reproduction repository ships a script that does it correctly:

```bash
git clone https://github.com/theislab/LabCompass.git
python scripts/data/build_loop_datasets.py              # downloads from this repo and builds the chain
python scripts/data/build_loop_datasets.py --variants 500k   # subsampled only: far smaller and faster
```

## Files

Every loop is published in two variants: the full measurement set, and a subsampled version
(`_500k` suffix) intended for fast iteration. The suffix is a naming convention carried over from
the source data, not a guaranteed cell count — the subsampled files vary in size.

| Loop | Full | Subsampled | Approx. size (full) |
| --- | --- | --- | --- |
| 0 (baseline) | `loops/loop0.h5ad` | `loops/loop0_500k.h5ad` | 36 GB |
| 1 | `loops/loop1.h5ad` | `loops/loop1_500k.h5ad` | 2.5 GB |
| 2 | `loops/loop2.h5ad` | `loops/loop2_500k.h5ad` | 3.9 GB |
| 2.5 | `loops/loop2p5.h5ad` | `loops/loop2p5_500k.h5ad` | 2.7 GB |
| 3 | `loops/loop3.h5ad` | `loops/loop3_500k.h5ad` | 6.3 GB |
| 4 | `loops/loop4.h5ad` | `loops/loop4_500k.h5ad` | 0.9 GB |
| 4.5 | `loops/loop4p5.h5ad` | `loops/loop4p5_500k.h5ad` | 0.5 GB |
| 5 | `loops/loop5.h5ad` | `loops/loop5_500k.h5ad` | 6.3 GB |

Loop 0 is the baseline screen and is by far the largest. The half-steps (2.5, 4.5) are follow-up
rounds within a design cycle and accumulate like any other loop, giving the chain

```
loop0 → loop1 → loop2 → loop2p5 → loop3 → loop4 → loop4p5 → loop5
```

The full set is roughly 60 GB; the subsampled set is a few GB.

## Format

Each file is an [AnnData](https://anndata.readthedocs.io/) `.h5ad` object:

- **`X`** — logicle-transformed SFC intensities: fluorescence channels and morphological scatter
  features, one row per cell.
- **`obs`** — per-cell metadata, in three groups:
  - *Acquisition:* `experiment_number`, `experiment_id`, `replicate`, `date`, `well_id`,
    `cytometer`, `cytometer_serial_no`, `count_beads`, `cell_counts`, `source_id`.
  - *Protocol axes* — the culture recipe, and the space LabCompass searches over. Cytokines and small
    molecules carry their units in the column name, e.g. `scf_[ng_ml]`, `tpo_[ng_ml]`,
    `il3_[ng_ml]`, `gm-csf_[ng_ml]`, `rhflt3l_[ng_ml]`, `ldl_[ng_ml]`, `sr1_[nm]`, `um171_[nm]`,
    `um729_[µm]`, `butyzamide_[nm]`, `retinoic_acid_[µm]`, `mtg_[µm]`, `740-yp_[µm]`, alongside
    culture conditions such as `o2_[%]` and `hydrogel_type`.
  - *Annotation:* cell-type labels, where available.

`experiment_number` identifies the physical experiment a cell came from (loop 1, for instance, spans
experiments 206–210), which makes it a convenient way to check which loops are present in a
concatenated object.

### The protocol schema grows across loops

Later loops vary axes that earlier loops never did. Loop 3 introduces `il7_[ng_ml]`,
`mcsf_[ng_ml]` and `ly_cocktail_[ul/well]`, which are absent from loops 0–2.5. When concatenating,
these must be **zero-filled on the earlier data** (they were held at zero, not missing) so both sides
share an `obs` schema. `build_loop_datasets.py` does this; a naive `anndata.concat` will silently
drop the columns instead.

## Loading

```python
import anndata as ad
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="theislab/LabCompass",
    filename="loops/loop3_500k.h5ad",
    repo_type="dataset",
)
adata = ad.read_h5ad(path)
```

## Citation

<!-- TODO: replace with the published reference before release. -->

```bibtex
@article{labcompass,
  title   = {TODO},
  author  = {Consoli, Lorenzo and Palma, Alessandro and others},
  journal = {TODO},
  year    = {TODO},
}
```
