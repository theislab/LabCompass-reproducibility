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
- **Contents:** per-loop measurements in `loops/`, trained models in `checkpoints/`
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

## Model checkpoints

`checkpoints/` holds the trained models behind the paper's designs, one folder per loop:

| Folder | Size | Forward model | Config group |
| --- | --- | --- | --- |
| `checkpoints/loop0/` | 1.7 GB | `likely-donkey-20` | `paths=loop0` |
| `checkpoints/loop1/` | 1.9 GB | `eager-feather-1` | `paths=loop1` |
| `checkpoints/loop2/` | 2.1 GB | `fresh-bee-21` | `paths=loop2_replicate` |
| `checkpoints/loop2p5/` | 2.3 GB | `celestial-fire-40` | `paths=loop2p5_replicate` |
| `checkpoints/loop3/` | 2.6 GB | `rich-sunset-44` | `paths=loop3_replicate` |
| `checkpoints/loop4/` | 2.6 GB | `fast-gorge-13` | `paths=loop4_replicate` |

**Loop *N*'s models are the ones that generated the designs executed in loop *N+1*.** So to reproduce
the candidates that were run at the bench in loop 1, use `checkpoints/loop0/`.

Each folder contains the four models the pipeline needs:

- **`*_FlowMatching.pkl`** — the forward model: predicts the cell-state distribution a protocol induces.
- **`*_TargetPredictionModel.pkl`** — the cell-type classifier, i.e. the phenotypic readout that the
  inverse objective is defined against.
- **`*_FlowMatchingWithScore.pkl`** — the generative prior over protocols.
- **`*_FlowMap.pkl`** — a distilled few-step version of that prior.

File names are the Weights & Biases run names, unchanged from training. `checkpoints/manifest.json`
records, for every file, which `paths` config group refers to it, its original size, and a sha256.

### Three things to know before using them

**They are inference-only.** The training and validation data that the original checkpoints carried
inside them has been removed — that is why a 113 GB file is 1 GB here. Everything inference touches
is intact (network weights bitwise unchanged, normalisation parameters, cell-type labels, the
condition key), and every checkpoint was verified tensor-by-tensor against its original and exercised
end-to-end through the pipeline. But you **cannot retrain a prior from these**: the scripts that do
so read the forward model's embedded training set, which is gone. Retrain the forward model from the
`loops/` data instead.

**They are CPU-resident.** The tensors load on any machine, with or without a GPU; move the model to
your device as you would any PyTorch module. The original checkpoints held CUDA tensors and could
only be loaded on a GPU node.

**Loops 3 and 4 use the expanded design space.** They were trained after M-CSF and the lymphoid
cocktail were added, so they expect the wider protocol vector and will fail with a shape mismatch if
you load them with the earlier annotation. Use `annotation=bloodplus_loop3` for those two; the
earlier loops use the default.

### Loading

```python
from huggingface_hub import hf_hub_download
from labcompass.models import FlowMatching

path = hf_hub_download(
    repo_id="theislab/LabCompass",
    filename="checkpoints/loop3/rich-sunset-44_FlowMatching.pkl",
    repo_type="dataset",
)
model = FlowMatching.load(path)
```

In the reproduction repository these are wired up through the `paths` config group, so pointing a run
at a downloaded loop is a matter of overriding the four checkpoint paths.

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
