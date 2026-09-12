# LabCompass — reproducibility repository

Official reproducibility repository for **LabCompass**, a method for generative modeling of
experimental design in single-cell data. It contains the complete pipeline behind the paper's
results: every training run, design sweep and analysis, from raw measurements to published figure.

The method is applied here to a closed-loop experimental design cycle for *in vitro* haematopoietic
differentiation, measured by **Spectral Flow Cytometry (SFC)**. A generative **forward model** learns
how a culture protocol shapes the resulting cell-state distribution; a **prior** learns which
protocols are plausible in the first place; and **loss-guided inverse design** then searches protocol
space for recipes that push the culture toward a desired cell type. The top candidates are executed
in the wet lab, the new measurements are folded back into the training data, and the cycle repeats —
each turn of that cycle is a **loop**.

The method itself lives in the [`labcompass`](https://github.com/theislab/LabCompass) package; this
repository is the experimental record of applying it. Wet-lab experiments and SFC measurements were
performed by the Göttgens Lab.

---

## Repository layout

| Path | Contents |
| --- | --- |
| `scripts/` | Entry points. `data/`, `forward/`, `prior/`, `inverse/`, `miscellaneous/`. |
| `forward/` | Hydra configs for the forward model (`conditional_model/flow_matching/config/`) and the cell-type classifier (`cell_type_classification/config/`), plus their notebooks. |
| `prior/` | Hydra configs and notebooks for the protocol prior and flow-map distillation. |
| `inverse/loss_guidance/` | Hydra configs and notebooks for inverse design. |
| `sbatch/sbatch_launchers/` | SLURM job scripts, one per stage. |
| `sbatch/submitters/` | Shell loops that `sbatch` a whole grid of jobs. |
| `shared_utils/` | Shared helpers imported by scripts and notebooks (`experiment_utils.py`, `data_utils.py`, `plot_utils.py`, …). |
| `data_notebooks/` | Per-loop analysis of the experimental results, and the figures built from them. |
| `misc/` | Side analyses (e.g. subpopulation distances). |
| `project_folder` | **Symlink** to the shared data/output directory. Not tracked by git. |

### Two conventions worth knowing before you start

**1. `project_folder` is a symlink, and every path goes through it.**
All data, checkpoints and outputs live outside the repository, in a shared workspace. The repo
reaches them through a `project_folder` symlink at its root, so nothing in the code depends on
anyone's home directory. Create it once, pointing wherever your copy of the data lives:

```bash
ln -s /path/to/your/SFC_workspace project_folder
```

It is listed in `.gitignore`; the link is yours, the layout underneath it is shared.

**2. `REPO_ROOT` resolves config paths.**
Hydra configs refer to data and checkpoints as `${oc.env:REPO_ROOT}/project_folder/...`. Every
entry-point script sets `os.environ["REPO_ROOT"]` from its own file location before Hydra composes
the config, so running a script from any working directory just works. **Notebooks must do the same
thing themselves** — each notebook that composes a config sets

```python
os.environ["REPO_ROOT"] = os.path.abspath("../../..")   # depth depends on the notebook
```

before calling `hydra.initialize()`. If you add a notebook and get
`InterpolationResolutionError: Environment variable 'REPO_ROOT' not found`, this line is what's missing.

---

## Setup

### Dependencies

Two packages must be installed from source:

| Package | Repository | Role |
| --- | --- | --- |
| `labcompass` | <https://github.com/theislab/LabCompass> | **The method.** Models (`FlowMatching`, `FlowMatchingWithScore`, `FlowMap`, `TargetPredictionModel`), data loaders, training loops. |
| `scopt` | <https://github.com/theislab/scOpt> | Optimisation methods used by the SGD baseline. |

```bash
conda create -n labcompass python=3.12 && conda activate labcompass
git clone https://github.com/theislab/LabCompass.git && pip install -e LabCompass
git clone https://github.com/theislab/scOpt.git      && pip install -e scOpt
```

The reference environment used for the paper runs Python 3.12.9 with `torch==2.10.0`,
`anndata==0.12.10`, `scanpy==1.12`, `hydra-core==1.3.2`, `omegaconf==2.3.0`, `cloudpickle==3.1.2`,
`numpy==2.3.5`, `pandas==2.3.3`, `scikit-learn==1.8.0`, `wandb==0.25.0`, `huggingface-hub==1.31.0`,
and — for the sweeps — `optuna==2.10.1`, `hydra-optuna-sweeper==1.2.0`, `hydra-submitit-launcher==1.2.0`.

> **Note.** This repository does not yet ship an `environment.yml` / `requirements.txt`. Pinning the
> versions above into one is the remaining step for a fully reproducible setup.

### Weights & Biases

Training runs log to W&B (entity `haematopoiesis-cambridge`). Run `wandb login`, or set
`WANDB_MODE=offline` to disable. **The W&B run name determines the checkpoint file name**, so note it
when a job finishes — you will need it to point the next stage at the right file.

### A note on old checkpoints

The `labcompass` package was previously named `sc_exp_design`. Checkpoints are `cloudpickle` files,
which record the module path of every class they contain, so checkpoints written before the rename
refer to a module that no longer exists. `BaseModel.load` installs a compatibility shim that
transparently redirects `sc_exp_design.*` imports to `labcompass.*`, so old checkpoints still load —
no conversion needed.

---

## The pipeline

Each stage below consumes the artefacts of the previous one. The config key that carries each
dependency is named explicitly, because **the checkpoint paths shipped in the `paths` configs point
at specific dated runs** — you will need to override them with the paths your own runs produced.

Stages are shown both as a direct `python` invocation (easiest to debug) and as the SLURM wrapper
(what you actually want on the cluster).

> The scripts themselves are cwd-independent (they resolve `REPO_ROOT` from their own location), but
> the `project_folder/...` arguments written literally in the examples below are plain relative paths.
> **Run the examples from the repository root**, or pass absolute paths instead.

### Step 0 — Build the datasets

The measurements are published per loop on the Hub at
[**theislab/LabCompass**](https://huggingface.co/datasets/theislab/LabCompass), one `h5ad` per loop
(`loops/loop3.h5ad`) plus a 500k-cell subsample of each (`loops/loop3_500k.h5ad`). Those files hold
only the cells measured *in that loop*; models train on the cumulative data, so loop *N*'s dataset is
`concat(dataset(N-1), loop N)`. `scripts/data/build_loop_datasets.py` replays that chain and writes a
`project_folder`-shaped tree:

```bash
# Download from the Hub and build every loop
python scripts/data/build_loop_datasets.py

# …or build from a local copy of the raw files
python scripts/data/build_loop_datasets.py --source-dir /path/to/raw

# Force specific loops to rebuild (e.g. after a new loop is published)
python scripts/data/build_loop_datasets.py --loops loop5

# Only the 500k subsamples — much faster, enough for most development
python scripts/data/build_loop_datasets.py --variants 500k
```

Loops whose output already exists are skipped and simply read back to continue the chain, so
re-running is cheap. **Budget the disk and time for a first full build**: the raw full-resolution
files total roughly 60 GB (loop 0 alone is 36 GB), and each cumulative dataset is written out in
addition. `--variants 500k` keeps the whole thing in the low tens of GB.

Which files to fetch, how they chain, and where they land is declared in
`scripts/data/loop_manifest.yaml` — **edit that file, not the script**, if the layout on the Hub
changes. `scripts/data/hf_dataset_card.md` is the companion dataset card published on the Hub; keep
the two in sync when the layout changes. Output:

```
project_folder/output/loops/data/
├── loop0/                  adata_{500k,full}.h5ad + *_residual.h5ad
├── loop1/                  …
├── loop2/replicate/        …
├── loop2p5/replicate/      …
├── loop3/replicate/        …
├── loop4/replicate/        …
├── loop4p5/replicate/      …
└── loop5/                  …
```

Each loop directory holds four files: the cells measured in that loop (`*_residual.h5ad`) and the
cumulative dataset (`adata_*.h5ad`), in a full version and a 500k-cell subsample used for fast
iteration. Loop 0 is the baseline, so its residual and cumulative files are identical.

The chain is `loop0 → loop1 → loop2 → loop2p5 → loop3 → loop4 → loop4p5 → loop5`. The half-steps
(`2p5`, `4p5`) are follow-up rounds within a design cycle and accumulate like any other loop.

This is the scripted equivalent of `data_notebooks/concatenate_loop_data.ipynb`, which remains as the
exploratory record of how the chain was assembled.

### Step 1 — Train the forward models

Two independent models. Both read a loop dataset and write a `.pkl` into `paths.dump_dir`.

**(a) Conditional flow matching** — the perturbation-response model. Maps noise to SFC cell states
conditioned on the protocol vector.

```bash
python scripts/forward/train/train_flow_matching.py \
    paths=loop3_replicate \
    data.sample_rep="X_channel_standardized+X_scatter_standardized"

sbatch sbatch/sbatch_launchers/forward/train/train_cfm.sbatch -sr <sample_rep> -p <protocol_axis>
```

Config root `forward/conditional_model/flow_matching/config/train_cfm.yaml`; groups `annotation`,
`callbacks`, `data`, `flow_matching`, `model`, `paths`, `split`, `training`, `transforms`, `vf`.
Key knobs: `paths.h5ad_path` (input), `split.mode` (`ood` / `in-distribution`) with
`split.unique_value_ids` for held-out protocols, `training.num_training_steps`.
→ **`<wandb_run_name>_FlowMatching.pkl`** in `paths.dump_dir`.

**(b) Cell-type classifier** — the `TargetPredictionModel` that the inverse objective steers.

```bash
python scripts/forward/train/train_target_prediction_model.py paths=hpc_subset
sbatch sbatch/sbatch_launchers/forward/train/train_target_prediction_model.sbatch <sample_rep>
```

Config root `forward/cell_type_classification/config/train_classifier.yaml`. Optional class
rebalancing via `training.use_class_weights` / `training.class_weights_obs_col`.
→ **`<wandb_run_name>_TargetPredictionModel.pkl`** in `paths.dump_dir`.

Hyperparameter sweeps (Optuna via `--multirun`) live in `sbatch/sbatch_launchers/forward/sweep/`:
`sweep_cfm.sbatch`, `sweep_cfm_loops.sbatch` (adds `-l|--loop` to pick the loop dataset), and
`sweep_target_prediction_model.sbatch`.

### Step 2 — Train the prior

The prior is generative **over protocols**, not over cells. It is built from the conditions stored
inside the forward model, so the **forward checkpoint from Step 1 is a hard prerequisite**
(`paths.forward_model_checkpoint_path`).

```bash
python scripts/prior/train/train_flow_matching_prior.py \
    paths=hpc_full \
    paths.forward_model_checkpoint_path=<...>_FlowMatching.pkl

sbatch sbatch/sbatch_launchers/prior/train_prior.sbatch paths=hpc_full
```

→ **`<wandb_run_name>_FlowMatchingWithScore.pkl`** in `paths.dump_dir`.

*Optional:* distil a few-step `FlowMap` student for faster inverse sampling. It needs **both** the
forward checkpoint and the Step-2 prior as teacher:

```bash
python scripts/prior/train/distill_flow_map_prior.py \
    paths=hpc_full_flow_map \
    paths.teacher_checkpoint_path=<...>_FlowMatchingWithScore.pkl
```

> Several `prior/config/paths/*_flow_map.yaml` ship with an **empty `teacher_checkpoint_path`** —
> always pass it explicitly.

The sbatch wrappers `prior/train_prior.sbatch` and `prior/train_flow_map.sbatch` forward `"$@"` to
Hydra but **append `annotation=loop3`**, which overrides any `annotation=` you pass.
`flow_maps/train_flow_maps.sbatch` is the same distillation without that forced override.

### Step 3 — Inverse design

Samples candidate protocols by guiding the prior's generative trajectory with a loss that rewards the
target cell type under the forward model + classifier.

Required inputs, all in the `paths` group:

| Key | Produced by |
| --- | --- |
| `paths.perturbation_prediction_path` | Step 1a (`FlowMatching`) |
| `paths.ct_classifier_path` | Step 1b (`TargetPredictionModel`) |
| `paths.prior_flow_path` | Step 2 (`FlowMatchingWithScore`) |
| `paths.prior_flow_map_path` | Step 2 distillation (optional; required for marker optimisation) |

```bash
python scripts/inverse/experiment/run_inverse_experiment.py \
    paths=bloodplus \
    sampling.target_cell_type='"EryPro"' \
    paths.dump_dir=project_folder/output/inverse/loss_guidance/my_run/…
```

Variants: `run_inverse_experiment.py` (unconstrained), `run_inverse_experiment_penalized.py` (box
constraints from `constraints.lbound_dict` / `ubound_dict` with a warm-up penalty schedule),
`run_sgd_baseline.py` (no prior — plain gradient flow in protocol space),
`run_inverse_experiment_marker_opt.py` (optimises marker levels instead of a cell type).

In practice you launch a **grid**, not a single run:

```bash
bash sbatch/submitters/launch_sweep_inverse.sh -e <env> -p bloodplus -d my_run_name
```

which sweeps target cell types × schedulers × optimisation types × query types, submitting
`sweep_inverse.sbatch` for each. That script accepts:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-t, --target-ct` | `HSCs` | Target cell type. |
| `-s, --scheduler` | `reciprocal` | Guidance-strength schedule: `constant`, `exp-decay`, `reciprocal`, `lin-decay`. |
| `-o, --optimization-type` | `unconstrained` | `unconstrained`, `penalized_oxy_days`, `penalized_all_axes`. |
| `-q, --query-type` | `pure_populations` | `pure_populations`, `custom_populations`, `custom_populations_masked`. |
| `-n, --n-samples` | `50` | Candidate protocols to sample. |
| `-nf, --n-fwd-samples` | `350` | Forward passes per candidate. |
| `-ctc, --ct-column` | `cell_type_leiden` | Cell-type annotation column. |
| `-d, --dump-name` | `bloodplus_inverse_fm_loop3` | Output subfolder under `output/inverse/loss_guidance/`. |
| `-e, --env-name` | `sc_exp_design` | Conda environment to run in. |
| `-p, --paths` | `default` | `paths` config group selecting the checkpoint set. |

> `sweep_inverse.sbatch` force-appends `annotation=bloodplus_loop3` to the Hydra command and derives
> `constraints` from `-o`, so passing either as an override has no effect — edit the script if you
> need different values.

Each run writes `<dump_dir>/<target_cell_type>/<timestamp>_<uuid>/` (with `/` in a cell-type name
replaced by `:`, so `GMP-Neutro/CD16-Mono` becomes `GMP-Neutro:CD16-Mono/`):

```
config.yaml                 # exact resolved config — the provenance record
candidates.csv              # the designed protocols + loss + predicted target proportion
inverse_results.npz         # guidance trajectory, loss/lambda history, noise
fwd_results.npz             # forward-model predictions for each candidate
plots/
```

`candidates.csv` is the scientific output: one row per designed protocol.

### Step 4 — Uncertainty estimation

Re-scores existing candidates under many noise draws to separate a confident prediction from a lucky
one. Needs **both** the Step-1 checkpoints and the Step-3 dumps.

```bash
python scripts/inverse/uncertainty/uncertainty_estimation.py \
    --result_dir  project_folder/output/inverse/loss_guidance/<dump_name> \
    --experiment_type unconstrained-pure_populations-reciprocal \
    --target_cell_types "late_MgkPro__EryPro" \
    --n_noise_samples 5000 --n_populations 10 \
    --base_config_path ../../../inverse/loss_guidance/config/ \
    --base_config_name run_inverse \
    --true_concentration_path project_folder/data/gdrive/loop3/unique_concentrations.h5ad \
    --paths loop3_replicate \
    --cell_type_column cell_type \
    --annotation bloodplus_loop3 \
    --constraints default_all_cols_loop3
```

`--result_dir` + `--experiment_type` must jointly reconstruct the sweep's `paths.dump_dir` — i.e.
`experiment_type` is the `<opt>-<query_type>-<scheduler>` folder. Target cell types are `__`-separated.
`--base_config_path` is resolved relative to the *script*, not the working directory.
`--base_config_path`, `--base_config_name` and `--true_concentration_path` are all mandatory, the last
one despite being unused (see [Known rough edges](#known-rough-edges)).
Writes `<run>/uncertainty_annotation/candidates_with_uncertainties.csv`, adding `<ct>_prop_std`,
`target_ct_loss_mean/std` and `ct_prop_total_variance`.

Batch wrappers: `sbatch/submitters/uncertainty_launcher.sh` loops the six standard experiment types.
The `uncertainty_delta_across_loops/` variants re-score an *earlier* loop's candidates under a *later*
loop's checkpoints — this is what quantifies how much each loop reduced uncertainty — and write
`candidates_with_uncertainties_next_loop.csv`.

### Step 5 — Per-run plots and sensitivity (optional)

```bash
python scripts/inverse/plots/run_plots_inverse.py --target_cell_type <ct> --run <run_id> \
    --base_dir <dump_dir> --experiment_type <opt-query-scheduler>

python scripts/inverse/sensitivity_analysis/run_sensitivity_analysis.py --grid_size 20 …
```

The first renders per-run and per-sample diagnostics from the `.npz` files; the second sweeps each
protocol axis around a chosen candidate and writes `sensitivity_response_data.pkl` plus plots.

---

## Analysis notebooks

`data_notebooks/` holds the per-loop analysis of what actually came back from the wet lab —
`loop0/` … `loop4/`, plus `loop3_loop4_valid/` (validation and trajectory analyses) and
`sequential_uncertainty_reduction/` (how uncertainty fell loop over loop). They read the datasets
from Step 0 and the designed protocols from Steps 3–4, and produce the paper figures.

> Loop 5 has a dataset (Step 0 builds it) but no analysis notebooks yet.

Notebooks route figures to a `plots/` folder beside the notebook and set `REPO_ROOT` themselves, as
described above. Heavy embedded outputs (rendered images) have been stripped to keep the repository
small, so committed notebooks keep their printed output but not their figures — re-run a notebook to
regenerate those into its `plots/` folder.

---

## Distributing the designed protocols

The analysis notebooks read inverse-design outputs that took GPU-days to produce and, more
importantly, that **cannot be regenerated exactly** — sampling is stochastic, and the protocols that
were actually executed in the wet lab are a specific historical draw. Telling readers to "just run
the sweep yourself" would silently break the link between the published figures and the experiments
that produced them. But the full sweep output is also far too large to ship wholesale.

The sensible split follows the size/regenerability boundary:

**Publish — small, and the actual scientific claim.**
`candidates.csv` and `candidates_with_uncertainties.csv` for every run the paper builds on, plus the
per-run `config.yaml` that records exactly how each was produced. These are kilobyte-to-megabyte
tabular files. They make every analysis notebook runnable, and they *are* the result: the designed
protocols, their predicted target proportions, and their uncertainties. Ship the executed protocols —
the ones that went to the bench — as a separate curated table, since those carry the most scientific
weight and are what most readers will want.

**Don't publish — large, and reproducible in kind.**
`inverse_results.npz` and `fwd_results.npz`, i.e. the full guidance trajectories and per-candidate
forward samples. These dominate the footprint, are only needed for trajectory and sensitivity plots,
and anyone who wants them can regenerate equivalents with Step 3. Document that, rather than
uploading tens of gigabytes that will be downloaded by almost nobody.

**Where.** The same dataset repository as the measurements
([`theislab/LabCompass`](https://huggingface.co/datasets/theislab/LabCompass)), under a `solutions/`
prefix alongside the existing `loops/`, mirroring the
`<dump_name>/<experiment_type>/<cell_type>/<run_id>/` layout the notebooks already expect — one
place, one access story, and versioned by revision so a paper can cite an exact state.
For a citable archival copy with a DOI, mirror that same tree to Zenodo at submission; the usual
arrangement is HuggingFace for working access and Zenodo for the frozen, citable snapshot.

**What makes it trustworthy.** Keep each run's `config.yaml` next to its CSVs — it names the exact
checkpoints, seeds and hyperparameters behind that run. Publish the model checkpoints from Steps 1–2
alongside the data, since without them neither the inverse sweep nor the uncertainty re-scoring can be
re-run at all. Together with `loop_manifest.yaml`, that closes the loop: raw measurements → datasets →
checkpoints → designed protocols → figures.

> **Open question for the release:** whether to publish *every* sweep run or only those the paper
> draws on. Publishing all of it is more honest about the search actually performed — including the
> configurations that did not work — at the cost of a much larger and less navigable archive. A
> reasonable middle ground is to publish all `candidates.csv` (they are small) and curate only which
> runs the notebooks point at by default.

---

## Known rough edges

Real issues in the current scripts, listed so they don't cost you an afternoon:

- **`forward/train/train_cfm.sbatch`** re-assigns `ENV_NAME="sc_exp_design"` *after* flag parsing, so
  `-e/--env-name` is silently ignored.
- **`inverse/sweep/sweep_inverse_sgd.sbatch`** passes `+sgd_init_from_data="${init_from_data}"`, but
  the variable it computes is `sgd_init_from_data` — the override expands to empty. The command also
  ends in a dangling `\`.
- **`uncertainty_estimation.py`** requires `--true_concentration_path`, but the only use of it is
  commented out.
- Flag parsing in `sbatch/submitters/` is inconsistent: only `launch_sweep_inverse.sh` uses a proper
  `while` loop and accepts several flags. Four (`launch_sweep_inverse_bloodplus{,_fm,_fm_loop2,_fm_old_measurement}.sh`)
  use a bare `case $1`, so only the **first** flag is honoured. The rest
  (`uncertainty_launcher.sh`, both `uncertainty_delta_across_loops/*.sh`, `…_fm_marker_opt.sh`) parse
  no flags at all and take a positional argument instead. `uncertainty_launcher.sh` additionally uses
  a relative `../sbatch_launchers/…` path, so it only works when run from `sbatch/submitters/`.
- Checkpoint paths in the `paths` config groups point at specific dated runs and must be overridden.
