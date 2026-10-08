"""Consolidate the inverse-design sweeps into one table per loop.

The sweep output is a deep tree of ~16,000 run directories, each holding a
50-row CSV. This flattens each loop into a single gzipped CSV, turning the
directory levels into columns, so the designs can be loaded and filtered
without walking the filesystem.

Nothing is filtered and nothing is dropped: every candidate every sweep
produced is included, because the analysis notebooks apply their own per-loop,
per-cell-type thresholds downstream. Source dumps are only ever read.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ["REPO_ROOT"] = str(REPO_ROOT)

DUMP_ROOT = REPO_ROOT / "project_folder/output/inverse/loss_guidance"

# paper loop -> the dump directory holding that loop's designs
LOOP_DUMPS = {
    "loop0": "bloodplus_inverse_fm_old_measurements_official",
    "loop1": "bloodplus_inverse_fm_official",
    "loop2": "bloodplus_inverse_fm_loop2",
    "loop2p5": "loop2p5_replicate",
    "loop3": "loop3_replicate",
    "loop4": "loop4_replicate",
}

# The uncertainty file is a superset of candidates.csv, so it is preferred when
# present. `next_loop` is the same candidates re-scored under a later loop's
# model, which is a separate quantity and is kept as extra rows.
UNCERTAINTY = "uncertainty_annotation/candidates_with_uncertainties.csv"
NEXT_LOOP = "uncertainty_annotation/candidates_with_uncertainties_next_loop.csv"
PLAIN = "candidates.csv"

PROVENANCE = ["loop", "experiment_type", "cell_type", "run_id", "uncertainty_scoring"]


def read_run(run_dir: Path) -> list[pd.DataFrame]:
    """Returns one frame per available scoring of this run's candidates."""
    frames = []

    same = run_dir / UNCERTAINTY
    if same.exists():
        frame = pd.read_csv(same)
        frame["uncertainty_scoring"] = "same_loop"
        frames.append(frame)
    else:
        # uncertainty estimation never ran here; the designs still exist
        plain = run_dir / PLAIN
        if plain.exists():
            frame = pd.read_csv(plain)
            frame["uncertainty_scoring"] = "none"
            frames.append(frame)

    later = run_dir / NEXT_LOOP
    if later.exists():
        frame = pd.read_csv(later)
        frame["uncertainty_scoring"] = "next_loop"
        frames.append(frame)

    return frames


def build(loop: str, out_dir: Path) -> None:
    dump = DUMP_ROOT / LOOP_DUMPS[loop]
    print(f"{loop}: reading {dump}", flush=True)

    frames, n_runs, n_empty = [], 0, 0
    for experiment_type in sorted(p for p in dump.iterdir() if p.is_dir()):
        for cell_type in sorted(p for p in experiment_type.iterdir() if p.is_dir()):
            for run in sorted(p for p in cell_type.iterdir() if p.is_dir()):
                n_runs += 1
                run_frames = read_run(run)
                if not run_frames:
                    n_empty += 1
                    continue
                for frame in run_frames:
                    frame["loop"] = loop
                    frame["experiment_type"] = experiment_type.name
                    frame["cell_type"] = cell_type.name
                    frame["run_id"] = run.name
                    frames.append(frame)
        print(f"  {experiment_type.name}: {n_runs} runs cumulative", flush=True)

    table = pd.concat(frames, ignore_index=True)
    table = table.loc[:, [c for c in table.columns if not str(c).startswith("Unnamed")]]
    table = table[PROVENANCE + [c for c in table.columns if c not in PROVENANCE]]

    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{loop}.csv.gz"
    table.to_csv(destination, index=False, compression="gzip")

    counts = table["uncertainty_scoring"].value_counts().to_dict()
    print(
        f"{loop}: {n_runs} runs -> {len(table):,} rows x {table.shape[1]} cols | "
        f"scoring={counts} | runs with no CSV: {n_empty} | "
        f"{destination.stat().st_size / 1e6:.1f} MB -> {destination}",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loop", required=True, choices=list(LOOP_DUMPS))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "project_folder/release/solutions"))
    a = p.parse_args()
    build(a.loop, Path(a.out_dir))
