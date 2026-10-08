"""Move experiment 205 from loop 0 into loop 1, where it belongs.

Design 205 is the first design of Loop 1 (the manuscript lists Loop 1 as designs
205-210), but it was written into the loop 0 file during data assembly. The
models were all trained without it, so the published loop 0 disagrees with both
the paper and the checkpoints.

This rewrites the two affected per-loop files for each variant. The cumulative
datasets are unaffected: the same cells end up on the other side of one
boundary. Sources are read only; corrected copies are written elsewhere.
"""

import argparse
import os
from pathlib import Path

import anndata as ad

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ["REPO_ROOT"] = str(REPO_ROOT)

LOOPS = REPO_ROOT / "project_folder/output/loops/data"
MOVED_EXPERIMENT = "205"

# variant -> (loop0 source, loop1 residual source, published names)
VARIANTS = {
    "full": (LOOPS / "loop0/adata_full.h5ad",
             LOOPS / "loop1/adata_full_residual.h5ad",
             "loop0.h5ad", "loop1.h5ad"),
    "500k": (LOOPS / "loop0/adata_500k.h5ad",
             LOOPS / "loop1/adata_500k_residual.h5ad",
             "loop0_500k.h5ad", "loop1_500k.h5ad"),
}


def run(variant: str, out_dir: Path) -> None:
    loop0_src, loop1_src, loop0_name, loop1_name = VARIANTS[variant]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{variant}] reading {loop0_src}", flush=True)
    loop0 = ad.read_h5ad(loop0_src)
    print(f"[{variant}] reading {loop1_src}", flush=True)
    loop1 = ad.read_h5ad(loop1_src)

    exp = loop0.obs["experiment_number"].astype(str)
    mask = exp == MOVED_EXPERIMENT
    print(f"[{variant}] loop0 {loop0.shape} | experiment {MOVED_EXPERIMENT}: {int(mask.sum())} cells", flush=True)
    print(f"[{variant}] loop1 {loop1.shape} | experiments {sorted(loop1.obs['experiment_number'].astype(str).unique())}", flush=True)

    if mask.sum() == 0:
        raise SystemExit(f"experiment {MOVED_EXPERIMENT} not present in {loop0_src} — nothing to move")

    moved = loop0[mask].copy()
    loop0_fixed = loop0[~mask].copy()
    loop1_fixed = ad.concat((loop1, moved), merge="same", uns_merge="same")
    loop1_fixed.obs_names_make_unique()

    for frame in (loop0_fixed.obs, loop0_fixed.var, loop1_fixed.obs, loop1_fixed.var):
        for col in frame.columns:
            if frame[col].dtype == "object":
                frame[col] = frame[col].astype("string")

    print(f"[{variant}] -> loop0 {loop0_fixed.shape}  loop1 {loop1_fixed.shape}", flush=True)
    print(f"[{variant}] loop1 experiments now: "
          f"{sorted(loop1_fixed.obs['experiment_number'].astype(str).unique())}", flush=True)

    loop0_fixed.write_h5ad(out_dir / loop0_name)
    loop1_fixed.write_h5ad(out_dir / loop1_name)
    for name in (loop0_name, loop1_name):
        print(f"[{variant}] wrote {(out_dir / name).stat().st_size / 1e9:.2f} GB  {name}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "project_folder/release/loops_corrected"))
    a = p.parse_args()
    run(a.variant, Path(a.out_dir))
