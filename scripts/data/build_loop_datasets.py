"""Build the cumulative per-loop AnnData datasets consumed by the models.

Each iteration of the experimental design cycle ("loop") measures a new batch of
cells. Models are always trained on every loop measured so far, so loop N's
dataset is `concat(dataset(N - 1), residual(N))`. This script downloads the raw
per-loop files from HuggingFace (or reads a local copy), replays that chain, and
writes the result into a `project_folder`-style tree.

This is the scripted equivalent of `data_notebooks/concatenate_loop_data.ipynb`.

Loops whose outputs already exist are skipped, so re-running is cheap; pass
`--loops` to force specific ones to rebuild, or `--overwrite` to rebuild all.

Run `python scripts/data/build_loop_datasets.py --help` for usage.
"""

import argparse
import logging
import os
from pathlib import Path

import anndata as ad
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ["REPO_ROOT"] = str(REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "loop_manifest.yaml"
VARIANTS = ("500k", "full")


def sanitize_anndata(adata: ad.AnnData) -> ad.AnnData:
    """Casts object-dtype columns to a nullable string dtype so h5ad writing succeeds.

    The nullable dtype is deliberate: `astype(str)` would turn missing values
    into the literal string "nan".
    """
    for frame in (adata.obs, adata.var):
        for col in frame.columns:
            if frame[col].dtype == "object":
                frame[col] = frame[col].astype("string")
    return adata


class SourceResolver:
    """Resolves manifest-relative source paths to local files."""

    def __init__(self, source_dir: Path | None, repo_id: str, revision: str):
        self.source_dir = source_dir
        self.repo_id = repo_id
        self.revision = revision

    def resolve(self, relative_path: str) -> Path:
        if self.source_dir is not None:
            local = self.source_dir / relative_path
            if not local.exists():
                msg = f"Source file not found: {local}"
                raise FileNotFoundError(msg)
            return local

        from huggingface_hub import hf_hub_download

        logger.info(f"Fetching {relative_path} from {self.repo_id} (cached after first download)...")
        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                filename=relative_path,
                repo_type="dataset",
                revision=self.revision,
            )
        )


def loop_output_paths(out_root: Path, manifest: dict, loop: dict) -> dict:
    loop_dir = out_root / manifest["output_subdir"] / loop["output_dir"]
    return {key: loop_dir / name for key, name in manifest["file_names"].items()}


def write_adata(adata: ad.AnnData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {adata.shape[0]} cells x {adata.shape[1]} features -> {path}")
    sanitize_anndata(adata).write_h5ad(path)


def build_seed(loop: dict, resolver: SourceResolver, outputs: dict, variants: tuple) -> dict:
    """Materializes the first loop; its cumulative dataset is just its own measurements."""
    cumulative = {}
    for variant in variants:
        adata = ad.read_h5ad(resolver.resolve(loop["source"][f"residual_{variant}"]))
        logger.info(f"[{loop['name']}] {variant}: seeding chain with {adata.shape[0]} cells")
        write_adata(adata, outputs[f"residual_{variant}"])
        write_adata(adata, outputs[f"concat_{variant}"])
        cumulative[variant] = adata
    return cumulative


def build_loop(loop: dict, previous: dict, resolver: SourceResolver, outputs: dict, variants: tuple) -> dict:
    """Concatenates one loop's residual measurements onto the accumulated data."""
    cumulative = {}
    for variant in variants:
        residual = ad.read_h5ad(resolver.resolve(loop["source"][f"residual_{variant}"]))
        logger.info(f"[{loop['name']}] {variant}: residual has {residual.shape[0]} cells")

        accumulated = previous[variant]
        fill = loop.get("fill_missing_obs_columns")
        if fill is not None:
            logger.info(f"[{loop['name']}] {variant}: zero-filling {fill['columns']} on accumulated data")
            accumulated = accumulated.copy()
            accumulated.obs[fill["columns"]] = fill["value"]

        concatenated = ad.concat((accumulated, residual), merge="same", uns_merge="same")
        concatenated.obs_names_make_unique()
        logger.info(f"[{loop['name']}] {variant}: accumulated to {concatenated.shape[0]} cells")

        write_adata(residual, outputs[f"residual_{variant}"])
        write_adata(concatenated, outputs[f"concat_{variant}"])
        cumulative[variant] = concatenated

    return cumulative


def main(args: argparse.Namespace) -> None:
    manifest_path = DEFAULT_MANIFEST if args.manifest is None else Path(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text())

    chain = manifest["loops"]
    known = [loop["name"] for loop in chain]
    unknown = set(args.loops or []) - set(known)
    if unknown:
        msg = f"Unknown loop(s) {sorted(unknown)}. Known loops: {known}"
        raise SystemExit(msg)

    variants = tuple(args.variants)
    out_root = Path(args.output_root).resolve()
    resolver = SourceResolver(
        source_dir=Path(args.source_dir).resolve() if args.source_dir else None,
        repo_id=args.hf_repo_id or manifest["hf_repo_id"],
        revision=args.hf_revision or manifest.get("hf_revision", "main"),
    )

    logger.info(f"Output root: {out_root}")
    logger.info(f"Variants: {list(variants)}")

    cumulative: dict = {}
    for loop in chain:
        outputs = loop_output_paths(out_root, manifest, loop)
        up_to_date = all(outputs[f"concat_{v}"].exists() for v in variants)
        forced = args.overwrite or (args.loops is not None and loop["name"] in args.loops)

        if up_to_date and not forced:
            logger.info(f"[{loop['name']}] Already built; reading it to continue the chain.")
            cumulative = {v: ad.read_h5ad(outputs[f"concat_{v}"]) for v in variants}
        elif loop.get("seed"):
            cumulative = build_seed(loop, resolver, outputs, variants)
        else:
            cumulative = build_loop(loop, cumulative, resolver, outputs, variants)

    logger.info("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-repo-id", default=None, help="HuggingFace dataset repo holding the raw per-loop files.")
    parser.add_argument("--hf-revision", default=None, help="Branch, tag or commit of the dataset repo.")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Build from a local copy of the raw files instead of downloading them.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "project_folder"),
        help="Root of the project_folder-style output tree (default: %(default)s).",
    )
    parser.add_argument("--manifest", default=None, help=f"Manifest to use (default: {DEFAULT_MANIFEST}).")
    parser.add_argument("--loops", nargs="+", default=None, help="Force these loops to rebuild.")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS, help="Dataset variants.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild every loop, even if already built.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
