"""Write inference-only copies of the model checkpoints for publication.

Checkpoints are cloudpickled model objects that carry their training and
validation data inside them (`DataManager.adata` and the `*_data` attributes),
which makes the forward models over 100 GB each. Inference needs none of it:
`FlowMatching.predict` touches `data_manager.has_controls`, a property derived
from `control_key`, and `TargetPredictionModel.predict` touches only the network.

This reads each checkpoint, drops the embedded data, and writes a slim copy to a
separate tree. It never modifies or removes the originals.
"""

import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import cloudpickle
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ["REPO_ROOT"] = str(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "shared_utils"))

CONFIG_DIR = REPO_ROOT / "inverse/loss_guidance/config/paths"

# Paper loop -> the `paths` config whose checkpoints back that loop's figures.
LOOP_SETS = {
    # Loop N's models are the ones that generated the designs executed in loop N+1.
    "loop0": "loop0",
    "loop1": "loop1",
    "loop2": "loop2_replicate",
    "loop2p5": "loop2p5_replicate",
    "loop3": "loop3_replicate",
    "loop4": "loop4_replicate",
}
CHECKPOINT_KEYS = [
    "perturbation_prediction_path",
    "ct_classifier_path",
    "prior_flow_path",
    "prior_flow_map_path",
]


def resolve(value: str) -> Path:
    return Path(str(value).replace("${oc.env:REPO_ROOT}/", f"{REPO_ROOT}/"))


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while block := fp.read(chunk):
            h.update(block)
    return h.hexdigest()


def _force_cpu_load() -> None:
    """Makes torch unpickle onto the CPU.

    The checkpoints hold CUDA tensors, and these run on CPU nodes: the tensors
    would not fit in GPU memory anyway. Published checkpoints are CPU-resident,
    so they load anywhere.
    """
    import torch

    if getattr(torch.load, "_cpu_forced", False):
        return
    original = torch.load

    def load_on_cpu(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return original(*args, **kwargs)

    load_on_cpu._cpu_forced = True
    torch.load = load_on_cpu


def strip_model(model) -> list[str]:
    """Drops everything that exists only for training. Returns what was cleared.

    The training arrays are reachable from several attributes at once
    (`train_data`, `data_manager.adata`, and the dataloaders), and cloudpickle
    keeps the data alive if *any* of them still points at it, so all must go.
    `predict` falls back to `validation_dataloader.batch_size`, so each
    dataloader is replaced by a stand-in carrying just that.
    """
    cleared = []

    # The pipeline reads three things out of the "training" data:
    #   adata.uns  -> z-normalisation params (get_forward_model)
    #   adata.obs  -> cell-type labels (every inverse entry point)
    #   data.perturbation_covariates -> the condition key (loss guidance)
    # Those are metadata; the cell matrices are the bulk and are what goes.
    train_data = getattr(model, "train_data", None)
    if train_data is not None:
        adata = getattr(train_data, "adata", None)
        if adata is not None:
            import anndata as ad

            # only the label columns are ever read (obs[cell_type_column]);
            # the numeric protocol columns are bulk and are published with the data
            labels = adata.obs.select_dtypes(exclude="number")
            train_data.adata = ad.AnnData(obs=labels.copy(), uns=dict(adata.uns))
            cleared.append(
                f"train_data.adata X/obsm/layers/numeric-obs "
                f"(kept {len(labels.columns)} label cols of {len(adata.obs.columns)}, {adata.n_obs} rows)"
            )

        inner = getattr(train_data, "data", None)
        for field in list(vars(train_data)):
            if field in ("adata", "data"):
                continue
            value = getattr(train_data, field, None)
            if value is None:
                continue
            # `perturbation_covariates` is derived from this dict's keys, so the
            # keys stay and only the arrays go
            if field == "perturbation_data" and isinstance(value, dict):
                setattr(train_data, field, {k: None for k in value})
                cleared.append(f"train_data.{field} values (kept {len(value)} keys)")
            else:
                setattr(train_data, field, None)
                cleared.append(f"train_data.{field}")

        if inner is not None:
            inner_adata = getattr(inner, "adata", None)
            if inner_adata is not None:
                inner.adata = None
                cleared.append("train_data.data.adata")
            for field in list(vars(inner)):
                if field in ("adata", "perturbation_covariates"):
                    continue
                value = getattr(inner, field, None)
                if value is None:
                    continue
                if field == "perturbation_data" and isinstance(value, dict):
                    setattr(inner, field, {k: None for k in value})
                    cleared.append(f"train_data.data.{field} values (kept {len(value)} keys)")
                else:
                    setattr(inner, field, None)
                    cleared.append(f"train_data.data.{field}")

    validation = getattr(model, "validation_data", None)
    if validation:
        model.validation_data = {} if isinstance(validation, dict) else None
        cleared.append("validation_data")

    manager = getattr(model, "data_manager", None)
    if manager is not None:
        # the manager and each of its schemas hold their own reference to the
        # same AnnData; every one of them has to go or the data stays pickled
        holders = [("data_manager", manager)]
        holders += [(f"data_manager.{n}", getattr(manager, n, None)) for n in vars(manager)]
        for label, holder in holders:
            if holder is not None and getattr(holder, "adata", None) is not None:
                holder.adata = None
                cleared.append(f"{label}.adata")

    for name in list(vars(model)):
        value = getattr(model, name, None)
        if value is None:
            continue
        if "dataloader" in name:
            # keep batch_size, which predict() reads when none is passed
            setattr(model, name, SimpleNamespace(batch_size=getattr(value, "batch_size", None)))
            cleared.append(f"{name} (kept batch_size)")
        elif "trainer" in name:
            setattr(model, name, None)
            cleared.append(name)

    # the weights are unpickled onto the CPU, so the declared device has to
    # match or the model is internally inconsistent; users move it to GPU
    import torch

    if getattr(model, "device_id", None) not in (None, "cpu"):
        model.device_id = "cpu"
        cleared.append("device_id -> cpu")
    if getattr(model, "device", None) is not None and str(model.device) != "cpu":
        model.device = torch.device("cpu")
        cleared.append("device -> cpu")

    return cleared


def process(source: Path, destination: Path, dry_run: bool) -> dict:
    record = {"source": str(source), "destination": str(destination)}
    record["source_bytes"] = source.stat().st_size

    if dry_run:
        return record

    print(f"    loading ({record['source_bytes'] / 1e9:.1f} GB)...", flush=True)
    from labcompass.models.base import _install_legacy_module_alias

    _install_legacy_module_alias()
    _force_cpu_load()
    with open(source, "rb") as fp:
        model = cloudpickle.load(fp)

    record["class"] = type(model).__name__
    record["cleared"] = strip_model(model)
    print(f"    cleared: {record['cleared']}", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".pkl.partial")
    with open(tmp, "wb") as fp:
        cloudpickle.dump(model, fp)
    tmp.rename(destination)

    record["dest_bytes"] = destination.stat().st_size
    record["ratio"] = record["source_bytes"] / max(record["dest_bytes"], 1)
    record["dest_sha256"] = sha256(destination)
    print(
        f"    wrote {record['dest_bytes'] / 1e9:.2f} GB "
        f"({record['ratio']:.0f}x smaller) -> {destination}",
        flush=True,
    )
    return record


def main(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root).resolve()
    selected = args.loops or list(LOOP_SETS)

    manifest_path = out_root / f"manifest-{selected[0]}.json" if len(selected) == 1 else out_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for loop in selected:
        config = yaml.safe_load((CONFIG_DIR / f"{LOOP_SETS[loop]}.yaml").read_text()) or {}
        print(f"\n=== {loop}  (paths={LOOP_SETS[loop]}) ===", flush=True)

        for key in CHECKPOINT_KEYS:
            if not config.get(key):
                continue
            source = resolve(config[key])
            destination = out_root / loop / source.name
            entry_id = f"{loop}/{source.name}"

            if not source.exists():
                print(f"  MISSING {source}", flush=True)
                continue
            if destination.exists() and not args.overwrite:
                print(f"  skip (exists) {entry_id}", flush=True)
                continue
            if args.only and args.only not in source.name:
                continue

            print(f"  {key} -> {source.name}", flush=True)
            manifest[entry_id] = process(source, destination, args.dry_run)
            manifest[entry_id]["paths_config"] = LOOP_SETS[loop]
            manifest[entry_id]["config_key"] = key

            if not args.dry_run:
                out_root.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nmanifest: {manifest_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(REPO_ROOT / "project_folder/release/checkpoints"))
    parser.add_argument("--loops", nargs="+", choices=list(LOOP_SETS), default=None)
    parser.add_argument("--only", default=None, help="Process only checkpoints whose filename contains this.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
