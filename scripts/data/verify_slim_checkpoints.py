"""Checks that slimmed checkpoints are functionally identical to the originals.

Compares every network parameter and buffer bitwise, so a difference in any
learned weight is caught, and confirms the slim copy loads and keeps the
configuration that inference depends on.
"""

import argparse
import os
import sys
from pathlib import Path

import cloudpickle
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ["REPO_ROOT"] = str(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "shared_utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# attributes inference depends on and that stripping must not disturb.
# `device_id`/`device` are deliberately moved to CPU so the published
# checkpoints load anywhere, so they are checked separately.
CONFIG_ATTRS = ["generate_from_noise", "num_time_steps", "solver_kwargs"]


def load(path: Path):
    from labcompass.models.base import _install_legacy_module_alias

    _install_legacy_module_alias()
    original = torch.load
    if not getattr(original, "_cpu_forced", False):
        def on_cpu(*args, **kwargs):
            kwargs.setdefault("map_location", "cpu")
            return original(*args, **kwargs)
        on_cpu._cpu_forced = True
        torch.load = on_cpu
    with open(path, "rb") as fp:
        return cloudpickle.load(fp)


def tensors(model) -> dict:
    """All parameters and buffers of every nn.Module the model holds."""
    found = {}
    for name in sorted(vars(model)):
        value = getattr(model, name, None)
        if isinstance(value, torch.nn.Module):
            for param_name, tensor in value.state_dict().items():
                found[f"{name}.{param_name}"] = tensor
    return found


def compare(fat_path: Path, slim_path: Path) -> dict:
    fat, slim = load(fat_path), load(slim_path)
    result = {
        "file": slim_path.name,
        "class_match": type(fat).__name__ == type(slim).__name__,
        "class": type(slim).__name__,
    }

    a, b = tensors(fat), tensors(slim)
    result["n_tensors"] = len(a)
    result["keys_match"] = sorted(a) == sorted(b)

    mismatched = [k for k in a if k in b and not torch.equal(a[k].cpu(), b[k].cpu())]
    result["tensors_identical"] = not mismatched and result["keys_match"] and len(a) > 0
    result["mismatched"] = mismatched[:5]

    config = {}
    for attr in CONFIG_ATTRS:
        if hasattr(fat, attr):
            config[attr] = repr(getattr(fat, attr, None)) == repr(getattr(slim, attr, None))
    result["config_match"] = all(config.values()) if config else True
    result["config_checked"] = list(config)

    manager = getattr(slim, "data_manager", None)
    result["has_controls_available"] = manager is None or hasattr(manager, "has_controls")

    # the slim copy must be self-consistent: CPU-resident weights and a
    # declared device to match
    slim_device = str(getattr(slim, "device", "cpu"))
    result["device_is_cpu"] = slim_device == "cpu" and all(
        t.device.type == "cpu" for t in b.values()
    )

    result["ok"] = (
        result["class_match"] and result["tensors_identical"]
        and result["config_match"] and result["has_controls_available"]
        and result["device_is_cpu"]
    )
    return result


def main(args):
    import yaml
    from slim_checkpoints import CHECKPOINT_KEYS, CONFIG_DIR, LOOP_SETS, resolve

    config = yaml.safe_load((CONFIG_DIR / f"{LOOP_SETS[args.loop]}.yaml").read_text()) or {}
    release = Path(args.release_root) / args.loop

    results = []
    for key in CHECKPOINT_KEYS:
        if not config.get(key):
            continue
        source = resolve(config[key])
        destination = release / source.name
        print(f"\n--- {args.loop}/{source.name}", flush=True)
        if not destination.exists():
            print("    FAIL  slim copy missing", flush=True)
            results.append({"ok": False})
            continue
        outcome = compare(source, destination)
        results.append(outcome)
        status = "PASS" if outcome["ok"] else "FAIL"
        print(f"    {status}  {outcome['class']}  tensors={outcome['n_tensors']}  "
              f"identical={outcome['tensors_identical']}  config={outcome['config_match']}  "
              f"cpu={outcome.get('device_is_cpu')}", flush=True)
        if not outcome["ok"]:
            print(f"    mismatched: {outcome.get('mismatched')}", flush=True)

    ok = sum(r["ok"] for r in results)
    print(f"\nRESULT {args.loop}: {ok}/{len(results)} passed", flush=True)
    sys.exit(0 if results and ok == len(results) else 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loop", required=True)
    p.add_argument("--release-root", default=str(REPO_ROOT / "project_folder/release/checkpoints"))
    main(p.parse_args())
