"""Artifact helpers: JSON-safe conversion, hashes, version records, logging."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np


def jsonable(o):
    """Recursively convert dataclasses / numpy / Path / tensors to JSON-safe values."""
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {f.name: jsonable(getattr(o, f.name)) for f in dataclasses.fields(o)}
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, float):
        return None if o != o else o  # NaN is not valid JSON
    if isinstance(o, (list, tuple, set)):
        return [jsonable(x) for x in o]
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, Path):
        return str(o)
    if o is None or isinstance(o, (str, int, bool)):
        return o
    if hasattr(o, "detach") and hasattr(o, "cpu"):
        return jsonable(o.detach().cpu().numpy())
    return repr(o)


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def versions_info(clf=None) -> dict:
    import sklearn
    import torch

    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    try:
        import tabpfn
        from importlib.metadata import version

        info["tabpfn"] = version("tabpfn")
        info["tabpfn_path"] = str(Path(tabpfn.__file__).parent)
    except Exception as e:  # pragma: no cover
        info["tabpfn"] = f"unavailable: {e!r}"
    if clf is not None:
        paths = getattr(clf, "model_path", None)
        info["model_path"] = str(paths)
        for p in paths if isinstance(paths, (list, tuple)) else [paths]:
            try:
                if p is not None and Path(p).is_file():
                    info.setdefault("checkpoint_sha256", {})[str(p)] = sha256_file(Path(p))
            except Exception:
                pass
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name()
    return info


def ensemble_config_record(clf) -> dict:
    cfgs = getattr(clf, "ensemble_configs_", None)
    return {
        "n_estimators_resolved": getattr(clf, "n_estimators_", None),
        "random_state": clf.random_state,
        "softmax_temperature": clf.softmax_temperature,
        "estimators": [jsonable(c) for c in cfgs] if cfgs is not None else None,
    }


def setup_logger(out_dir: Path | None, name: str = "tabfm") -> logging.Logger:
    """Log to stdout and (optionally) ``out_dir/run.log``."""
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(out_dir) / "run.log", mode="w")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log
