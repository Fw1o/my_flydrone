"""Configuration loading: packaged defaults deep-merged with a user YAML file."""

from __future__ import annotations

import copy
from importlib import resources
from pathlib import Path

import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if v is None:
            # A YAML key with no value parses as None.  Treating that as "override with
            # nothing" wipes a whole section (it once turned `decoder:` into None and
            # crashed the decoder); keep the default instead.
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not k.endswith("terms"):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_config() -> dict:
    text = resources.files("flydrones").joinpath("defaults.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def load_config(
    path: str | Path | list[str | Path] | None = None,
    overrides: dict | None = None,
) -> dict:
    """Deep-merge the packaged defaults with one or more YAML files, then ``overrides``.

    ``path`` may be a single file or a list; lists are merged left to right, so the
    last file wins.  That is what lets a real flight combine a brain config with a
    per-drone config:``--config configs/fly_core2.yaml --config configs/tello.yaml``.
    """
    cfg = default_config()
    paths = path if isinstance(path, (list, tuple)) else ([path] if path else [])
    for p in paths:
        if not p:
            continue
        file = Path(p)
        if not file.exists():
            raise FileNotFoundError(f"config file not found: {file}")
        with open(file, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg
