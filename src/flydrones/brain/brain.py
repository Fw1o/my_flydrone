"""Brain = connectome + LIF dynamics + named input/output groups."""

from __future__ import annotations

import re
import time
import warnings
from pathlib import Path

import numpy as np

from .connectome import Connectome, GroupSpec
from .lif import LIFNetwork, LIFParams
from .synthetic import build_minifly


def load_connectome(source: str | Path) -> Connectome:
    if str(source).lower() in ("minifly", "synthetic", "mini"):
        return build_minifly()
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(
            f"brain file {path} not found. Run `flydrones download malecns` and "
            "`flydrones build-brain` first, or use brain.source: minifly"
        )
    return Connectome.load(path)


def make_network(weights, params: LIFParams, seed: int, bcfg: dict):
    """Build the LIF network implementation selected by ``brain.backend``.

    * ``cpu`` (default) -- NumPy/SciPy :class:`LIFNetwork`
    * ``torch`` -- CUDA :class:`~flydrones.brain.lif_torch.TorchLIFNetwork`
    * ``auto`` -- torch when a usable CUDA device is present, otherwise cpu

    The torch backend is imported lazily so the CPU path keeps working in
    environments where PyTorch is not installed.  An explicit ``torch`` request
    raises instead of silently falling back to the slower CPU path.
    """
    backend = str(bcfg.get("backend") or "cpu").lower()
    if backend not in ("cpu", "torch", "auto"):
        raise ValueError(f"unknown brain.backend {backend!r}; expected cpu|torch|auto")
    if backend == "cpu":
        return LIFNetwork(weights, params, seed=seed)

    try:
        from .lif_torch import TorchLIFNetwork, torch_device_available
    except ImportError:
        if backend == "torch":
            raise
        return LIFNetwork(weights, params, seed=seed)

    if backend == "auto" and not torch_device_available():
        warnings.warn("brain.backend=auto but no CUDA device is available; using the CPU backend", stacklevel=2)
        return LIFNetwork(weights, params, seed=seed)

    opts = dict(bcfg.get("torch") or {})
    return TorchLIFNetwork(weights, params, seed=seed, **opts)


class Brain:
    """A simulated fly brain with sensory entry points and motor read-outs."""

    def __init__(self, connectome: Connectome, cfg: dict, seed: int | None = None, _net: LIFNetwork | None = None):
        self.cfg = cfg
        self.connectome = connectome
        bcfg = cfg.get("brain", {})
        self.input_specs = {k: GroupSpec.from_dict(k, v) for k, v in cfg.get("inputs", {}).items()}
        self.output_specs = {k: GroupSpec.from_dict(k, v) for k, v in cfg.get("outputs", {}).items()}
        for s in self.input_specs.values():
            s.role = "input"
        for s in self.output_specs.values():
            s.role = "output"

        connectome.resolve_groups({**self.input_specs, **self.output_specs})  # always follow the current config
        connectome.meta.setdefault("roles", {})
        connectome.meta["roles"].update({k: "input" for k in self.input_specs})
        connectome.meta["roles"].update({k: "output" for k in self.output_specs})

        self.empty = [k for k in {**self.input_specs, **self.output_specs} if connectome.group(k).size == 0]
        if self.empty:
            warnings.warn(f"groups with no matching neurons in {connectome.name}: {', '.join(self.empty)}", stacklevel=2)

        params = LIFParams.from_dict(bcfg.get("lif"))
        seed = bcfg.get("seed", 0) if seed is None else seed
        self.net = _net if _net is not None else make_network(connectome.weights, params, seed, bcfg)
        for g, mv in (bcfg.get("bias") or {}).items():
            idx = connectome.group(g)
            if idx.size == 0:
                # A bias key may also be a cell-type pattern: group names only resolve
                # when they are listed under `inputs`/`outputs`, and real configs use
                # subtypes such as `DNg02_a` that never appear there.
                try:
                    idx = connectome.select([g])
                except re.error as e:
                    warnings.warn(
                        f"brain.bias entry {g!r} is not a valid cell-type pattern ({e}); ignored",
                        stacklevel=2,
                    )
                    continue
            if idx.size:
                self.net.set_bias(idx, float(mv))
            else:
                warnings.warn(
                    f"brain.bias entry {g!r} matches no neurons in {connectome.name} and has no "
                    "effect; use a group name from `inputs`/`outputs` or a cell-type pattern "
                    "(see `flydrones inspect`)",
                    stacklevel=2,
                )

        rng = np.random.default_rng(seed)
        n_rec = int(bcfg.get("record_neurons", 400))
        outs = np.concatenate([connectome.group(k) for k in self.output_specs] or [np.zeros(0, np.int64)])
        pool = np.setdiff1d(np.arange(connectome.n), outs)
        extra = rng.choice(pool, size=min(max(0, n_rec - outs.size), pool.size), replace=False) if pool.size else pool
        self.record = np.concatenate([np.sort(extra), outs]).astype(np.int64)
        self.last_raster: list[tuple[float, np.ndarray]] = []
        self.last_rates: dict[str, float] = {}
        self.realtime_factor = float("nan")

    # ------------------------------------------------------------------
    def copy(self, seed: int) -> Brain:
        """Same wiring, independent neural state and noise (one fly -> many pilots)."""
        return Brain(self.connectome, self.cfg, seed=seed, _net=self.net.copy(seed=seed))

    @property
    def n_neurons(self) -> int:
        return self.connectome.n

    def tick(self, input_rates: dict[str, np.ndarray | float], ms: float) -> dict[str, float]:
        """Apply input rates (Hz per neuron), simulate ``ms`` milliseconds, return output rates (Hz)."""
        idx_all, rate_all = [], []
        for name, rates in input_rates.items():
            idx = self.connectome.group(name)
            if idx.size == 0:
                continue
            r = np.broadcast_to(np.asarray(rates, dtype=np.float32), idx.shape)
            idx_all.append(idx)
            rate_all.append(r)
        if idx_all:
            self.net.set_input(np.concatenate(idx_all), np.concatenate(rate_all))
        else:
            self.net.set_input(np.zeros(0, np.int64), 0.0)

        t0 = time.perf_counter()
        counts, raster = self.net.run(ms, record=self.record)
        wall = time.perf_counter() - t0
        self.realtime_factor = (ms / 1000.0) / wall if wall > 0 else float("inf")
        self.last_raster = raster
        rates = {}
        for name in list(self.output_specs) + list(self.input_specs):
            idx = self.connectome.group(name)
            rates[name] = float(counts[idx].mean() * 1000.0 / ms) if idx.size else 0.0
        self.last_rates = rates
        self.last_counts = counts
        return rates

    def stimulate(self, group: str, hz: float, ms: float) -> dict[str, float]:
        """Poke one group with Poisson spikes and see what the motor neurons do."""
        return self.tick({group: hz}, ms)
