"""CUDA / PyTorch backend for the whole-connectome LIF simulator.

Drop-in replacement for :class:`flydrones.brain.lif.LIFNetwork`: same public
surface, same update order, same semantics.  Design notes:

1. The connectome and all neuron state stay resident on the device; nothing is
   copied back per step.  ``run()`` syncs once, at the very end.
2. Every per-step tensor has a *static* shape, so the step is CUDA-Graph
   capturable.  The two ragged quantities are made fixed-size:

   * refractory neurons -> a boolean mask + countdown array over the whole
     population (instead of a compacted index list),
   * spiking neurons -> a fixed-capacity list built with ``searchsorted`` over
     the cumulative spike mask (instead of ``nonzero``, which is dynamic).

3. The delay-ring slot is a *device* scalar, not a Python int.  A graph replays
   GPU work only, so any Python-side counter would freeze -- the slot has to
   advance on device.

Removing the per-op CPU submission cost is the whole point: measured on this
workload it was ~345 us of a ~523 us step.
"""

from __future__ import annotations

import warnings

import numpy as np

from .lif import LIFParams, StepStats

try:
    import torch
except ImportError as e:  # pragma: no cover - optional dependency
    raise ImportError(
        "the torch backend needs PyTorch. Install it with:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu126\n"
        "(on Windows/Python 3.10 use torch==2.14.x, the last release with cp310 wheels)"
    ) from e

try:
    from scipy import sparse as _sparse
except ImportError as e:  # pragma: no cover
    raise ImportError("scipy is required") from e


def torch_device_available() -> bool:
    """True when a CUDA device is present and usable."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def default_device() -> str:
    return "cuda" if torch_device_available() else "cpu"


class TorchLIFNetwork:
    """Device-resident LIF network with the same contract as LIFNetwork."""

    def __init__(
        self,
        weights: _sparse.spmatrix,
        params: LIFParams | None = None,
        seed: int | None = 0,
        device: str | None = None,
        max_spikes: int | None = None,
        max_events: int | None = None,
        use_cuda_graph: bool = True,
    ):
        self.p = params or LIFParams()
        p = self.p

        W = _sparse.csc_matrix(weights, dtype=np.float32)
        if W.shape[0] != W.shape[1]:
            raise ValueError(f"connectivity must be square, got {W.shape}")
        W.sum_duplicates()
        W.eliminate_zeros()
        self.W = W
        self.n = int(W.shape[0])

        self.device = torch.device(device or default_device())
        dev = self.device

        self._indptr = torch.from_numpy(np.asarray(W.indptr, dtype=np.int64)).to(dev)
        self._indices = torch.from_numpy(np.asarray(W.indices, dtype=np.int64)).to(dev)
        self._data = torch.from_numpy((W.data * np.float32(p.w_syn)).astype(np.float32)).to(dev)
        self._nnz = int(self._indices.numel())

        self.delay_steps = max(1, int(round(p.delay / p.dt)))
        self.ref_steps = max(1, int(round(p.t_ref / p.dt)))
        self._decay_g = float(np.exp(-p.dt / p.tau_syn))
        self._k_v = float(p.dt / p.tau_m)
        self._v_rest = float(p.v_rest)
        self._v_reset = float(p.v_reset)
        self._v_th = float(p.v_th)

        # Capacities drive the per-step work directly -- every slot of the event
        # buffer is walked each step -- so they must hug the real workload.  With a
        # captured CUDA graph the replay time is almost exactly proportional to
        # max_events: 532480 -> 0.82x, 655360 -> 0.71x on the full connectome.
        # Observed peaks for full MaleCNS under `flydrones bench`: 1468 spikes and
        # 528,455 events per step.  Overflow is reported with the observed peak.
        self.max_spikes = int(max_spikes) if max_spikes else 4096
        self.max_events = int(max_events) if max_events else 540672

        self._alloc(seed)
        self.stats = StepStats()
        self._input_idx = torch.zeros(0, dtype=torch.int64, device=dev)
        self._input_rate = torch.zeros(0, dtype=torch.float32, device=dev)
        self._input_hit = torch.zeros(0, dtype=torch.bool, device=dev)
        self._input_contrib = torch.zeros(0, dtype=torch.float32, device=dev)
        self._t = 0
        self.t_ms = 0.0
        self._graph = None
        self.use_cuda_graph = bool(use_cuda_graph)

    # ------------------------------------------------------------------ setup
    def _alloc(self, seed: int | None) -> None:
        dev = self.device
        n = self.n
        p = self.p

        self.v = torch.full((n,), self._v_rest, dtype=torch.float32, device=dev)
        self.g = torch.zeros(n, dtype=torch.float32, device=dev)
        self.bias = torch.zeros(n, dtype=torch.float32, device=dev)
        self._buf = torch.zeros((self.delay_steps, n), dtype=torch.float32, device=dev)
        self._tmp = torch.zeros(n, dtype=torch.float32, device=dev)

        self._ref_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ref_left = torch.zeros(n, dtype=torch.int32, device=dev)
        self._g_frozen = torch.zeros(n, dtype=torch.float32, device=dev)
        self._spk_mask = torch.zeros(n, dtype=torch.bool, device=dev)

        self._spk = torch.zeros(self.max_spikes, dtype=torch.int64, device=dev)
        self._spk_valid = torch.zeros(self.max_spikes, dtype=torch.bool, device=dev)
        self._arange_spk = torch.arange(self.max_spikes, dtype=torch.int64, device=dev) + 1
        self._pos = torch.arange(self.max_events, dtype=torch.int64, device=dev)
        self._nnz_or_1 = max(int(self._nnz), 1)

        self._out = torch.zeros(n, dtype=torch.float32, device=dev)
        self._counts = torch.zeros(n, dtype=torch.int32, device=dev)

        # pre-allocated scalars / device-side step state
        self._zero_i = torch.zeros((), dtype=torch.int64, device=dev)
        self._zero_f = torch.zeros((), dtype=torch.float32, device=dev)
        self._v_reset_t = torch.full((), self._v_reset, dtype=torch.float32, device=dev)
        self._ref_steps_t = torch.full((), self.ref_steps, dtype=torch.int32, device=dev)
        self._spike_acc = torch.zeros((), dtype=torch.int64, device=dev)
        self._syn_acc = torch.zeros((), dtype=torch.int64, device=dev)
        self._max_spk = torch.zeros((), dtype=torch.int64, device=dev)
        self._max_ev = torch.zeros((), dtype=torch.int64, device=dev)
        # the delay-ring slot MUST live on the device (see module docstring)
        self._slot = torch.zeros((), dtype=torch.int64, device=dev)

        self.rng = torch.Generator(device=dev)
        self.rng.manual_seed(int(seed or 0))

        if p.noise_mv > 0:
            std = float(p.noise_mv * np.sqrt(p.dt))
            self._noise = torch.randn((16, n), dtype=torch.float32, device=dev, generator=self.rng) * std
            # The noise *draw* stays outside the graph (``_draw_noise``) and only these
            # pre-allocated buffers are read inside it: a captured RNG would replay the
            # same draw every step, and sampling with ``int(tensor.item())`` -- a host
            # sync -- aborts the capture outright (cudaErrorStreamCaptureInvalidated).
            self._noise_k = torch.zeros(1, dtype=torch.int64, device=dev)
            self._noise_o = torch.zeros(1, dtype=torch.int64, device=dev)
            self._arange_n = torch.arange(n, dtype=torch.int64, device=dev)
        else:
            self._noise = None
            self._noise_k = None
            self._noise_o = None
            self._arange_n = None
        self._graph_failed = False
        self._capacity_warned = False

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.v.fill_(self._v_rest)
        self.g.zero_()
        self._buf.zero_()
        self._tmp.zero_()
        self._ref_mask.zero_()
        self._ref_left.zero_()
        self._g_frozen.zero_()
        self._spk_mask.zero_()
        self._spk_valid.zero_()
        self._counts.zero_()
        self._spike_acc.zero_()
        self._syn_acc.zero_()
        self._max_spk.zero_()
        self._max_ev.zero_()
        self._slot.zero_()
        self._t = 0
        self.t_ms = 0.0
        self._graph = None
        self._graph_failed = False
        self._capacity_warned = False

    def copy(self, seed: int | None = None) -> TorchLIFNetwork:
        """Same wiring, independent neuron state and RNG (for swarms)."""
        clone = TorchLIFNetwork.__new__(TorchLIFNetwork)
        clone.p = self.p
        clone.W = self.W
        clone.n = self.n
        clone.device = self.device
        clone._indptr, clone._indices, clone._data, clone._nnz = (
            self._indptr, self._indices, self._data, self._nnz,
        )
        clone.delay_steps, clone.ref_steps = self.delay_steps, self.ref_steps
        clone._decay_g, clone._k_v = self._decay_g, self._k_v
        clone._v_rest, clone._v_reset, clone._v_th = self._v_rest, self._v_reset, self._v_th
        clone.max_spikes, clone.max_events = self.max_spikes, self.max_events
        clone.use_cuda_graph = self.use_cuda_graph
        clone._graph = None
        clone._alloc(seed)
        clone.bias.copy_(self.bias)
        clone.stats = StepStats()
        clone._input_idx = self._input_idx.clone()
        clone._input_rate = self._input_rate.clone()
        clone._input_hit = torch.zeros_like(self._input_rate, dtype=torch.bool)
        clone._input_contrib = torch.zeros_like(self._input_rate)
        clone._t = 0
        clone.t_ms = 0.0
        return clone

    # ------------------------------------------------------------------ input
    def set_input(self, idx: np.ndarray, rates_hz: np.ndarray | float) -> None:
        idx = np.asarray(idx, dtype=np.int64)
        rates = np.broadcast_to(np.asarray(rates_hz, dtype=np.float32), idx.shape)
        keep = rates > 0
        idx = np.ascontiguousarray(idx[keep])
        rates = np.ascontiguousarray(rates[keep])

        # Same input-set size -> update the buffers in place.  A captured CUDA graph
        # holds their addresses, so reallocating them here would force a re-capture on
        # every control tick: measured on the full connectome, 48.9 ms/tick (1.02x real
        # time) becomes 56.3 ms/tick (0.89x) purely from that capture.
        if self._input_idx.numel() == idx.size:
            self._input_idx.copy_(torch.as_tensor(idx, device=self.device))
            self._input_rate.copy_(torch.as_tensor(rates, device=self.device))
            self._input_contrib.zero_()
            return

        self._input_idx = torch.from_numpy(idx).to(self.device)
        self._input_rate = torch.from_numpy(rates).to(self.device)
        self._input_hit = torch.zeros(self._input_idx.numel(), dtype=torch.bool, device=self.device)
        self._input_contrib = torch.zeros(self._input_idx.numel(), dtype=torch.float32, device=self.device)
        self._graph = None

    def set_bias(self, idx: np.ndarray, mv: np.ndarray | float) -> None:
        i = torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(self.device)
        if np.isscalar(mv):
            self.bias[i] = float(mv)
        else:
            self.bias[i] = torch.from_numpy(np.asarray(mv, dtype=np.float32)).to(self.device)

    def inject(self, idx: np.ndarray, mv: float) -> None:
        i = torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(self.device)
        self.g[i] += float(mv)

    # ------------------------------------------------------------- core step
    def _propagate(self) -> None:
        """Scatter the spiking columns into ``_out``. Device only, static shape."""
        if self._nnz == 0:            # no edges at all (e.g. a 1-neuron test net)
            self._out.zero_()
            return
        starts = self._indptr.index_select(0, self._spk)
        nxt = self._indptr.index_select(0, self._spk + 1)
        lens = torch.where(self._spk_valid, nxt - starts, self._zero_i)
        cs = torch.cumsum(lens, 0)
        total = cs[-1]
        self._syn_acc += total
        self._max_ev.copy_(torch.maximum(self._max_ev, total))
        pos = self._pos
        # base[j] = flat index of spike j's first event, so the gather below is a
        # single random access per slot instead of two.  It carries one extra
        # sentinel entry so searchsorted's out-of-range result (== max_spikes)
        # needs no clamp pass.
        base = torch.cat([
            starts - torch.cat([self._zero_i.unsqueeze(0), cs[:-1]]),
            self._zero_i.unsqueeze(0),
        ])
        real = base.index_select(0, torch.searchsorted(cs, pos, right=True)) + pos
        # Real slots are already < nnz, so the modulo is a no-op for them; for the
        # masked-out tail it wraps them *spread out* over indices[] instead of
        # piling them onto one row, which would serialise atomicAdd on a single
        # cache line (measured 1.63x).  It also keeps them in bounds for graphs
        # whose edge count is smaller than max_events.
        offs = torch.remainder(real, self._nnz_or_1)
        post = self._indices.index_select(0, offs)
        w = torch.where(pos < total, self._data.index_select(0, offs), self._zero_f)
        self._out.zero_()
        self._out.index_add_(0, post, w)

    def _compact_spikes(self) -> None:
        """Spike mask -> fixed-capacity index list. Device only, static shape."""
        cum = torch.cumsum(self._spk_mask.to(torch.int64), 0)
        total = cum[-1]
        self._max_spk.copy_(torch.maximum(self._max_spk, total))
        self._spk_valid.copy_(self._arange_spk <= total)
        idx = torch.searchsorted(cum, self._arange_spk, right=False)
        self._spk.copy_(idx.clamp_(max=self.n - 1))
        self._spk.masked_fill_(~self._spk_valid, 0)

    def _step_device(self) -> None:
        """One dt, entirely on device, every shape static, no host sync."""
        p = self.p
        slot = self._slot.unsqueeze(0)

        # 1. synaptic input that arrives now (after delay)
        arriving = self._buf.index_select(0, slot).squeeze(0)
        self.g += arriving
        self._buf.index_fill_(0, slot, 0.0)

        # 2. external Poisson drive.  The random draw happens OUTSIDE the graph
        #    (see _draw_input): a captured RNG replays the same draws forever,
        #    handing identical Poisson spikes to identical neurons every step and
        #    doubling the population rate (measured 2.02x).  Only the purely
        #    deterministic scatter is captured.
        if self._input_idx.numel():
            self.g.index_add_(0, self._input_idx, self._input_contrib)

        # 3. freeze g on refractory neurons (sampled before the decay)
        ref = self._ref_mask
        self._g_frozen.copy_(torch.where(ref, self.g, self._zero_f))

        # 4. membrane update:  dv = dt/tau_m * (v_rest - v + g + bias)
        #    spelled out as separate in-place calls: chained in-place expressions
        #    do not replay correctly inside a captured CUDA graph
        tmp = self._tmp
        tmp.copy_(self.v)
        tmp.neg_()
        tmp.add_(self._v_rest)
        tmp.add_(self.g)
        tmp.add_(self.bias)
        tmp.mul_(self._k_v)
        self.v += tmp

        # 5. optional membrane noise (off for MaleCNS).  Nothing here touches the host:
        #    the draw itself happened in _draw_noise, so the shapes stay static and a
        #    captured graph can replay the step.
        if self._noise is not None:
            row = self._noise.index_select(0, self._noise_k).squeeze(0)
            idx = torch.remainder(self._arange_n + self._noise_o, self.n)
            self.v.add_(row.index_select(0, idx))

        self.g *= self._decay_g

        # 6. refractory neurons stay clamped
        self.v.masked_fill_(ref, self._v_reset)
        self.g.masked_fill_(ref, 0.0)
        self.g.add_(self._g_frozen)
        self._ref_left -= 1
        self._ref_left.clamp_(min=0)

        # 7. threshold + compact the spikes.
        #    NOTE: these must be written as an explicit ``copy_`` of a freshly
        #    produced tensor.  Passing the destination through the ``out=`` kwarg
        #    of torch.gt/torch.le silently does nothing when the step is replayed
        #    from a captured CUDA graph, which would leave the spike mask stale.
        self._spk_mask.copy_(self.v > self._v_th)
        self._compact_spikes()
        self._counts.index_add_(0, self._spk, self._spk_valid.to(torch.int32))
        self._spike_acc += self._spk_valid.sum()

        # 8. reset spiking neurons, put them into refractory
        self.v.masked_fill_(self._spk_mask, self._v_reset)
        self.g.masked_fill_(self._spk_mask, 0.0)
        self._ref_left.masked_fill_(self._spk_mask, self.ref_steps)
        self._ref_mask.copy_(self._ref_left > 0)

        # 9. propagate into the delay ring
        self._propagate()
        self._buf.index_add_(0, slot, self._out.unsqueeze(0))

        self._slot += 1
        self._slot.remainder_(self.delay_steps)
        self._t += 1
        self.t_ms += p.dt
        self.stats.steps += 1

    # --------------------------------------------------------- graph plumbing
    def capture_graph(self) -> bool:
        """Capture one step into a CUDA Graph. Returns True on success.

        Capture must not perturb the simulation: the warmup iterations that CUDA
        requires would otherwise advance v/g/the delay ring, desynchronise the
        device-side delay slot from the host's bookkeeping and inflate the capacity
        peaks that the overflow warning reports.  State is snapshotted and restored
        around the capture.

        This method never raises.  A failed capture can leave the stream in an
        *invalidated* state, so the restore below is guarded too -- if even that
        fails, the graph path is switched off and the eager path keeps running.
        """
        if not self.use_cuda_graph or self.device.type != "cuda" or self._graph_failed:
            return False
        state = [
            self.v, self.g, self._buf, self._tmp,
            self._ref_mask, self._ref_left, self._g_frozen,
            self._spk_mask, self._spk, self._spk_valid,
            self._out, self._counts, self._spike_acc, self._syn_acc,
            self._slot,
        ]
        snapshot = [t.clone() for t in state]
        py_state = (self._t, self.t_ms, self.stats.steps)
        peak_spk = self._max_spk.clone()
        peak_ev = self._max_ev.clone()
        ok = False
        self._graph = None
        try:
            with torch.cuda.device(self.device):
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self._step_device()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._step_device()
                torch.cuda.synchronize()
            self._graph = g
            ok = True
        except Exception as e:  # pragma: no cover - depends on driver
            warnings.warn(f"CUDA graph capture failed, falling back to eager: {e}", stacklevel=2)
            # do not retry: whatever aborted the capture would abort it again, and the
            # retry would burn the warmup steps on every single run()/control tick
            self._graph_failed = True
        finally:
            self._t, self.t_ms, self.stats.steps = py_state
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass
            try:
                for t, snap in zip(state, snapshot):
                    t.copy_(snap)
                self._max_spk.copy_(peak_spk)
                self._max_ev.copy_(peak_ev)
            except Exception as e:  # pragma: no cover - depends on driver
                warnings.warn(
                    f"network state could not be restored after a failed CUDA graph capture "
                    f"({e}); continuing on the eager path",
                    stacklevel=2,
                )
                self._graph = None
                self._graph_failed = True
                ok = False
        return ok

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _draw_input(self) -> None:
        """Draw this step's Poisson input on the host side, outside any graph."""
        m = self._input_idx.numel()
        if not m:
            return
        thr = self._input_rate * np.float32(self.p.dt / 1000.0)
        hit = torch.lt(
            torch.rand(m, dtype=torch.float32, device=self.device, generator=self.rng),
            thr,
            out=self._input_hit,
        )
        self._input_contrib.copy_(
            hit.to(torch.float32) * np.float32(self.p.w_syn * self.p.poisson_factor)
        )

    def _draw_noise(self) -> None:
        """Draw this step's noise bank row / roll offset, outside any graph.

        Same reason as the Poisson draw: a captured RNG replays the same draw on every
        replay, which would freeze the noise into a fixed pattern instead of noise.
        """
        if self._noise is None:
            return
        self._noise_k.copy_(
            torch.randint(len(self._noise), (1,), device=self.device, generator=self.rng)
        )
        self._noise_o.copy_(
            torch.randint(self.n, (1,), device=self.device, generator=self.rng)
        )

    def _one_step(self) -> None:
        # the Poisson and noise draws happen on the host for BOTH paths, so the eager
        # and graph paths see identical input handling
        self._draw_input()
        self._draw_noise()
        if self._graph is not None:
            self._graph.replay()
            # keep host-side bookkeeping in step with the device
            self._t += 1
            self.t_ms += self.p.dt
            self.stats.steps += 1
        else:
            self._step_device()

    def _capacity_warning(self) -> bool:
        """Warn when the observed peak load exceeded a static capacity.

        Peaks are reported *per run* (``run`` resets them), so a single hot step does
        not keep warning forever; a part of the workload larger than the capacity is
        silently truncated otherwise.
        """
        peak_spk = int(self._max_spk.item())
        peak_ev = int(self._max_ev.item())
        if peak_spk <= self.max_spikes and peak_ev <= self.max_events:
            return False
        warnings.warn(
            f"static capacity exceeded: peak {peak_spk} spikes and {peak_ev} events "
            f"vs capacity {self.max_spikes} / {self.max_events}; results are truncated. "
            "Raise brain.torch.max_spikes and brain.torch.max_events to at least those peaks.",
            stacklevel=3,
        )
        return True

    def step(self) -> np.ndarray:
        """Advance one dt; returns the indices that spiked (host copy).

        Like :meth:`run` this keeps ``self._counts`` accumulating, but a single step has
        no return channel for capacity problems, so it reports them itself -- throttled
        to one warning, since a hot step would otherwise warn on every step.
        """
        self._one_step()
        self._sync()
        spk = self._spk[self._spk_valid].cpu().numpy().astype(np.int64)
        if not self._capacity_warned and self._capacity_warning():
            self._capacity_warned = True
        return spk

    def run(self, ms: float, record: np.ndarray | None = None):
        """Run ``ms`` milliseconds; the host is synced once, at the end.

        The returned counts cover *this call only*, while ``self._counts`` and
        ``self.stats`` keep accumulating over the whole life of the network -- so mixing
        :meth:`step` with :meth:`run` never loses spikes.
        """
        steps = max(1, int(round(ms / self.p.dt)))
        if self._graph is None and not self._graph_failed:
            self.capture_graph()
        # Capacity peaks are reported per run, not for the life of the object: once a
        # single hot step had been seen, a sticky warning would repeat on every control
        # tick (20 Hz) long after the network cooled down.
        self._max_spk.zero_()
        self._max_ev.zero_()
        counts_before = self._counts.clone()
        spike_before = self._spike_acc.clone()
        syn_before = self._syn_acc.clone()
        rec_idx = None
        if record is not None and len(record):
            rec_idx = torch.from_numpy(np.asarray(record, dtype=np.int64)).to(self.device)
        rec_rows: list[torch.Tensor] = []
        rec_times: list[float] = []
        for _ in range(steps):
            self._one_step()
            if rec_idx is not None:
                # Collect only the recorded neurons' spike flags.  This is a fixed-size
                # tensor (len(record)), so the loop performs no device->host copy and no
                # sync -- a per-step ``.cpu()`` here costs ~2000 syncs per call and takes
                # the backend from 0.89x back down to 0.33x.
                rec_rows.append(self._spk_mask.index_select(0, rec_idx))
                rec_times.append(self.t_ms)
        counts = (self._counts - counts_before).clamp_(min=0).cpu().numpy().astype(np.int32)
        raster: list[tuple[float, np.ndarray]] = []
        if rec_rows:
            flags = torch.stack(rec_rows).cpu().numpy()      # the single sync for raster
            for t, row in zip(rec_times, flags):
                pos = np.flatnonzero(row)
                if pos.size:
                    raster.append((t, pos))
        self.stats.spikes += int((self._spike_acc - spike_before).item())
        self.stats.synaptic_events += int((self._syn_acc - syn_before).item())
        self._capacity_warning()
        return counts, raster
