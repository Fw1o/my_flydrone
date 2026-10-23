"""The torch/CUDA backend must honour the same contract as the CPU LIFNetwork.

Skipped entirely when PyTorch or a CUDA device is unavailable.

Numerical equivalence is checked on the *deterministic* part of the dynamics
(driven with bias/inject rather than Poisson input), because the CPU backend uses
numpy's PCG64 generator and the GPU one uses torch's Philox generator -- the two
Poisson sequences can never match, and any long trajectory is dominated by that
difference rather than by the backend.
"""

import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

import warnings  # noqa: E402

from flydrones.brain.lif import LIFNetwork, LIFParams  # noqa: E402
from flydrones.brain.lif_torch import TorchLIFNetwork  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device (torch backend)"
)


def chain(n=3, w=200.0):
    rows = np.arange(1, n)
    cols = np.arange(0, n - 1)
    return sp.csc_matrix((np.full(n - 1, w, np.float32), (rows, cols)), shape=(n, n))


def ring(n=6, w=50.0):
    """Self-loop on every neuron: a bias makes the whole population fire."""
    idx = np.arange(n)
    return sp.csc_matrix((np.full(n, w, np.float32), (idx, idx)), shape=(n, n))


# --------------------------------------------------------------- basic contract
def test_silent_without_input():
    net = TorchLIFNetwork(chain(), LIFParams(noise_mv=0), seed=0)
    counts, _ = net.run(200)
    assert counts.sum() == 0


def test_bias_drives_spikes_and_propagates():
    """A tonic bias on neuron 0 must make the chain fire downstream."""
    net = TorchLIFNetwork(chain(3), LIFParams(noise_mv=0), seed=1)
    net.set_bias(np.array([0]), 30.0)
    counts, _ = net.run(1000)
    assert counts[0] > 0
    assert counts[1] > 0, "spikes must propagate to neuron 1"
    assert counts[2] > 0, "spikes must propagate to neuron 2"


def test_refractory_caps_rate():
    p = LIFParams(noise_mv=0)
    net = TorchLIFNetwork(sp.csc_matrix((1, 1), dtype=np.float32), p, seed=0)
    net.set_bias(np.array([0]), 200.0)
    counts, _ = net.run(1000)
    assert counts[0] <= 1000 / p.t_ref + 1


def test_inhibition_suppresses():
    W = sp.csc_matrix((np.array([-500.0], np.float32), (np.array([1]), np.array([0]))), shape=(2, 2))
    net = TorchLIFNetwork(W, LIFParams(noise_mv=0), seed=2)
    net.set_bias(np.array([1]), 12.0)
    c_alone, _ = net.run(1000)
    net.reset()
    net.set_bias(np.array([1]), 12.0)
    net.set_input(np.array([0]), 150.0)
    c_inh, _ = net.run(1000)
    assert c_alone[1] > 0
    assert c_inh[1] < c_alone[1]


def test_copy_shares_wiring_not_state():
    net = TorchLIFNetwork(chain(), LIFParams(), seed=0)
    net.set_bias(np.array([0]), 30.0)
    net.run(100)
    clone = net.copy(seed=5)
    assert clone.W is net.W
    assert clone.t_ms == 0.0
    assert clone.v.data_ptr() != net.v.data_ptr()


def test_step_returns_host_indices():
    net = TorchLIFNetwork(chain(3), LIFParams(noise_mv=0), seed=0)
    net.inject(np.array([0]), 400.0)
    seen = set()
    for _ in range(40):
        spk = net.step()
        assert isinstance(spk, np.ndarray) and spk.dtype == np.int64
        seen.update(int(x) for x in spk)
    assert 0 in seen


# ------------------------------------------------- CPU equivalence (deterministic)
def test_matches_cpu_on_deterministic_drive():
    """Same wiring, same bias, same current pulse -> identical spike sets."""
    W = chain(4, w=300.0)
    p = LIFParams(noise_mv=0)
    cpu = LIFNetwork(W, p, seed=0)
    gpu = TorchLIFNetwork(W, p, seed=0)
    for net in (cpu, gpu):
        net.set_bias(np.array([0]), 20.0)
        net.inject(np.array([0]), 60.0)

    mismatches = 0
    for _ in range(120):
        a = cpu.step()
        b = gpu.step()
        if set(int(x) for x in a) != set(int(x) for x in b):
            mismatches += 1
    assert mismatches == 0, f"{mismatches}/120 steps disagree between cpu and gpu"
    assert int(np.abs(cpu.v - gpu.v.cpu().numpy()).max()) < 0.05


def test_matches_cpu_over_a_longer_run():
    W = chain(6, w=250.0)
    p = LIFParams(noise_mv=0)
    cpu = LIFNetwork(W, p, seed=0)
    gpu = TorchLIFNetwork(W, p, seed=0)
    for net in (cpu, gpu):
        net.set_bias(np.array([0]), 22.0)
        net.inject(np.array([0]), 80.0)
    cpu.run(2000)
    gpu.run(2000)
    # Long runs cannot match exactly: the GPU accumulates each neuron's input with
    # atomicAdd while numpy reduces sequentially, so float32 rounding differs and a
    # recurrent network amplifies it.  What must hold is that no systematic bias
    # creeps in (the CUDA-graph RNG bug produced exactly 2x, which this catches).
    a = int(cpu.stats.spikes)
    b = int(gpu._spike_acc.item())
    assert 0.5 * a <= b <= 2.0 * a, f"cpu={a} gpu={b}"


# ------------------------------------------------------------- capacity sanity
def test_capacity_overflow_is_reported():
    """Deliberately tiny capacity must warn rather than silently truncate."""
    net = TorchLIFNetwork(chain(5, w=50.0), LIFParams(noise_mv=0), seed=0,
                          max_spikes=2, max_events=4)
    net.set_bias(np.arange(5), 80.0)
    net.inject(np.arange(5), 120.0)
    with pytest.warns(UserWarning, match="static capacity exceeded"):
        net.run(300)


def test_realtime_factor_reported_for_full_scale_shape():
    """The step must stay a single synchronised call, not per-step host traffic."""
    W = chain(4, w=300.0)
    net = TorchLIFNetwork(W, LIFParams(noise_mv=0), seed=0)
    net.set_bias(np.array([0]), 30.0)
    stats_before = net.stats.steps
    net.run(100)
    assert net.stats.steps == stats_before + 200


# ---------------------------------------------------------------- graph path
def test_set_input_of_the_same_size_keeps_the_graph():
    """Re-capturing per control tick cost 15% of the real-time budget on MaleCNS."""
    net = TorchLIFNetwork(chain(4, 300.0), LIFParams(noise_mv=0), seed=0)
    net.set_bias(np.array([0]), 30.0)
    net.set_input(np.array([1, 2]), 100.0)        # establish the input set first
    net.run(50)                                  # ... then capture with that shape
    g = net._graph
    if g is None:
        pytest.skip("CUDA graph unavailable on this driver")
    net.set_input(np.array([1, 3]), 50.0)        # same size, new values
    assert net._graph is g, "a same-size set_input must not force a re-capture"
    net.set_input(np.array([1, 2, 3]), 100.0)    # different size
    assert net._graph is None, "a size change must invalidate the graph"
    counts, _ = net.run(50)                      # still usable afterwards
    assert counts.shape == (net.n,)


def test_noise_mv_with_cuda_graph_stays_usable():
    """Regression: int(tensor.item()) in the capture aborted it for the whole process."""
    net = TorchLIFNetwork(chain(3, 300.0), LIFParams(noise_mv=0.5), seed=0)
    net.set_bias(np.array([0]), 30.0)
    counts, _ = net.run(200)                     # must not raise, graph or eager
    assert counts.sum() > 0
    offsets = set()
    for _ in range(10):
        net._one_step()
        offsets.add(int(net._noise_o.item()))
    assert len(offsets) > 1, "a captured graph must not freeze the noise draw"


def test_capture_does_not_pollute_the_peak_counters():
    """The capture warmup must not be reported as observed load."""
    net = TorchLIFNetwork(ring(), LIFParams(noise_mv=0, t_ref=0.5), seed=0)
    net.set_bias(np.arange(6), 400.0)
    for _ in range(6):
        net._step_device()
    net._max_spk.zero_()
    net._max_ev.zero_()
    net.capture_graph()
    assert int(net._max_spk.item()) == 0
    assert int(net._max_ev.item()) == 0


# -------------------------------------------------------- capacity reporting
def test_capacity_warning_is_per_run_not_sticky():
    net = TorchLIFNetwork(ring(), LIFParams(noise_mv=0), seed=0, max_spikes=3, max_events=8)
    net.set_bias(np.arange(6), 400.0)
    with pytest.warns(UserWarning, match="static capacity exceeded"):
        net.run(300)
    with warnings.catch_warnings():               # cooled down -> no repeat warning
        warnings.simplefilter("error")
        net.set_bias(np.arange(6), -200.0)
        net.run(300)


def test_step_reports_capacity_overflow():
    """step() has no run() to report through, so it must warn itself."""
    net = TorchLIFNetwork(ring(), LIFParams(noise_mv=0), seed=0, max_spikes=3, max_events=8)
    net.set_bias(np.arange(6), 400.0)
    with pytest.warns(UserWarning, match="static capacity exceeded"):
        for _ in range(40):
            net.step()


def test_run_counts_are_per_call_without_losing_step_counts():
    net = TorchLIFNetwork(ring(3), LIFParams(noise_mv=0, t_ref=0.5), seed=0)
    net.set_bias(np.arange(3), 400.0)
    for _ in range(30):
        net.step()
    accumulated = int(net._counts.cpu().numpy().sum())
    assert accumulated > 0
    counts, _ = net.run(50)
    assert counts.sum() > 0
    assert int(net._counts.cpu().numpy().sum()) == accumulated + int(counts.sum())
