"""Backend selection must work in environments without PyTorch.

The CPU path is the default and must keep working when torch is not installed,
so this module deliberately does NOT import torch.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from flydrones.brain.brain import make_network
from flydrones.brain.lif import LIFNetwork, LIFParams


def _tiny():
    return sp.csc_matrix(np.eye(4, dtype=np.float32))


def test_default_backend_is_cpu():
    assert isinstance(make_network(_tiny(), LIFParams(), 0, {}), LIFNetwork)


def test_explicit_cpu_backend():
    assert isinstance(make_network(_tiny(), LIFParams(), 0, {"backend": "cpu"}), LIFNetwork)


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown brain.backend"):
        make_network(_tiny(), LIFParams(), 0, {"backend": "quantum"})


def test_auto_backend_never_raises():
    """auto must degrade to cpu when torch/CUDA is unavailable, not crash."""
    net = make_network(_tiny(), LIFParams(), 0, {"backend": "auto"})
    assert net is not None
    assert hasattr(net, "step") and hasattr(net, "run")


def test_torch_request_without_torch_is_explicit():
    """An explicit torch request must raise (never silently fall back)."""
    try:
        import torch  # noqa: F401
        pytest.skip("torch is installed; this checks the missing-torch path")
    except ImportError:
        pass
    with pytest.raises(ImportError):
        make_network(_tiny(), LIFParams(), 0, {"backend": "torch"})
