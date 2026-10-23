import numpy as np
import pytest
import scipy.sparse as sp

from flydrones.brain import Brain, Connectome, build_minifly
from flydrones.config import load_config


def brain():
    cfg = load_config()
    return Brain(build_minifly(), cfg), cfg


def test_minifly_groups_resolve():
    b, cfg = brain()
    for name in list(cfg["inputs"]) + list(cfg["outputs"]):
        assert b.connectome.group(name).size > 0, name


def test_optomotor_and_lift_responses():
    b, _ = brain()
    b.tick({}, 800)
    rest = b.tick({}, 800)
    up = b.tick({"T4c_L": 80, "T4c_R": 80}, 800)
    b.tick({}, 400)
    right = b.tick({"T4a_R": 80, "T4b_L": 80}, 800)
    lift_rest = rest["DNg02_L"] + rest["DNg02_R"]
    assert up["DNg02_L"] + up["DNg02_R"] > lift_rest * 1.3
    assert right["DNg02_R"] > right["DNg02_L"] + 8


def test_looming_drives_giant_fiber_and_saccade_side():
    b, _ = brain()
    b.tick({}, 500)
    r = b.tick({"LPLC2_L": 150, "LC4_L": 120}, 600)
    assert r["DNp01_L"] > r["DNp01_R"]
    assert r["DNp03_L"] > r["DNp03_R"]


def test_save_load_roundtrip(tmp_path):
    c = build_minifly()
    c.groups["x"] = np.array([1, 2, 3])
    path = c.save(tmp_path / "b.npz")
    d = Connectome.load(path)
    assert d.n == c.n and d.n_connections == c.n_connections
    assert list(d.group("x")) == [1, 2, 3]


def test_sensorimotor_core_keeps_io():
    b, _ = brain()
    core = b.connectome.sensorimotor_core(hops=4)
    assert core.n <= b.connectome.n
    assert core.group("DNg02_L").size == b.connectome.group("DNg02_L").size


def test_swarm_copy_independent():
    b, _ = brain()
    c = b.copy(seed=99)
    assert c.connectome is b.connectome
    b.tick({"T4c_L": 100}, 200)
    assert c.net.t_ms == 0.0


def test_bias_group_with_no_neurons_warns():
    """A bias entry that resolves to nobody used to be dropped without a word."""
    cfg = load_config()
    cfg["brain"]["bias"] = {"DNg02_L": 9.0, "DNg02_typo": 8.0}
    with pytest.warns(UserWarning, match="DNg02_typo"):
        Brain(build_minifly(), cfg)


def test_bias_accepts_a_cell_type_pattern():
    """Real configs bias DNg02 subtypes, which are never `inputs`/`outputs` groups."""
    cfg = load_config()
    cfg["brain"]["bias"] = {"T4c": 5.0}          # a cell type, not a group name
    b = Brain(build_minifly(), cfg)
    assert np.count_nonzero(np.asarray(b.net.bias)) > 0


def test_bias_with_invalid_pattern_warns_instead_of_crashing():
    cfg = load_config()
    cfg["brain"]["bias"] = {"DNg02[": 1.0}       # unbalanced bracket -> bad regex
    with pytest.warns(UserWarning, match="not a valid cell-type pattern"):
        Brain(build_minifly(), cfg)


# --------------------------------------------------- left/right side inference
def test_infer_missing_sides_from_wiring():
    """A neuron wired only to left-side partners should be labelled left."""
    # column 1 holds the partners of neuron 1 (i.e. 1 -> 0 and 1 -> 2)
    W = sp.csc_matrix(
        (np.float32([1.0, 1.0]), (np.array([0, 2]), np.array([1, 1]))), shape=(3, 3)
    )
    c = Connectome(
        name="toy", weights=W,
        types=np.array(["a", "b", "c"]),
        sides=np.array(["L", "", "L"]),
    )
    assert c.infer_missing_sides() == 1
    assert c.sides[1] == "L"
    assert c.meta["inferred_sides"] == 1


def test_infer_missing_sides_leaves_ambiguous_cells_alone():
    """A 1-vs-1 split must stay unlabelled instead of being guessed."""
    W = sp.csc_matrix(
        (np.float32([1.0, 1.0]), (np.array([0, 2]), np.array([1, 1]))), shape=(3, 3)
    )
    c = Connectome(
        name="toy", weights=W,
        types=np.array(["a", "b", "c"]),
        sides=np.array(["L", "", "R"]),
    )
    assert c.infer_missing_sides() == 0
    assert c.sides[1] == ""


def test_load_fills_missing_sides_for_old_files(tmp_path):
    """Old brain files ship without somaSide for some populations."""
    c = build_minifly()
    sides = c.sides.astype(str).copy()
    mask = c.types.astype(str) == "T4c"          # one whole population, as the real data
    assert mask.sum() > 0.05 * c.n
    sides[mask] = ""
    c.sides = sides
    path = c.save(tmp_path / "old.npz")

    with pytest.warns(UserWarning, match="without a left/right label"):
        d = Connectome.load(path)
    assert np.count_nonzero(d.sides.astype(str) == "") < int(mask.sum())

    raw = Connectome.load(path, infer_sides=False)   # explicit opt-out stays raw
    assert np.count_nonzero(raw.sides.astype(str) == "") == int(mask.sum())
