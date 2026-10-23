from flydrones.brain import Brain, build_minifly
from flydrones.calibrate import fit_readout, record_responses
from flydrones.cli import main
from flydrones.config import load_config
from flydrones.drones import SimDrone
from flydrones.runtime import Pilot, run_sim
from flydrones.senses import ScriptedGestures, demo_timeline


def test_gesture_demo_climbs_holds_descends():
    cfg = load_config()
    p = Pilot(Brain(build_minifly(), cfg), SimDrone(start=(-1.5, 0, 0)), cfg, gestures=ScriptedGestures(demo_timeline()))
    run_sim([p], 21)
    alt = {round(h["t"], 2): h["alt"] for h in p.history}
    assert alt[4.5] > alt[2.5] + 0.3  # open palm -> climb
    assert abs(alt[9.0] - alt[6.0]) < 0.25  # fist -> hold
    assert alt[20.95] < alt[15.5] - 0.5  # hand dropped -> descend
    assert p.drone.collisions == 0


def test_calibration_recovers_signs():
    cfg = load_config()
    b = Brain(build_minifly(), cfg)
    outs, X, Y = record_responses(b, cfg, settle_ms=200, measure_ms=600, repeats=1)
    res = fit_readout(outs, X, Y)
    yaw = res["axes"]["yaw"]["terms"]
    thr = res["axes"]["throttle"]["terms"]
    assert yaw["DNg02_R"] > 0 > yaw["DNg02_L"]
    assert thr["DNg02_L"] + thr["DNg02_R"] > 0


def test_cli_demo_and_bench(tmp_path, capsys):
    assert main(["bench", "--ms", "200"]) == 0
    assert main(["demo", "--seconds", "3", "--log", str(tmp_path / "log.csv")]) == 0
    assert (tmp_path / "log.csv").exists()


def test_cli_fly_sim_dry(capsys):
    assert main(["fly", "--drone", "sim", "--input", "camera", "--seconds", "3"]) == 0


def test_load_config_merges_multiple_files(tmp_path):
    """A flight combines a brain config with a drone config: later files win."""
    (tmp_path / "brain.yaml").write_text(
        "brain:\n"
        "  source: data/brain.npz\n"
        "  torch:\n"
        "    max_events: 262144\n"
        "safety:\n"
        "  max_throttle: 0.9\n",
        encoding="utf-8",
    )
    (tmp_path / "drone.yaml").write_text(
        "brain:\n"
        "  torch:\n"
        "    max_events: 16384\n"
        "safety:\n"
        "  max_throttle: 0.5\n"
        "vision:\n"
        "  flow_gain: 2.0\n",
        encoding="utf-8",
    )
    cfg = load_config([tmp_path / "brain.yaml", tmp_path / "drone.yaml"])
    assert cfg["brain"]["source"] == "data/brain.npz"          # only in the first file
    assert cfg["brain"]["torch"]["max_events"] == 16384        # second file wins
    assert cfg["brain"]["torch"]["max_spikes"] == 4096         # untouched default
    assert cfg["safety"]["max_throttle"] == 0.5                # second file wins
    assert cfg["safety"]["max_alt_m"] == 2.2                   # untouched default
    assert cfg["vision"]["flow_gain"] == 2.0


def test_load_config_single_path_and_missing_file(tmp_path):
    single = tmp_path / "one.yaml"
    single.write_text("safety:\n  max_yaw: 0.3\n", encoding="utf-8")
    assert load_config(single)["safety"]["max_yaw"] == 0.3       # single path still works
    assert load_config()["brain"]["backend"] == "cpu"            # no config at all
    try:
        load_config([tmp_path / "missing.yaml"])
    except FileNotFoundError as e:
        assert "missing.yaml" in str(e)
    else:
        raise AssertionError("a missing --config file must raise, not silently pass")


def test_trailing_empty_key_does_not_wipe_a_section(tmp_path):
    """`decoder:` with only comments under it parses as None - it must not clobber defaults."""
    bad = tmp_path / "empty_section.yaml"
    bad.write_text("safety:\n  max_yaw: 0.3\ndecoder:\n  # nothing here\n", encoding="utf-8")
    cfg = load_config([bad])
    assert cfg["safety"]["max_yaw"] == 0.3
    assert isinstance(cfg["decoder"], dict), "an empty YAML key must not turn a section into None"
    assert cfg["decoder"]["axes"]["throttle"]["gain"] > 0


def test_every_shipped_config_loads_and_keeps_the_decoder():
    """Every configs/*.yaml must leave a usable brain/decoder (catches empty-key mistakes)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    configs = sorted(root.glob("*.yaml"))
    assert configs, "no shipped configs found"
    for path in configs:
        cfg = load_config([path])
        assert isinstance(cfg.get("brain"), dict), f"{path.name}: brain is not a mapping"
        assert isinstance(cfg.get("decoder"), dict), f"{path.name}: decoder is not a mapping"
        assert cfg["decoder"].get("axes"), f"{path.name}: decoder.axes is missing"
        assert cfg["brain"].get("source"), f"{path.name}: brain.source is missing"
