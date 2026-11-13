"""Actual macOS AUv2 hosting, native physical curves, graph clocks and MIDI."""

import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import Chain, Gain, Mix, Pedalboard
from pedalboard_native._internal import AddLatency

from .au_automation_plugin.register import register_bundle

if platform.system() == "Darwin":
    from pedalboard import AudioUnitPlugin

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="Audio Units require macOS")


@pytest.fixture(scope="session")
def au_probe(tmp_path_factory):
    source = Path(__file__).parent / "au_automation_plugin"
    # Imported fixture aliases must share one process-local AU registration.
    # CoreAudio rejects a second factory for the same component description.
    build = tmp_path_factory.getbasetemp() / "au-automation-build"
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.fail("AU automation tests require CMake and a C++17 compiler")
    for command in [
        [cmake, "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        [cmake, "--build", str(build), "--parallel", "2"],
    ]:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    return register_bundle(build / "AutomationAU.component")


def load(path, name="Automation AU"):
    return AudioUnitPlugin(path, plugin_name=name)


def signal(n=1025):
    return np.random.default_rng(178).uniform(-0.4, 0.4, (2, n)).astype(np.float32)


def curves(n):
    return np.linspace(-30, 6, n), np.geomspace(20, 20000, n)


def expected(x, gain, frequency, maximum=20000):
    raw_g = ((gain + 60) / 72).astype(np.float32).astype(float)
    raw_f = ((frequency - 20) / (maximum - 20)).astype(np.float32).astype(float)
    # The host's normalized float32 target is scaled to the AU's float32 units.
    g = (-60 + 72 * raw_g).astype(np.float32).astype(float)
    f = (20 + (maximum - 20) * raw_f).astype(np.float32).astype(float)
    return (x.astype(float) * 10 ** (g / 20) * f / 20000).astype(np.float32)


@pytest.mark.parametrize("rate", [22050, 48000, 96000])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
@pytest.mark.parametrize("layout", ["mono", "planar", "interleaved"])
def test_db_and_hz_curves_match_actual_au_audio(au_probe, rate, buffer_size, layout):
    x = signal()
    gain, frequency = curves(x.shape[1])
    reference = expected(x, gain, frequency)
    if layout == "mono":
        x, reference = x[0], reference[0]
    elif layout == "interleaved":
        x, reference = x.T, reference.T
    p = load(au_probe)
    result = p(
        x, rate, buffer_size=buffer_size, automation={"gain_db": gain, "frequency_hz": frequency}
    )
    np.testing.assert_allclose(result, reference, atol=3e-7, rtol=3e-7)
    assert abs(float(p._get_parameter("Gain").string_value) - 6) < 1e-4
    assert abs(float(p._get_parameter("Frequency").string_value) - 20000) < 0.01


@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
def test_delay_keeps_curves_on_their_source_frames(au_probe, buffer_size):
    x = signal()
    gain, frequency = curves(x.shape[1])
    p = load(au_probe, "Automation AU Delayed")
    result = p(
        x, 48000, buffer_size=buffer_size, automation={"gain_db": gain, "frequency_hz": frequency}
    )
    assert p.reported_latency_samples == 20
    np.testing.assert_allclose(result, expected(x, gain, frequency), atol=3e-7, rtol=3e-7)


def test_constant_runs_coalesce_and_all_knobs_change_together(au_probe):
    p = load(au_probe)
    x = signal(257)
    gain = np.r_[np.full(128, -12.0), np.full(129, 3.0)]
    freq = np.r_[np.full(128, 200.0), np.full(129, 10000.0)]
    actual = p(x, 48000, automation={"frequency_hz": freq, "gain_db": gain})
    np.testing.assert_allclose(actual, expected(x, gain, freq), atol=3e-7, rtol=3e-7)
    assert int(p._get_parameter("Blocks").string_value) == 2


@pytest.mark.parametrize("name", ["Automation AU", "Automation AU Delayed"])
def test_sliced_streams_and_omitted_lanes_hold_values(au_probe, name):
    x = signal(1537)
    gain, frequency = curves(1537)

    def run(sliced):
        p = load(au_probe, name)
        p.reset()
        pieces, start = [], 0
        for end, size in (
            [(8, 31), (9, 1), (11, 8192), (257, 31), (1537, 8192)] if sliced else [(1537, 31)]
        ):
            pieces.append(
                p(
                    x[:, start:end],
                    48000,
                    buffer_size=size,
                    reset=False,
                    automation={"gain_db": gain[start:end], "frequency_hz": frequency[start:end]},
                )
            )
            start = end
        pieces.append(p(np.zeros((2, 100), np.float32), 48000, reset=False))
        return np.concatenate(pieces, axis=1)

    assert_array_equal(run(True), run(False))


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("leading_delay", [0, 701])
def test_graph_pending_au_curves_and_builtin_nodes(au_probe, parallel, leading_delay):
    x = signal(1025)
    g, f = curves(1025)

    def run(sliced):
        p, gain = load(au_probe, "Automation AU Delayed"), Gain(-3)
        branch = Chain([AddLatency(leading_delay), p, gain])
        board = Pedalboard([Mix([branch, Gain(-6)])]) if parallel else Pedalboard([branch])
        board.reset()
        parts, start = [], 0
        for end in [13, 511, 1025] if sliced else [1025]:
            parts.append(
                board(
                    x[:, start:end],
                    48000,
                    reset=False,
                    automation={
                        p: {"gain_db": g[start:end], "frequency_hz": f[start:end]},
                        gain: {"gain_db": np.full(end - start, -3.0)},
                    },
                )
            )
            start = end
        parts.append(board(np.zeros((2, 1600), np.float32), 48000, reset=False))
        return np.concatenate(parts, axis=1)

    assert_array_equal(run(True), run(False))


@pytest.mark.parametrize(
    "parameter,curve,error",
    [
        ("gain_db", np.zeros(3), ValueError),
        ("gain_db", np.full(257, np.nan), ValueError),
        ("frequency_hz", np.full(257, 30000.0), ValueError),
        ("gain_db", np.zeros(257, dtype=complex), TypeError),
        ("gain_db", np.zeros((257, 1)), ValueError),
        ("gain_db", [0] * 257, TypeError),
        ("blocks", np.zeros(257), ValueError),
        ("setup", np.zeros(257), ValueError),
        ("broken", np.zeros(257), ValueError),
        ("missing", np.zeros(257), ValueError),
    ],
)
def test_invalid_late_lane_does_not_change_prior_parameters(au_probe, parameter, curve, error):
    p = load(au_probe)
    x = signal(257)
    p(x, 48000, automation={"gain_db": np.full(257, -9.0)})
    before = p._get_parameter("Gain").raw_value
    targets = {"bypass": np.ones(257, dtype=bool), parameter: curve}
    with pytest.raises(error):
        p(x, 48000, automation=targets)
    assert p._get_parameter("Gain").raw_value == before
    assert not p._get_parameter("Bypass").raw_value


@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
@pytest.mark.parametrize("rate", [22050, 44100, 48000, 96000])
def test_instrument_notes_and_curves_share_input_clock(au_probe, buffer_size, rate):
    p = load(au_probe, "Automation AU Instrument")
    duration, n = 0.125, int(rate * 0.125)
    intervals = [(7, 45), (127, 511), (1025, n - 1)]
    notes, mask = [], np.zeros(n)
    for start, end in intervals:
        notes.extend([([144, 60, 127], start / rate), ([128, 60, 0], end / rate)])
        mask[start:end] = 1
    g, f = curves(n)
    actual = p(
        notes, duration, rate, buffer_size=buffer_size, automation={"gain_db": g, "frequency_hz": f}
    )
    reference = expected(np.tile(mask, (2, 1)), g, f)
    np.testing.assert_allclose(actual, reference, atol=3e-7, rtol=3e-7)
    assert not p([], duration, rate, buffer_size=buffer_size, reset=False).any()


def test_instrument_held_note_and_final_targets_survive_scalar_continuation(au_probe):
    p = load(au_probe, "Automation AU Instrument")
    n, duration, rate = 1500, 0.03125, 48000
    g, f = curves(n)
    a = p([([144, 60, 127], 0)], duration, rate, automation={"gain_db": g, "frequency_hz": f})
    b = p([], duration, rate, reset=False, buffer_size=31)
    assert_array_equal(b, np.repeat(a[:, -1:], n, axis=1))
    assert p([], 0, rate, automation={"gain_db": np.zeros(0)}).shape == (2, 0)
    with pytest.raises(ValueError):
        Pedalboard([p])


@pytest.mark.parametrize("rate", [22050, 48000, 96000])
def test_prepared_rate_dependent_hz_range(au_probe, rate):
    p = load(au_probe, "Automation AU Rate Dependent")
    n = 129
    frequency = np.geomspace(100, 10000, n)
    y = p(
        np.ones(n, np.float32), rate, automation={"gain_db": np.zeros(n), "frequency_hz": frequency}
    )
    np.testing.assert_allclose(
        y, expected(np.ones(n, np.float32), np.zeros(n), frequency, rate / 2), atol=3e-7, rtol=3e-7
    )
    p.prepare_automation(96000, num_channels=1)
    y = p(
        np.ones(n, np.float32),
        96000,
        reset=False,
        automation={"gain_db": np.zeros(n), "frequency_hz": np.full(n, 30000)},
    )
    np.testing.assert_allclose(y, 1.5, atol=2e-6)


def test_mode_and_warm_scalar_transition_require_explicit_restart(au_probe):
    p = load(au_probe)
    assert p.automation_mode == "sample_steps"
    with pytest.raises(ValueError, match="sample_steps"):
        p.automation_mode = "vst3_events"
    x = signal(257)
    p(x, 48000)
    with pytest.raises(ValueError, match="reset"):
        p(x, 48000, reset=False, automation={"gain_db": np.zeros(257)})
    p.reset()
    p(x, 48000, reset=False, automation={"gain_db": np.zeros(257)})
    with pytest.raises(ValueError, match="format"):
        p(x, 44100, reset=False)


def test_boolean_and_infinite_display_endpoint(au_probe):
    x = signal(257)
    p = load(au_probe)
    bypass = np.arange(257) % 2 == 0
    y = p(x, 48000, automation={"bypass": bypass})
    np.testing.assert_allclose(y, np.where(bypass, x, x * 0.001), atol=3e-7, rtol=3e-7)
    p = load(au_probe, "Automation AU Infinite Gain")
    target = np.r_[-np.inf, np.linspace(-60, 0, 256)]
    y = p(x, 48000, automation={"gain": target})
    raw = (10 ** (target / 20) / 2).astype(np.float32).astype(float)
    np.testing.assert_allclose(y, x * (2 * raw) * 0.001, atol=3e-7, rtol=3e-7)


def test_strided_readonly_curves_are_owned_and_empty_calls_do_not_reset(au_probe):
    p = load(au_probe)
    x = signal(257)
    g, f = curves(257)
    g = np.repeat(g, 2)[::2]
    g.flags.writeable = f.flags.writeable = x.flags.writeable = False
    y = p(x, 48000, automation={"gain_db": g, "frequency_hz": f})
    np.testing.assert_allclose(y, expected(x, g, f), atol=3e-7, rtol=3e-7)
    before = p._get_parameter("Gain").raw_value
    assert p(np.zeros((2, 0), np.float32), 48000, automation={"gain_db": np.zeros(0)}).shape == (
        2,
        0,
    )
    assert p._get_parameter("Gain").raw_value == before


@pytest.mark.parametrize("force_reload", [False, True])
def test_stable_ids_survive_resets_and_reinstantiation(au_probe, force_reload):
    from pedalboard import ExternalPluginReloadType

    p = load(au_probe)
    x = signal(257)
    g, f = curves(257)
    reference = p(x, 48000, automation={"gain_db": g, "frequency_hz": f})
    if force_reload:
        type(p)._reload_type.__set__(p, ExternalPluginReloadType.PersistsAudioOnReset)
    for _ in range(2):
        p.reset()
        actual = p(x, 48000, automation={"gain_db": g, "frequency_hz": f})
        assert_array_equal(actual, reference)


def test_real_chow_au_accepts_db_hz_and_matches_constant_scalar_path():
    path = Path(__file__).parent / "plugins/effect/Darwin/CHOWTapeModel.component"
    assert path.exists(), "Repository CHOW AU fixture is required on macOS"
    path = register_bundle(path)
    p = AudioUnitPlugin(path)
    x = signal(257)
    y = p(
        x,
        48000,
        automation={
            "input_gain": np.linspace(-12, 6, 257),
            "input_low_cut": np.geomspace(100, 1000, 257),
            "input_filters_on_off": np.ones(257, dtype=bool),
        },
    )
    assert y.shape == x.shape and np.isfinite(y).all()
    assert abs(float(p._get_parameter("Input Gain").string_value) - 6) < 0.02
    fixed, scalar = AudioUnitPlugin(path), AudioUnitPlugin(path)
    # The legacy Python wrapper guesses quantized ranges (e.g. -6 -> -5.99).
    # Compare identical controller targets, rather than that separate approximation.
    for name, target in [("Input Gain", "-6"), ("Input Low Cut", "1000")]:
        parameter = scalar._get_parameter(name)
        parameter.raw_value = parameter.get_raw_value_for_text(target)
    a = fixed(
        x,
        48000,
        automation={"input_gain": np.full(257, -6.0), "input_low_cut": np.full(257, 1000.0)},
    )
    b = scalar(x, 48000)
    np.testing.assert_allclose(a, b, atol=3e-7, rtol=3e-7)


def test_reset_and_audio_render_share_the_au_instance_lock(au_probe):
    from concurrent.futures import ThreadPoolExecutor

    p = load(au_probe)
    x = signal(257)
    g, f = curves(257)
    reference = expected(x, g, f)

    def render_work():
        for _ in range(12):
            actual = p(x, 48000, automation={"gain_db": g, "frequency_hz": f})
            np.testing.assert_allclose(actual, reference, atol=3e-7, rtol=3e-7)

    def reset_work():
        for _ in range(12):
            p.reset()

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(render_work), pool.submit(reset_work)
        a.result(timeout=20)
        b.result(timeout=20)


@pytest.mark.parametrize("mutate", [False, True])
def test_graph_pending_curves_are_owned_and_omitted_lanes_hold(au_probe, mutate):
    p = load(au_probe)
    board = Pedalboard([AddLatency(701), p])
    x = signal(1025)
    g, f = curves(1025)
    reference_f = np.r_[f[:513], np.full(512, f[512])]
    board.reset()
    a = board(
        x[:, :513],
        48000,
        reset=False,
        automation={p: {"gain_db": g[:513], "frequency_hz": f[:513]}},
    )
    if mutate:
        g[:513], f[:513] = 12, 20000
    b = board(x[:, 513:], 48000, reset=False, automation={p: {"gain_db": g[513:]}})
    c = board(np.zeros((2, 2000), np.float32), 48000, reset=False)
    actual = np.concatenate([a, b, c], axis=1)[:, :1025]
    reference_g = curves(1025)[0]
    np.testing.assert_allclose(actual, expected(x, reference_g, reference_f), atol=3e-7, rtol=3e-7)


@pytest.mark.parametrize("method", ["process", "__call__"])
def test_real_magical_instrument_au_accepts_midi_and_gain_curves(method):
    path = Path(__file__).parent / "plugins/instrument/Darwin/Magical8bitPlug2.component"
    assert path.exists(), "Repository instrument AU fixture is required on macOS"
    p = AudioUnitPlugin(register_bundle(path))
    result = getattr(p, method)(
        [([144, 60, 127], 0), ([128, 60, 0], 0.1)],
        0.125,
        48000,
        automation={"gain": np.linspace(0.1, 0.5, 6000)},
    )
    assert result.shape == (2, 6000) and np.isfinite(result).all()
    assert np.max(np.abs(result)) > 0.001
    assert abs(float(p._get_parameters()["gain"].string_value) - 0.5) < 0.001


def test_special_endpoint_display_does_not_require_a_full_string_parser(au_probe):
    p = load(au_probe, "Automation AU Partial Display")
    x = signal(257)
    gain = np.r_[-np.inf, np.linspace(-59, 0, 256)]
    result = p(x, 48000, automation={"gain_db": gain})
    reference = expected(x, np.r_[-60, gain[1:]], np.full(257, 20))
    reference[:, 0] = 0
    np.testing.assert_allclose(result, reference, atol=3e-7, rtol=3e-7)
