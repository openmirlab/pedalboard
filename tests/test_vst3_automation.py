"""Actual VST3 loading, physical-unit conversion and native timed delivery."""

import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import Chain, Gain, Mix, Pedalboard, VST3Plugin
from pedalboard_native._internal import AddLatency

MODES = ["sample_steps", "vst3_events"]


@pytest.fixture(scope="session")
def vst3_probe(tmp_path_factory):
    source = Path(__file__).parent / "vst3_automation_plugin"
    build = tmp_path_factory.mktemp("vst3-automation-build")
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.fail("VST3 automation tests require CMake and a C++17 compiler")
    for command in [
        [cmake, "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        [cmake, "--build", str(build), "--config", "Release", "--parallel", "2"],
    ]:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    path = build / "AutomationProbe.vst3"
    assert path.exists()
    return str(path)


def load(path, name="Automation Probe", mode="sample_steps"):
    plugin = VST3Plugin(path, plugin_name=name)
    plugin.automation_mode = mode
    return plugin


def signal(n=1025, channels=2):
    return np.random.default_rng(73).uniform(-0.4, 0.4, (channels, n)).astype(np.float32)


def curves(n):
    return np.linspace(-30, 6, n), np.geomspace(20, 20000, n)


def expected(audio, gain, frequency):
    # Float32 normalized controls match the existing scalar host precision.
    raw_gain = ((gain + 60) / 72).astype(np.float32).astype(np.float64)
    raw_freq = (np.log(frequency / 20) / np.log(1000)).astype(np.float32).astype(np.float64)
    g = 10 ** ((-60 + 72 * raw_gain) / 20)
    f = 20 * 1000**raw_freq
    return (audio.astype(np.float64) * g * f / 20000).astype(np.float32)


def render(plugin, audio, rate, gain, frequency, **kwargs):
    return plugin(audio, rate, automation={"gain_db": gain, "frequency_hz": frequency}, **kwargs)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rate", [22050, 48000, 96000])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
@pytest.mark.parametrize("layout", ["mono", "planar", "interleaved"])
def test_physical_db_and_nonlinear_hz_match_oracle(vst3_probe, mode, rate, buffer_size, layout):
    x = signal()
    gain, frequency = curves(x.shape[1])
    reference = expected(x, gain, frequency)
    if layout == "mono":
        x, reference = x[0], reference[0]
    elif layout == "interleaved":
        x, reference = x.T, reference.T
    plugin = load(vst3_probe, mode=mode)
    actual = render(plugin, x, rate, gain, frequency, buffer_size=buffer_size)
    np.testing.assert_allclose(actual, reference, atol=3e-7, rtol=3e-7)
    assert plugin._get_parameter("Gain").raw_value == np.float32((gain[-1] + 60) / 72)
    assert plugin._get_parameter("Frequency").raw_value == np.float32(1)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
def test_real_reported_latency_keeps_curve_on_source_audio(vst3_probe, mode, buffer_size):
    x = signal()
    gain, frequency = curves(x.shape[1])
    p = load(vst3_probe, "Automation Probe Delayed", mode)
    actual = render(p, x, 48000, gain, frequency, buffer_size=buffer_size)
    np.testing.assert_allclose(actual, expected(x, gain, frequency), atol=3e-7, rtol=3e-7)
    assert p.reported_latency_samples == 20


@pytest.mark.parametrize("buffer_size", [31, 8192])
def test_default_steps_support_last_point_plugins(vst3_probe, buffer_size):
    p = load(vst3_probe, "Automation Probe Last Point")
    x = signal()
    gain, frequency = curves(x.shape[1])
    np.testing.assert_allclose(
        render(p, x, 48000, gain, frequency, buffer_size=buffer_size),
        expected(x, gain, frequency),
        atol=3e-7,
        rtol=3e-7,
    )
    assert p.automation_mode == "sample_steps"


def test_events_expose_last_point_plugin_behavior_without_claiming_precision(vst3_probe):
    p = load(vst3_probe, "Automation Probe Last Point", "vst3_events")
    x = signal(257)
    gain, frequency = curves(257)
    actual = render(p, x, 48000, gain, frequency, buffer_size=31)
    held_gain, held_frequency = gain.copy(), frequency.copy()
    for start in range(0, 257, 31):
        end = min(start + 31, 257)
        held_gain[start:end], held_frequency[start:end] = gain[end - 1], frequency[end - 1]
    np.testing.assert_allclose(actual, expected(x, held_gain, held_frequency), atol=3e-7, rtol=3e-7)
    assert np.max(abs(actual - expected(x, gain, frequency))) > 0.01


@pytest.mark.parametrize("mode,blocks", [("sample_steps", 2), ("vst3_events", 1)])
def test_steps_coalesce_constant_runs_and_apply_simultaneous_changes(vst3_probe, mode, blocks):
    p = load(vst3_probe, mode=mode)
    x = signal(257)
    gain = np.r_[np.full(128, -12.0), np.full(129, 3.0)]
    frequency = np.r_[np.full(128, 200.0), np.full(129, 10000.0)]
    actual = render(p, x, 48000, gain, frequency)
    np.testing.assert_allclose(actual, expected(x, gain, frequency), atol=3e-7, rtol=3e-7)
    assert int(p._get_parameter("Blocks").string_value) == blocks


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ["Automation Probe", "Automation Probe Delayed"])
def test_direct_streams_keep_state_with_small_calls_and_buffer_growth(vst3_probe, mode, name):
    x = signal(1537)
    gain, frequency = curves(x.shape[1])
    whole = render(load(vst3_probe, name, mode), x, 48000, gain, frequency, buffer_size=31)
    p = load(vst3_probe, name, mode)
    p.reset()
    parts = []
    start = 0
    for end, size in [(8, 31), (9, 1), (11, 8192), (257, 31), (1537, 8192)]:
        parts.append(
            render(
                p,
                x[:, start:end],
                48000,
                gain[start:end],
                frequency[start:end],
                buffer_size=size,
                reset=False,
            )
        )
        start = end
    parts.append(p(np.zeros((2, 40), np.float32), 48000, reset=False))
    assert_array_equal(np.concatenate(parts, axis=1)[:, :1537], whole)


@pytest.mark.parametrize("mode", MODES)
def test_nested_graphs_share_vst_and_builtin_curves(vst3_probe, mode):
    x = signal(1025)
    gain, frequency = curves(x.shape[1])
    p, post = load(vst3_probe, "Automation Probe Delayed", mode), Gain()
    board = Pedalboard([Mix([Chain([AddLatency(57), p]), None]), post])
    target = np.linspace(-18, 0, x.shape[1], dtype=np.float32)
    actual = board(
        x,
        48000,
        buffer_size=31,
        automation={p: {"gain_db": gain, "frequency_hz": frequency}, post: {"gain_db": target}},
    )
    reference = expected(x, gain, frequency) + x
    reference = Gain()(reference, 48000, automation={"gain_db": target})
    np.testing.assert_allclose(actual, reference, atol=5e-7, rtol=5e-7)


@pytest.mark.parametrize("mode", MODES)
def test_graph_pending_controls_and_omitted_vst_lanes(vst3_probe, mode):
    x = signal(1537, 1)[0]
    gain, frequency = curves(len(x))
    gain[513:] = gain[512]
    whole = render(load(vst3_probe, mode=mode), x, 48000, gain, frequency)
    p = load(vst3_probe, mode=mode)
    board = Pedalboard([AddLatency(701), p])
    board.reset()
    a = board(
        x[:513],
        48000,
        reset=False,
        automation={p: {"gain_db": gain[:513], "frequency_hz": frequency[:513]}},
    )
    b = board(x[513:], 48000, reset=False, automation={p: {"frequency_hz": frequency[513:]}})
    c = board(np.zeros(1402, np.float32), 48000, reset=False)
    assert_array_equal(np.concatenate([a, b, c])[: len(x)], whole)


@pytest.mark.parametrize(
    "bad",
    [
        "unknown",
        "length",
        "ndim",
        "list",
        "complex",
        "nan",
        "range",
        "readonly",
        "broken",
        "misparsed",
    ],
)
def test_bad_physical_curves_preserve_parameters_and_delayed_audio(vst3_probe, bad):
    x = signal(1025)
    p, reference = [load(vst3_probe, "Automation Probe Delayed") for _ in range(2)]
    for plugin in (p, reference):
        plugin(x, 48000, reset=False)
    before = [(a.raw_value) for a in p._parameters]
    mapping = {"gain_db": np.full(1025, -12.0)}
    if bad == "unknown":
        mapping["missing"] = np.ones(1025)
    elif bad == "length":
        mapping["frequency_hz"] = np.ones(1024)
    elif bad == "ndim":
        mapping["frequency_hz"] = np.ones((1, 1025))
    elif bad == "list":
        mapping["frequency_hz"] = [1000] * 1025
    elif bad == "complex":
        mapping["frequency_hz"] = np.full(1025, 1000j)
    elif bad == "nan":
        mapping["frequency_hz"] = np.full(1025, np.nan)
    elif bad == "range":
        mapping["frequency_hz"] = np.full(1025, 20001)
    elif bad == "readonly":
        mapping["blocks"] = np.ones(1025)
    elif bad == "broken":
        mapping["broken"] = np.ones(1025)
    else:
        mapping["misparsed"] = np.full(1025, 0.5)
    with pytest.raises((ValueError, TypeError)):
        p(x, 48000, automation=mapping)
    assert before == [a.raw_value for a in p._parameters]
    assert_array_equal(p(x, 48000, reset=False), reference(x, 48000, reset=False))


def test_graph_late_error_does_not_activate_a_vst_or_mutate_its_values(vst3_probe):
    p, reference = [load(vst3_probe) for _ in range(2)]
    post = Gain()
    board = Pedalboard([p, post])
    x = signal()
    before = p._get_parameter("Gain").raw_value
    with pytest.raises(ValueError):
        board(
            x,
            48000,
            automation={
                p: {"gain_db": np.full(1025, -12)},
                post: {"gain_db": np.full(1025, np.nan)},
            },
        )
    assert p._get_parameter("Gain").raw_value == before
    assert_array_equal(p(x, 48000, reset=False), reference(x, 48000, reset=False))


@pytest.mark.parametrize("mode", MODES)
def test_midi_notes_and_physical_curves_share_sample_offsets(vst3_probe, mode):
    p = load(vst3_probe, "Automation Probe Instrument", mode)
    rate, duration = 48000, 0.0625
    n = int(duration * rate)
    gain, frequency = curves(n)
    midi = [([144, 60, 127], 0), ([128, 60, 0], duration / 2)]
    actual = p(
        midi,
        duration=duration,
        sample_rate=rate,
        buffer_size=31,
        automation={"gain_db": gain, "frequency_hz": frequency},
    )
    reference = expected(np.ones((2, n), np.float32), gain, frequency)
    reference[:, n // 2 :] = 0
    np.testing.assert_allclose(actual, reference, atol=3e-7, rtol=3e-7)
    assert int(p._get_parameter("Blocks").string_value) == (
        n if mode == "sample_steps" else (n + 30) // 31
    )


@pytest.mark.parametrize("mode", MODES)
def test_midi_streams_hold_notes_and_final_targets(vst3_probe, mode):
    rate, duration = 48000, 0.03125
    n = int(rate * duration)
    p = load(vst3_probe, "Automation Probe Instrument", mode)
    gain, frequency = curves(n)
    p.reset()
    a = p(
        [([144, 60, 127], 0)],
        duration,
        rate,
        buffer_size=31,
        reset=False,
        automation={"gain_db": gain, "frequency_hz": frequency},
    )
    b = p([], duration, rate, buffer_size=8192, reset=False)
    reference = expected(np.ones((2, n), np.float32), gain, frequency)
    np.testing.assert_allclose(a, reference, atol=3e-7, rtol=3e-7)
    assert_array_equal(b, np.repeat(reference[:, -1:], n, axis=1))
    with pytest.raises(ValueError):
        p([], duration, rate, automation={"frequency_hz": np.full(n, -1)})
    assert_array_equal(p([], duration, rate, reset=False), b)


def test_mode_setter_does_not_trigger_legacy_range_guessing(vst3_probe):
    p = VST3Plugin(vst3_probe, plugin_name="Automation Probe")
    before = vars(p).copy()
    p.automation_mode = "vst3_events"
    assert vars(p) == before
    assert p.automation_mode == "vst3_events"
    with pytest.raises(ValueError):
        p.automation_mode = "unknown"
    x = signal()
    p(x, 48000, automation={"gain_db": np.zeros(1025)})
    with pytest.raises(ValueError):
        p.automation_mode = "sample_steps"
    p.reset()
    p.automation_mode = "sample_steps"


def test_warm_scalar_to_curve_requires_explicit_reset(vst3_probe):
    p = load(vst3_probe)
    x = signal()
    p(x, 48000)
    with pytest.raises(ValueError, match="reset"):
        p(x, 48000, reset=False, automation={"gain_db": np.zeros(1025)})
    p.reset()
    p(x, 48000, reset=False, automation={"gain_db": np.zeros(1025)})


def test_boolean_and_literal_infinite_display_endpoints(vst3_probe):
    x = signal(257)
    p = load(vst3_probe)
    bypass = np.arange(257) % 2 == 0
    actual = p(x, 48000, automation={"bypass": bypass})
    reference = x.copy()
    reference[:, ~bypass] = expected(x, np.zeros(257), np.full(257, 20))[:, ~bypass]
    np.testing.assert_allclose(actual, reference, atol=3e-7, rtol=3e-7)
    with pytest.raises(ValueError):
        p(x, 48000, automation={"bypass": np.full(257, 0.5)})
    p = load(vst3_probe, "Automation Probe Infinite Gain")
    gain = np.r_[-np.inf, np.linspace(-60, 0, 256)]
    actual = p(x, 48000, automation={"gain_db": gain})
    raw = (10 ** (gain / 20) / 2).astype(np.float32).astype(float)
    np.testing.assert_allclose(
        actual, (x * (2 * raw) * 0.001).astype(np.float32), atol=3e-7, rtol=3e-7
    )
    assert np.all(actual[:, 0] == 0)


def test_empty_audio_and_owned_strided_readonly_curves(vst3_probe):
    p = load(vst3_probe)
    x = signal()
    gain, frequency = curves(1025)
    gain = np.repeat(gain, 2)[::2]
    frequency = np.repeat(frequency, 2)[::2]
    gain.flags.writeable = frequency.flags.writeable = x.flags.writeable = False
    actual = render(p, x, 48000, gain, frequency)
    np.testing.assert_allclose(actual, expected(x, gain, frequency), atol=3e-7, rtol=3e-7)
    before = p._get_parameter("Gain").raw_value
    p(np.zeros((2, 0), np.float32), 48000, automation={"gain_db": np.zeros(0)})
    assert p._get_parameter("Gain").raw_value == before


def test_public_containers_still_reject_instruments(vst3_probe):
    p = load(vst3_probe, "Automation Probe Instrument")
    with pytest.raises(ValueError, match="instrument"):
        Pedalboard([p])


def test_real_chow_tape_accepts_db_hz_and_boolean_curves():
    folder = Path(__file__).parent / "plugins" / "effect" / platform.system()
    path = folder / "CHOWTapeModel.vst3"
    if not path.exists():
        pytest.skip("Platform-specific CHOWTapeModel fixture is not installed")
    p = VST3Plugin(str(path))
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


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rate", [22050, 44100, 48000, 96000])
@pytest.mark.parametrize("automated", [False, True])
def test_sample_aligned_midi_does_not_round_to_preceding_frame(vst3_probe, mode, rate, automated):
    p = load(vst3_probe, "Automation Probe Instrument", mode)
    duration = 0.125
    n = int(duration * rate)
    kwargs = {"automation": {"gain_db": np.zeros(n)}} if automated else {}
    y = p(
        [([144, 60, 127], 7 / rate), ([128, 60, 0], 45 / rate)],
        duration,
        rate,
        buffer_size=31,
        **kwargs,
    )
    assert_array_equal(np.flatnonzero(y[0] != 0), np.arange(7, 45))


def test_fresh_automated_stream_starts_with_correct_latency_count(vst3_probe):
    p = load(vst3_probe, "Automation Probe Delayed")
    x = signal(129)
    y = p(x, 48000, reset=False, automation={"gain_db": np.zeros(129)})
    assert y.shape == (2, 109)
    np.testing.assert_allclose(
        y, expected(x[:, :109], np.zeros(109), np.full(109, 20)), atol=3e-7, rtol=3e-7
    )


def test_active_format_changes_require_a_root_restart(vst3_probe):
    p = load(vst3_probe)
    x = signal()
    p(x, 48000, automation={"gain_db": np.zeros(1025)})
    with pytest.raises(ValueError, match="format"):
        p(x, 44100, reset=False)
    p(x, 44100, automation={"gain_db": np.zeros(1025)})


def test_vst_reset_and_render_share_the_instance_lock(vst3_probe):
    from concurrent.futures import ThreadPoolExecutor

    p = load(vst3_probe)
    x = signal(257)
    target = np.linspace(-12, 0, 257)
    reference = expected(x, target, np.full(257, 20))

    def render_work():
        for _ in range(12):
            result = p(x, 48000, automation={"gain_db": target})
            np.testing.assert_allclose(result, reference, atol=3e-7, rtol=3e-7)

    def reset_work():
        for _ in range(12):
            p.reset()

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(render_work), pool.submit(reset_work)
        a.result(timeout=20)
        b.result(timeout=20)


@pytest.mark.parametrize("mode", MODES)
def test_stable_param_ids_survive_full_plugin_reinstantiation(vst3_probe, mode):
    from pedalboard import ExternalPluginReloadType

    p = load(vst3_probe, mode=mode)
    x = signal(257)
    gain, frequency = curves(257)
    render(p, x, 48000, gain, frequency)
    state = p.raw_state
    # Use the existing native descriptor directly, avoiding Python range guesses
    # for an unrelated host option in this fixture with deliberately bad knobs.
    type(p)._reload_type.__set__(p, ExternalPluginReloadType.PersistsAudioOnReset)
    for _ in range(2):
        p.reset()
        p.raw_state = state
        actual = render(p, x, 48000, gain[::-1].copy(), frequency[::-1].copy())
        np.testing.assert_allclose(
            actual, expected(x, gain[::-1], frequency[::-1]), atol=3e-7, rtol=3e-7
        )


@pytest.mark.parametrize("mode", MODES)
def test_graph_owns_pending_physical_curves_after_caller_mutation(vst3_probe, mode):
    x = signal(1025, 1)[0]

    def run(mutate):
        p = load(vst3_probe, mode=mode)
        board = Pedalboard([AddLatency(701), p])
        gain, frequency = curves(1025)
        board.reset()
        a = board(
            x[:513],
            48000,
            reset=False,
            automation={p: {"gain_db": gain[:513], "frequency_hz": frequency[:513]}},
        )
        if mutate:
            gain[:513], frequency[:513] = 12, 20000
        b = board(
            x[513:],
            48000,
            reset=False,
            automation={p: {"gain_db": gain[513:], "frequency_hz": frequency[513:]}},
        )
        c = board(np.zeros(1402, np.float32), 48000, reset=False)
        return np.concatenate([a, b, c])[: len(x)]

    assert_array_equal(run(True), run(False))


def test_empty_midi_curves_and_bad_timestamps_preserve_held_notes(vst3_probe):
    p = load(vst3_probe, "Automation Probe Instrument")
    n, duration = 1500, 0.03125
    p([([144, 60, 127], 0.0)], duration, 48000, automation={"gain_db": np.zeros(n)})
    empty = p([], 0.0, 48000, automation={"gain_db": np.zeros(0)})
    assert empty.shape == (2, 0)
    for time in [np.nan, np.inf]:
        with pytest.raises(ValueError, match="finite"):
            p([([128, 60, 0], time)], duration, 48000, automation={"gain_db": np.zeros(n)})
    held = p([], duration, 48000, reset=False)
    assert np.all(held != 0)
    # Very distant events cannot wrap an integer offset into the current render.
    p.reset()
    assert not p(
        [([144, 60, 127], 1e100)], duration, 48000, automation={"gain_db": np.zeros(n)}
    ).any()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rate", [22050, 48000, 96000])
def test_frequency_units_are_resolved_again_after_format_preparation(vst3_probe, mode, rate):
    p = load(vst3_probe, "Automation Probe Rate Dependent", mode)
    n = 129
    frequency = np.geomspace(100, 10000, n)
    y = p(
        np.ones(n, np.float32),
        rate,
        automation={"gain_db": np.zeros(n), "frequency_hz": frequency},
    )
    normalized = (np.log(frequency / 20) / np.log(rate / 40)).astype(np.float32)
    actual_hz = 20 * (rate / 40) ** normalized.astype(np.float64)
    np.testing.assert_allclose(y, actual_hz / 20000, atol=3e-7, rtol=3e-7)
    assert abs(float(p._get_parameter("Frequency").string_value) - 10000) < 0.01


@pytest.mark.parametrize("mode", MODES)
def test_explicit_preparation_exposes_new_rate_dependent_range(vst3_probe, mode):
    p = load(vst3_probe, "Automation Probe Rate Dependent", mode)
    p.prepare_automation(96000, num_channels=1)
    n = 129
    targets = {"gain_db": np.zeros(n), "frequency_hz": np.full(n, 30000.0)}
    a = p(np.ones(n, np.float32), 96000, reset=False, automation=targets)
    b = p(np.ones(n, np.float32), 96000, reset=False, automation=targets)
    np.testing.assert_allclose(a, 1.5, atol=2e-6)
    assert_array_equal(a, b)
    for rate, channels in [(0, 1), (np.nan, 1), (48000, 0), (48000, 3)]:
        with pytest.raises(ValueError):
            p.prepare_automation(rate, num_channels=channels)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
def test_reused_midi_buffers_do_not_repeat_notes_or_feed_back_plugin_output(
    vst3_probe, mode, buffer_size
):
    p = load(vst3_probe, "Automation Probe Instrument", mode)
    rate, duration, n = 48000, 0.03125, 1500
    intervals = [(7, 12), (31, 93), (127, 128), (511, 600), (1025, 1499)]
    notes = []
    mask = np.zeros(n, np.float32)
    for start, end in intervals:
        notes.extend([([144, 60, 127], start / rate), ([128, 60, 0], end / rate)])
        mask[start:end] = 1
    gain, frequency = curves(n)
    result = p(
        notes,
        duration,
        rate,
        buffer_size=buffer_size,
        automation={"gain_db": gain, "frequency_hz": frequency},
    )
    reference = expected(np.tile(mask, (2, 1)), gain, frequency)
    np.testing.assert_allclose(result, reference, atol=3e-7, rtol=3e-7)
    assert not p([], duration, rate, reset=False, buffer_size=buffer_size).any()
