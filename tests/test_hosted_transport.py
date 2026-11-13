"""Native DSP observations of per-call playhead timing, not Python attributes."""

import platform

import numpy as np
import pytest

from pedalboard import VST3Plugin

from .test_au_automation import au_probe as _au_probe  # noqa: F401
from .test_vst3_automation import vst3_probe as _vst3_probe  # noqa: F401


@pytest.fixture(params=["vst3", "au"])
def clock_plugin(request):
    if request.param == "au":
        if platform.system() != "Darwin":
            pytest.skip("Audio Units require macOS")
        from pedalboard import AudioUnitPlugin

        bundle, cls, prefix = (
            request.getfixturevalue("_au_probe"),
            AudioUnitPlugin,
            "Automation AU ",
        )
    else:
        bundle, cls, prefix = (
            request.getfixturevalue("_vst3_probe"),
            VST3Plugin,
            "Automation Probe ",
        )

    def create(label="Clock"):
        return cls(bundle, plugin_name=prefix + label)

    create.format = request.param
    return create


def render(p, clock, rate=48000, n=259, buffer=31, **kwargs):
    return p.process_buses(
        {0: np.zeros((2, n), np.float32)},
        rate,
        output_buses={0: 2, 1: 2},
        transport=clock,
        buffer_size=buffer,
        **kwargs,
    )


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("buffer", [1, 31, 8192, 65536])
@pytest.mark.parametrize("playing", [False, True])
@pytest.mark.parametrize("ppq", [-2.0, 2.0])
def test_actual_plugin_tempo_ppq_flags_and_default_restoration(
    clock_plugin, rate, buffer, playing, ppq
):
    p = clock_plugin()
    clock = {
        "bpm": 90.0,
        "ppq_position": ppq,
        "sample_position": -1234,
        "playing": playing,
        "recording": True,
        "time_signature": (7, 8),
    }
    output = render(p, clock, rate=rate, buffer=buffer)
    beat = ppq + (np.arange(259) if playing else np.zeros(259)) * 90 / (60 * rate)
    np.testing.assert_array_equal(output[0][0], np.full(259, np.float32(0.09)))
    np.testing.assert_allclose(output[0][1], beat / 1000, atol=1e-9, rtol=2e-7)
    expected_flags = (1 if playing else 0) + 2 + 8 + 16 + 32 + 128
    np.testing.assert_array_equal(output[1][0], np.full(259, np.float32(expected_flags / 256)))
    np.testing.assert_allclose(output[1][1], np.sin(beat * 2 * np.pi), atol=3e-7, rtol=3e-7)
    default = render(p, None, rate=rate, buffer=buffer, reset=False)
    np.testing.assert_array_equal(default[0][0], np.full(259, np.float32(0.12)))
    np.testing.assert_array_equal(default[0][1], np.zeros(259))


def decode_position(output):
    words = np.concatenate([output[0], output[1]], axis=0)
    integer = np.rint(words.astype(np.float64) * 65536).astype(np.uint64)
    combined = sum(integer[i] << np.uint64(16 * i) for i in range(4))
    return combined.view(np.int64)


@pytest.mark.parametrize("position", [-(2**40), -7, 0, 2**40])
@pytest.mark.parametrize("buffer", [1, 31, 8192])
def test_signed_64_bit_sample_clock_reaches_actual_dsp(clock_plugin, position, buffer):
    p = clock_plugin("Position")
    output = render(p, {"sample_position": position, "ppq_position": 0.0}, buffer=buffer)
    np.testing.assert_array_equal(decode_position(output), position + np.arange(259))


@pytest.mark.parametrize("buffer", [1, 31, 8192, 65536])
def test_loop_crosses_inside_native_block(clock_plugin, buffer):
    rate = 48000
    # At120 BPM this region has exactly240 frames, starting5 frames before end.
    clock = {
        "bpm": 120.0,
        "ppq_position": 235 / 24000,
        "sample_position": 235,
        "playing": True,
        "looping": True,
        "loop_start_ppq": 0.0,
        "loop_end_ppq": 0.01,
    }
    p = clock_plugin()
    output = render(p, clock, rate=rate, n=1001, buffer=buffer)
    beat = ((235 + np.arange(1001)) % 240) / 24000
    np.testing.assert_allclose(output[0][1], beat / 1000, atol=1e-10, rtol=2e-7)
    position = render(clock_plugin("Position"), clock, rate=rate, n=1001, buffer=buffer)
    np.testing.assert_array_equal(decode_position(position), (235 + np.arange(1001)) % 240)


@pytest.mark.parametrize(
    "invalid",
    [
        {"bpm": 0},
        {"bpm": 1e-20},
        {"bpm": float("inf")},
        {"bpm": 120, "ppq_position": 1e30},
        {"ppq_position": float("inf")},
        {"sample_position": True},
        {"sample_position": -(2**54)},
        {"recording": 1},
        {"looping": 1},
        {"time_signature": (4,)},
        {"time_signature": (True, 4)},
        {"time_signature": (4, 2**32)},
        {"loop_start_ppq": float("nan"), "loop_end_ppq": 1},
        {"bpm": float("nan")},
        {"playing": 1},
        {"bpm": True},
        {"time_signature": (4, 3)},
        {"time_signature": (0, 4)},
        {"looping": True},
        {"loop_start_ppq": 0},
        {"loop_start_ppq": 4, "loop_end_ppq": 2},
        {"looping": True, "loop_start_ppq": 0, "loop_end_ppq": 4, "ppq_position": 4},
        {"sample_position": 0.5},
        {"sample_position": 2**54},
        {"beats_per_minute": 90},
    ],
)
def test_bad_transport_does_not_reset_or_consume_midi(clock_plugin, invalid):
    p = clock_plugin()
    render(p, {"bpm": 90})
    before = p._get_parameter("Blocks").string_value
    consumed = []

    def midi():
        consumed.append(True)
        yield (bytes([0x90, 60, 100]), 0.0)

    with pytest.raises((ValueError, TypeError, OverflowError)):
        render(p, invalid, midi_messages=midi())
    assert consumed == []
    assert p._get_parameter("Blocks").string_value == before


@pytest.mark.parametrize("buffer", [1, 31, 8192])
def test_fractional_loop_keeps_negative_half_frame_translation(clock_plugin, buffer):
    clock = {
        "bpm": 120.0,
        "sample_position": 2,
        "ppq_position": 3.5 / 24000,
        "looping": True,
        "loop_start_ppq": 0.0,
        "loop_end_ppq": 4.5 / 24000,
    }
    output = render(clock_plugin("Position"), clock, n=103, buffer=buffer)
    phase = np.remainder(3.5 + np.arange(103), 4.5)
    exact = 2 + phase - 3.5
    expected = np.floor(exact + 0.5).astype(np.int64)
    expected[0] = 2
    np.testing.assert_array_equal(decode_position(output), expected)


@pytest.mark.parametrize("buffer", [1, 31, 8192])
def test_meter_bar_and_next_beat_are_actual_callback_values(clock_plugin, buffer):
    rate = 48000
    clock = {
        "bpm": 90.0,
        "ppq_position": -2.25,
        "sample_position": -99,
        "time_signature": (7, 8),
        "playing": False,
    }
    output = render(clock_plugin("Meter"), clock, rate=rate, buffer=buffer)
    np.testing.assert_array_equal(output[0][0], np.full(259, np.float32(7 / 32)))
    np.testing.assert_array_equal(output[0][1], np.full(259, np.float32(8 / 32)))
    np.testing.assert_allclose(output[1][0], -3.5 / 1000, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(output[1][1] * 65536, np.full(259, 8000))


def test_loop_bounds_are_visible_even_without_playback(clock_plugin):
    output = render(
        clock_plugin("Loop"),
        {
            "playing": False,
            "looping": True,
            "loop_start_ppq": -4.0,
            "loop_end_ppq": -1.0,
            "ppq_position": -2.0,
        },
    )
    np.testing.assert_array_equal(output[0][0], np.full(259, np.float32(-0.004)))
    np.testing.assert_array_equal(output[0][1], np.full(259, np.float32(-0.001)))


@pytest.mark.parametrize("buffer", [1, 31, 8192])
def test_many_fractional_wraps_and_explicit_split_anchors_are_exact(clock_plugin, buffer):
    check_many_wraps(clock_plugin("Position"), buffer)


def check_many_wraps(p, buffer):
    # Exact rational oracle: period882/41 frames. Its odd denominator avoids
    # ambiguous half-frame rounding while exercising190 loop crossings.
    rate, n, bpm = 44100, 4097, 123.0
    step = bpm / (60 * rate)
    positions = np.arange(n + 1, dtype=np.float64)
    unwrapped = 0.00017 + positions * step
    wraps = np.floor(unwrapped / 0.001)
    phase = unwrapped - wraps * 0.001
    exact = 100 + positions - wraps * (882 / 41)
    frames = np.floor(exact + 0.5).astype(np.int64)
    initial = {
        "bpm": bpm,
        "ppq_position": 0.00017,
        "sample_position": 100,
        "looping": True,
        "loop_start_ppq": 0.0,
        "loop_end_ppq": 0.001,
    }
    whole = render(p, initial, rate=rate, n=n, buffer=buffer)
    np.testing.assert_array_equal(decode_position(whole), frames[:n])
    pieces = []
    for start, end in [(0, 17), (17, 131), (131, n)]:
        clock = {
            **initial,
            "sample_position": int(frames[start]),
            "ppq_position": float(phase[start]),
        }
        pieces.append(render(p, clock, rate=rate, n=end - start, buffer=buffer, reset=start == 0))
    combined = {index: np.concatenate([part[index] for part in pieces], axis=1) for index in [0, 1]}
    np.testing.assert_array_equal(decode_position(combined), frames[:n])


def test_au_transport_changed_reports_seek_but_not_continuous_blocks(clock_plugin):
    if clock_plugin.format != "au":
        pytest.skip("VST3 has no transport-state-changed flag")
    p = clock_plugin("Loop")
    first = render(p, {"bpm": 120.0}, n=31, buffer=31)
    np.testing.assert_array_equal(first[1][0], np.ones(31))
    continued = render(p, {"bpm": 120.0, "sample_position": 31}, n=31, buffer=31, reset=False)
    np.testing.assert_array_equal(continued[1][0], np.zeros(31))
    seek = render(p, {"bpm": 120.0, "sample_position": 300}, n=31, buffer=31, reset=False)
    np.testing.assert_array_equal(seek[1][0], np.ones(31))


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("buffer", [1, 31, 8192])
@pytest.mark.parametrize("label", ["Clock", "Meter", "Loop", "Position"])
@pytest.mark.parametrize("scenario", ["playing", "stopped", "loop"])
def test_independent_sdk_host_clock_matches_actual_dsp(
    request, tmp_path, rate, buffer, label, scenario
):
    import subprocess
    from pathlib import Path

    bundle = Path(request.getfixturevalue("_vst3_probe"))
    if platform.system() == "Darwin":
        binary, host = bundle / "Contents/MacOS/AutomationProbe", bundle.parent / "NativeBusHost"
    elif platform.system() == "Windows":
        binary, host = bundle, bundle.parent / "Release/NativeBusHost.exe"
    else:
        binary = next(bundle.glob("Contents/*/AutomationProbe.so"))
        host = bundle.parent / "NativeBusHost"
    playing, looping = scenario != "stopped", scenario == "loop"
    ppq, sample = (235 * 120 / (60 * rate), 235) if looping else (-2.25, -99)
    end = 240 * 120 / (60 * rate)
    output = tmp_path / "clock.f32"
    command = [str(host), str(binary), str(output)] + list(
        map(
            str,
            [
                rate,
                buffer,
                2,
                0,
                2,
                1001,
                label,
                120,
                ppq,
                sample,
                int(playing),
                1,
                int(looping),
                0,
                end,
                7,
                8,
            ],
        )
    )
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    native = np.fromfile(output, np.float32).reshape(4, 1001)
    # Independent SDK host is itself checked against simple known clock values.
    if label == "Clock":
        beat = ppq + (np.arange(1001) if playing else np.zeros(1001)) * 120 / (60 * rate)
        if looping:
            beat = ((235 + np.arange(1001)) % 240) * 120 / (60 * rate)
        np.testing.assert_array_equal(native[0], np.full(1001, np.float32(0.12)))
        np.testing.assert_allclose(native[1], beat / 1000, atol=1e-9, rtol=2e-7)
        np.testing.assert_allclose(native[3], np.sin(beat * 2 * np.pi), atol=3e-7, rtol=3e-7)
    elif label == "Position":
        wanted = (
            (235 + np.arange(1001)) % 240
            if looping
            else sample + (np.arange(1001) if playing else np.zeros(1001, dtype=int))
        )
        np.testing.assert_array_equal(decode_position({0: native[:2], 1: native[2:]}), wanted)
    elif label == "Meter":
        np.testing.assert_array_equal(native[0], np.full(1001, np.float32(7 / 32)))
        np.testing.assert_array_equal(native[1], np.full(1001, np.float32(8 / 32)))
    else:
        np.testing.assert_array_equal(native[0], np.zeros(1001))
        np.testing.assert_array_equal(native[1], np.full(1001, np.float32(end / 1000)))
    clock = {
        "bpm": 120,
        "ppq_position": ppq,
        "sample_position": sample,
        "playing": playing,
        "recording": True,
        "looping": looping,
        "loop_start_ppq": 0,
        "loop_end_ppq": end,
        "time_signature": (7, 8),
    }
    actual = render(
        VST3Plugin(str(bundle), plugin_name="Automation Probe " + label),
        clock,
        rate=rate,
        n=1001,
        buffer=buffer,
    )
    if label == "Clock":
        np.testing.assert_allclose(
            np.concatenate(list(actual.values())), native, atol=3e-7, rtol=3e-7
        )
    else:
        np.testing.assert_array_equal(np.concatenate(list(actual.values())), native)


def check_joint_clock_controls_and_midi(p, *, native_events=False):
    n, rate = 259, 48000
    if native_events:
        p.automation_mode = "vst3_events"
    gain = np.linspace(-24, 0, n)
    frequency = np.geomspace(200, 20000, n)
    midi = [(bytes([0x90, 60, 100]), 7 / rate), (bytes([0x80, 60, 0]), 128 / rate)]
    clock = {"bpm": 90, "sample_position": -99, "ppq_position": -2.0, "recording": True}
    actual, returned = render(
        p,
        clock,
        n=n,
        buffer=31,
        return_midi=True,
        midi_messages=midi,
        automation={"gain_db": gain, "frequency_hz": frequency},
    )
    beat = -2 + np.arange(n) * 90 / (60 * rate)
    np.testing.assert_allclose(actual[0][1], beat / 1000, atol=1e-9, rtol=2e-7)
    np.testing.assert_allclose(actual[1][1], np.sin(beat * 2 * np.pi), atol=3e-7, rtol=3e-7)
    assert abs(float(p._get_parameter("Gain").raw_value) - 60 / 72) < 1e-6
    assert float(p._get_parameter("Frequency").raw_value) == 1.0
    if p.midi_capabilities["produces_output"]:
        notes = [
            (round(t * rate), b[0], b[1])
            for b, t in returned
            if b[0] in [0x80, 0x90] and b[1] == 72
        ]
        assert notes == [(7, 0x90, 72), (128, 0x80, 72)]
    else:
        assert returned == []


@pytest.mark.parametrize("native_events", [False, True])
def test_clock_curves_and_notes_share_the_same_spans(clock_plugin, native_events):
    if native_events and clock_plugin.format == "au":
        pytest.skip("AU has no VST3 event delivery mode")
    check_joint_clock_controls_and_midi(clock_plugin(), native_events=native_events)


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("buffer", [31, 8192])
@pytest.mark.parametrize("bpm", [90, 120])
def test_unmodified_steinberg_sync_delay_uses_real_tempo(rate, buffer, bpm):
    import os

    path = os.environ.get("PEDALBOARD_TEST_STEINBERG_SYNC_DELAY")
    if not path:
        pytest.skip("Set PEDALBOARD_TEST_STEINBERG_SYNC_DELAY to the unchanged SDK Sync Delay")
    # The unchanged source's initial setting is four quarter notes, wet-only.
    delay = int((60.0 / bpm) * 4 * rate)
    x = np.zeros((2, delay + 37), np.float32)
    x[:, 0] = [1.0, -0.5]
    p = VST3Plugin(path)
    actual = p.process_buses(
        {0: x},
        rate,
        output_buses={0: 2},
        buffer_size=buffer,
        transport={"bpm": bpm, "sample_position": 0},
    )[0]
    expected = np.zeros_like(x)
    expected[:, delay] = x[:, 0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("origin", [-(2**40), 0, 2**40])
@pytest.mark.parametrize("buffer", [1, 31, 8192])
def test_fractional_rounding_is_independent_of_large_integer_origin(clock_plugin, origin, buffer):
    check_large_origin(clock_plugin("Position"), origin, buffer)


def check_large_origin(p, origin, buffer):
    # Rational period2501/1000 frames: source3 wraps to .499, not .5.
    n = 103
    indices = np.arange(n + 1, dtype=np.int64)
    # Exact integer oracle, including the nearest-frame rule and loop count.
    remainder = (indices * 1000) % 2501
    rounded = (remainder + 500) // 1000
    clock = {
        "sample_position": origin,
        "ppq_position": 0.0,
        "bpm": 120.0,
        "looping": True,
        "loop_start_ppq": 0.0,
        "loop_end_ppq": 2.501 / 24000,
    }
    actual = render(p, clock, n=n, buffer=buffer)
    np.testing.assert_array_equal(decode_position(actual), origin + rounded[:n])
    pieces = []
    for start, end in [(0, 3), (3, 27), (27, n)]:
        anchor = {
            **clock,
            "sample_position": int(origin + rounded[start]),
            "ppq_position": float(remainder[start] / 1000 / 24000),
        }
        pieces.append(render(p, anchor, n=end - start, buffer=buffer, reset=start == 0))
    together = {i: np.concatenate([v[i] for v in pieces], axis=1) for i in [0, 1]}
    np.testing.assert_array_equal(decode_position(together), origin + rounded[:n])
