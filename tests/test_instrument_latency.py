"""Real VST3/AUv2 offline instrument alignment, drain and raw streaming contracts."""

import os
import sys

import numpy as np
import pytest

from pedalboard import VST3Plugin

from .test_au_automation import au_probe  # noqa: F401
from .test_vst3_automation import vst3_probe  # noqa: F401


@pytest.fixture(params=["vst3", "au"])
def instrument(request):
    format = request.param
    if format == "au" and sys.platform != "darwin":
        pytest.skip("Audio Units require macOS")
    path = request.getfixturevalue("vst3_probe" if format == "vst3" else "au_probe")

    def load(latency=20, bypass=True):
        names = {
            0: "Instrument",
            20: "Delayed Instrument",
            4097: "Long Delayed Instrument",
            16385: "Huge Delayed Instrument",
        }
        prefix = "Automation Probe " if format == "vst3" else "Automation AU "
        if format == "au":
            from pedalboard import AudioUnitPlugin

            p = AudioUnitPlugin(path, plugin_name=prefix + names[latency])
        else:
            p = VST3Plugin(path, plugin_name=prefix + names[latency])
        p._get_parameter("Bypass").raw_value = float(bypass)
        return p

    return load


def notes(rate, *events):
    return [
        (bytes([0x90 if on else 0x80, 60, 100 if on else 0]), frame / rate) for frame, on in events
    ]


def length(n, rate):
    duration = float(np.float32(n / rate))
    return duration, int(np.float32(duration) * np.float32(rate))


@pytest.mark.parametrize("latency", [20, 4097])
@pytest.mark.parametrize("channels", [1, 2])
def test_raw_delayed_onset_is_not_implicitly_shifted(instrument, latency, channels):
    p = instrument(latency)
    raw = p(
        notes(48000, (7, True), (45, False)), duration=1, sample_rate=48000, num_channels=channels
    )
    expected = np.zeros_like(raw)
    expected[:, 7 + latency : 45 + latency] = 1
    np.testing.assert_array_equal(raw, expected)
    assert p.reported_latency_samples == latency


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
@pytest.mark.parametrize("latency", [0, 20, 4097])
def test_compensated_onset_and_final_frame(instrument, rate, channels, buffer_size, latency):
    duration, n = length(257, rate)
    p = instrument(latency)
    # The last note must survive even when latency exceeds duration. Late events
    # deliberately fall into the drain window and must not be delivered.
    midi = notes(rate, (0, True), (3, False), (7, True), (45, False), (n - 2, True), (n + 1, False))
    actual = p(
        midi,
        duration=duration,
        sample_rate=rate,
        num_channels=channels,
        buffer_size=buffer_size,
        compensate_latency=True,
    )
    expected = np.zeros((channels, n), np.float32)
    expected[:, :3] = 1
    expected[:, 7:45] = 1
    expected[:, -2:] = 1
    np.testing.assert_array_equal(actual, expected)
    assert p.reported_latency_samples == latency


@pytest.mark.parametrize("mode", ["sample_steps", "vst3_events"])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
def test_compensated_curves_remain_on_source_frames(instrument, mode, buffer_size):
    p, reference = instrument(20, False), instrument(0, False)
    try:
        p.automation_mode = reference.automation_mode = mode
    except ValueError:
        pytest.skip("AU exposes only sample_steps")
    rate = 48000
    duration, n = length(257, rate)
    curves = {"gain_db": np.linspace(-30, 6, n), "frequency_hz": np.geomspace(20, 20000, n)}
    midi = notes(rate, (0, True), (31, False), (64, True))
    kwargs = {
        "duration": duration,
        "sample_rate": rate,
        "buffer_size": buffer_size,
        "automation": curves,
    }
    expected = reference(midi, **kwargs)
    actual = p.process(midi, **kwargs, compensate_latency=True)
    np.testing.assert_array_equal(actual, expected)
    # Draining must leave the final source targets, not wrap to the curve start.
    assert (
        abs(p._get_parameter("Gain").raw_value - reference._get_parameter("Gain").raw_value) < 1e-6
    )
    assert (
        abs(
            p._get_parameter("Frequency").raw_value
            - reference._get_parameter("Frequency").raw_value
        )
        < 1e-6
    )


def test_compensation_rejects_stream_before_consuming_midi(instrument):
    p = instrument(20)
    used = []

    def midi():
        used.append(True)
        yield (bytes([0x90, 60, 100]), 0)

    before = p._configuration_snapshot()["state"]
    with pytest.raises(ValueError, match="reset=True"):
        p(midi(), duration=1, sample_rate=48000, reset=False, compensate_latency=True)
    assert not used
    assert p._configuration_snapshot()["state"] == before


def test_explicit_false_preserves_raw_stream(instrument):
    p, reference = instrument(20), instrument(20)
    first = notes(48000, (7, True))
    for midi in [first, [], notes(48000, (11, False))]:
        kwargs = {"duration": 0.01, "sample_rate": 48000, "reset": False, "buffer_size": 31}
        np.testing.assert_array_equal(
            p(midi, **kwargs, compensate_latency=False), reference(midi, **kwargs)
        )


def test_compensated_reset_is_repeatable_and_zero_duration_safe(instrument):
    p = instrument(4097)
    kwargs = {"duration": 0.01, "sample_rate": 48000, "compensate_latency": True}
    midi = notes(48000, (1, True), (479, False))
    np.testing.assert_array_equal(p(midi, **kwargs), p(midi, **kwargs))
    empty = p([], duration=0, sample_rate=48000, compensate_latency=True)
    assert empty.shape == (2, 0)


@pytest.mark.parametrize("latency", [20, 4097])
def test_drain_does_not_feed_post_duration_notes(instrument, latency):
    p = instrument(latency)
    duration, n = length(257, 48000)
    # An incorrectly delivered late note has no time to reach the compensated
    # interval, but it would remain audible in the following raw state probe.
    a = p(
        notes(48000, (n + 1, True)), duration=duration, sample_rate=48000, compensate_latency=True
    )
    b = p([], duration=0.01, sample_rate=48000, reset=False)
    assert not np.any(a)
    assert not np.any(b)


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("stages", ["", "csf"])
def test_actual_bass_plugin_alignment_and_release(monkeypatch, rate, stages):
    path = os.environ.get("PEDALBOARD_TEST_BASS_PLUGIN")
    if not path:
        pytest.skip("Set PEDALBOARD_TEST_BASS_PLUGIN for the original bass integration")
    monkeypatch.setenv("TACTUS_BASS_SRC", "noise")
    monkeypatch.setenv("TACTUS_BASS_STAGES", stages)
    p, reference = VST3Plugin(path), VST3Plugin(path)
    midi = notes(rate, (1000, True), (rate - 2, False))
    raw = reference(midi, duration=1.1, sample_rate=rate, buffer_size=31)
    actual = p(midi, duration=1, sample_rate=rate, buffer_size=31, compensate_latency=True)
    assert p.reported_latency_samples == 20
    assert actual.shape == (2, rate)
    assert np.max(np.abs(actual)) > 0.001
    np.testing.assert_array_equal(actual, raw[:, 20 : 20 + rate])
    if not stages:
        assert np.flatnonzero(raw[0])[0] == 1020
        assert np.flatnonzero(actual[0])[0] == 1000


@pytest.mark.parametrize("mode", ["sample_steps", "vst3_events"])
def test_long_drain_respects_prepared_automation_block_capacity(instrument, mode):
    p, reference = instrument(16385, False), instrument(0, False)
    try:
        p.automation_mode = reference.automation_mode = mode
    except ValueError:
        pytest.skip("AU exposes only sample_steps")
    duration, n = length(257, 48000)
    controls = {"gain_db": np.linspace(-30, 6, n), "frequency_hz": np.full(n, 20000.0)}
    kwargs = {
        "duration": duration,
        "sample_rate": 48000,
        "buffer_size": 65536,
        "automation": controls,
    }
    midi = notes(48000, (0, True))
    np.testing.assert_array_equal(
        p(midi, **kwargs, compensate_latency=True), reference(midi, **kwargs)
    )
