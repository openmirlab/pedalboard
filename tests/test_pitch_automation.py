"""Pitch targets follow source audio into the windowed Rubber Band engine."""

from itertools import pairwise

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import PitchShift


def signal(n, rate, channels):
    t = np.arange(n) / rate
    return np.stack(
        [
            (0.2 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.cos(2 * np.pi * 1713 * t)) * 0.7**c
            for c in range(channels)
        ]
    ).astype(np.float32)


def prepared(rate, channels, pitch=0):
    effect = PitchShift(pitch)
    effect(np.zeros((channels, 8192), np.float32), rate, buffer_size=128, reset=False)
    effect.reset()
    return effect


@pytest.mark.parametrize("rate", [22050, 48000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("kind", ["constant", "step", "dense"])
@pytest.mark.parametrize("buffer_size", [31, 8192])
def test_original_engine_with_delayed_scalar_targets(rate, channels, kind, buffer_size):
    n = 2049
    x = signal(n, rate, channels)
    if kind == "constant":
        curve = np.full(n, 7, np.float32)
    elif kind == "step":
        curve = np.where(np.arange(n) < n // 2, -3, 7).astype(np.float32)
    else:
        curve = np.linspace(-3, 7, n, dtype=np.float32)
    a, b = [prepared(rate, channels, float(curve[0])) for _ in range(2)]
    actual = a(x, rate, buffer_size=buffer_size, automation={"semitones": curve})
    # One second of priming plus sufficient silence to drain the engine. Changes
    # are applied immediately before their source frames enter Rubber Band.
    feed = np.concatenate([x, np.zeros((channels, 2 * rate), np.float32)], axis=1)
    index = np.clip(np.arange(feed.shape[1]) - rate, 0, n - 1)
    targets = curve[index]
    edges = np.r_[0, np.flatnonzero(targets[1:] != targets[:-1]) + 1, len(targets)]
    parts = []
    for start, end in pairwise(edges):
        b.semitones = float(targets[start])
        parts.append(
            b(
                feed[:, start:end],
                rate,
                buffer_size=1 if kind != "constant" else buffer_size,
                reset=False,
            )
        )
    expected = np.concatenate(parts, axis=1)[:, :n]
    assert_array_equal(actual, expected)
    assert a.semitones == float(curve[-1])


@pytest.mark.parametrize("buffer_size", [31, 513, 8192])
@pytest.mark.parametrize("pitch", [-12, 0, 7, 12])
def test_constant_curves_retain_scalar_output(buffer_size, pitch):
    rate = 22050
    x = signal(6001, rate, 2)
    assert_array_equal(
        PitchShift(pitch)(x, rate, buffer_size=buffer_size),
        PitchShift()(
            x, rate, buffer_size=buffer_size, automation={"semitones": np.full(6001, pitch)}
        ),
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -73, 73])
def test_invalid_curve_keeps_warm_state(value):
    rate = 22050
    x = signal(4097, rate, 2)
    a, b = prepared(rate, 2), prepared(rate, 2)
    for effect in (a, b):
        effect(x, rate, reset=False)
    with pytest.raises((ValueError, TypeError)):
        a(x, rate, automation={"semitones": np.full(4097, value)})
    assert a.semitones == b.semitones
    assert_array_equal(a(x, rate, reset=False), b(x, rate, reset=False))


def test_frequency_step_follows_source_not_priming_silence():
    rate = 22050
    n = 4 * rate
    t = np.arange(n) / rate
    x = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    curve = np.where(t < 1, 0, 12).astype(np.float32)
    y = PitchShift()(x, rate, automation={"semitones": curve})
    # Compare stable pitch to the retained scalar engine, whose resampling
    # ratio is not assumed to equal ideal equal-tempered tuning exactly.
    reference = PitchShift(12)(x, rate)
    for seconds, pitch in [(0.4, 0), (1.5, 12), (2.5, 12)]:
        start = int(seconds * rate)
        part = y[start : start + 4096]
        f = np.fft.rfftfreq(len(part), 1 / rate)
        peak = f[np.argmax(np.abs(np.fft.rfft(part * np.hanning(len(part)))))]
        steady = x[start : start + 4096] if pitch == 0 else reference[start : start + 4096]
        expected = f[np.argmax(np.abs(np.fft.rfft(steady * np.hanning(len(steady)))))]
        assert abs(peak - expected) <= rate / len(part)


@pytest.mark.parametrize("warm", [False, True])
def test_chunked_curves_preserve_pending_targets_through_scalar_tail(warm):
    rate = 22050
    n = 4097
    x = signal(n, rate, 2)
    curve = np.linspace(-3, 7, n, dtype=np.float32)
    a, b = [prepared(rate, 2, -3) for _ in range(2)]
    if warm:
        prefix = signal(rate // 3, rate, 2)
        for effect in (a, b):
            effect(prefix, rate, buffer_size=128, reset=False)
    whole = b(x, rate, buffer_size=128, reset=False, automation={"semitones": curve})
    parts = []
    for start, end in [(0, 7), (7, 300), (300, 2049), (2049, n)]:
        parts.append(
            a(
                x[:, start:end],
                rate,
                buffer_size=8192,
                reset=False,
                automation={"semitones": curve[start:end]},
            )
        )
    tail = np.zeros((2, 2 * rate), np.float32)
    actual = np.concatenate([*parts, a(tail, rate, reset=False)], axis=1)
    expected = np.concatenate([whole, b(tail, rate, reset=False)], axis=1)
    assert_array_equal(actual, expected)


def test_warm_scalar_to_different_constant_curve_does_not_apply_future_target_early():
    rate = 22050
    x = signal(2049, rate, 2)
    a, b = [prepared(rate, 2, 0.123456789) for _ in range(2)]
    for effect in (a, b):
        effect(signal(rate // 3, rate, 2), rate, buffer_size=128, reset=False)
    actual = a(x, rate, reset=False, automation={"semitones": np.full(2049, 7)})
    # All newly entered source frames are still waiting in the priming line.
    # The queued older scalar pitch must remain active at the engine input.
    expected = b(x, rate, buffer_size=1, reset=False)
    tail = np.zeros((2, int(0.7 * rate)), np.float32)
    actual = np.concatenate([actual, a(tail, rate, reset=False)], axis=1)
    expected = np.concatenate([expected, b(tail, rate, buffer_size=1, reset=False)], axis=1)
    assert actual.shape[1] > 1000
    assert_array_equal(actual, expected)
    assert a.semitones == 7


@pytest.mark.parametrize("first,last", [(-3, 7), (7, -3), (0, 12)])
def test_reset_true_repeated_render_matches_fresh_instance(first, last):
    rate = 22050
    x = signal(4097, rate, 2)
    curve = np.linspace(first, last, 4097, dtype=np.float32)
    effect = PitchShift()
    expected = effect(x, rate, automation={"semitones": curve})
    assert_array_equal(effect(x, rate, automation={"semitones": curve}), expected)
    assert_array_equal(PitchShift()(x, rate, automation={"semitones": curve}), expected)
