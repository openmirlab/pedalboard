"""Delay curves retain integer read-head behavior, feedback and channel history."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import Delay

NAMES = ["delay_seconds", "feedback", "mix"]


def signal(channels=2, frames=6000, rate=48000):
    t = np.arange(frames) / rate
    x = 0.35 * np.sin(2 * np.pi * 173 * t) + 0.1 * np.cos(2 * np.pi * 3337 * t)
    return np.stack([x * 0.63**c for c in range(channels)]).astype(np.float32)


def controls(frames=1025):
    return {
        k: np.linspace(a, b, frames, dtype=np.float32)
        for k, a, b in zip(NAMES, [0.003, 0.2, 0.2], [0.018, 0.8, 0.75])
    }


def pair(channels, rate, curves, history):
    initial = {k: float(v[0]) for k, v in curves.items()}
    a, b = Delay(**initial), Delay(**initial)
    for effect in [a, b]:
        # Preallocate the original scalar reference before warming its history.
        effect(np.zeros((channels, 16384), np.float32), rate, buffer_size=16384)
        if history != "cold":
            effect(signal(channels, rate=rate), rate, reset=False)
    return a, b


def replay(effect, audio, curves, rate, reset):
    parts = []
    for i in range(audio.shape[1]):
        for name in NAMES:
            if name in curves:
                setattr(effect, name, float(curves[name][i]))
        parts.append(effect(audio[:, i : i + 1], rate, buffer_size=1, reset=reset and i == 0))
    return np.concatenate(parts, axis=1)


@pytest.mark.parametrize("rate", [44100, 48000, 96000])
@pytest.mark.parametrize("channels", [1, 2, 4])
@pytest.mark.parametrize("history", ["cold", "warm", "warm_reset"])
@pytest.mark.parametrize("buffer_size", [1, 31, 16384])
def test_synchronized_targets_and_nonzero_future_echoes(rate, channels, history, buffer_size):
    curves, audio = controls(), signal(channels, 1025, rate)
    effect, scalar = pair(channels, rate, curves, history)
    reset = history != "warm"
    saved = {k: v.copy() for k, v in curves.items()}
    actual = effect(audio, rate, reset=reset, buffer_size=buffer_size, automation=curves)
    assert_array_equal(actual, replay(scalar, audio, curves, rate, reset))
    for name, curve in curves.items():
        assert getattr(effect, name) == float(curve[-1])
        assert_array_equal(curve, saved[name])
    future = np.zeros((channels, 8192), np.float32)
    tail = effect(future, rate, reset=False)
    assert_array_equal(tail, scalar(future, rate, reset=False))
    assert np.max(np.abs(tail)) > 1e-5


@pytest.mark.parametrize("shape", ["jump", "bypass"])
def test_fragmentation_reversed_mapping_and_block_growth_preserve_echoes(shape):
    curves = controls(4096)
    if shape == "jump":
        curves = {
            k: np.where(np.arange(4096) < 113, v[0], v[-1]).astype(np.float32)
            for k, v in curves.items()
        }
    else:
        curves["delay_seconds"][100:800] = 0
    initial = {k: float(v[0]) for k, v in curves.items()}
    effect, scalar = Delay(**initial), Delay(**initial)
    prefix = signal(frames=3500)
    effect(prefix, 48000, buffer_size=1)
    scalar(np.zeros((2, 16384), np.float32), 48000, buffer_size=16384)
    scalar(prefix, 48000)
    audio = np.zeros((2, 4096), np.float32)
    expected = replay(scalar, audio, curves, 48000, False)
    parts, start = [], 0
    for end, size in [(1, 1), (3, 2), (40, 32), (114, 128), (4096, 16384)]:
        parts.append(
            effect(
                audio[:, start:end],
                48000,
                reset=False,
                buffer_size=size,
                automation={k: v[start:end] for k, v in reversed(list(curves.items()))},
            )
        )
        start = end
    actual = np.concatenate(parts, axis=1)
    assert_array_equal(actual, expected)
    assert np.max(np.abs(actual)) > 1e-5
    future = np.zeros((2, 8192), np.float32)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


@pytest.mark.parametrize("name,upper", [("delay_seconds", 30), ("feedback", 1), ("mix", 1)])
@pytest.mark.parametrize("bad", ["below", "above", "nan", "inf"])
@pytest.mark.parametrize("reset", [False, True])
def test_invalid_late_targets_reject_before_reset_or_mutation(name, upper, bad, reset):
    curves = {k: v.astype(np.float64) for k, v in controls().items()}
    effect, scalar = pair(2, 48000, curves, "warm")
    curves[name][-1] = {"below": -1e-100, "above": upper + 1e-8, "nan": np.nan, "inf": np.inf}[bad]
    with pytest.raises(ValueError):
        effect(signal(frames=1025), 48000, reset=reset, automation=curves)
    for k in NAMES:
        assert getattr(effect, k) == getattr(scalar, k)
    future = np.zeros((2, 8192), np.float32)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("reset", [False, True])
def test_preflight_validates_omitted_nonfinite_scalars(name, reset):
    effect, scalar = pair(2, 48000, controls(), "warm")
    old = getattr(effect, name)
    setattr(effect, name, np.nan)
    supplied = "mix" if name != "mix" else "feedback"
    with pytest.raises(ValueError):
        effect(signal(frames=1025), 48000, reset=reset, automation={supplied: np.full(1025, 0.9)})
    setattr(effect, name, old)
    future = np.zeros((2, 8192), np.float32)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


@pytest.mark.parametrize("rate", [np.inf, np.nan, 0, -1, 1e12, 36_000_000])
@pytest.mark.parametrize("reset", [False, True])
def test_unsafe_rate_rejected_before_tail_reset(rate, reset):
    effect, scalar = pair(2, 48000, controls(), "warm")
    with pytest.raises(ValueError):
        effect(signal(frames=1025), rate, reset=reset, automation={"mix": np.zeros(1025)})
    assert effect.mix == scalar.mix
    future = np.zeros((2, 8192), np.float32)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


@pytest.mark.parametrize("seconds,index", [(0, 0), (0.0005, 30001), (30, 30000)])
def test_zero_subsample_and_maximum_delay_keep_original_read_behavior(seconds, index):
    audio = np.zeros(30010, np.float32)
    audio[0] = 1
    actual = Delay(feedback=0, mix=1)(
        audio, 1000, automation={"delay_seconds": np.full(len(audio), seconds)}
    )
    expected = np.zeros_like(audio)
    expected[index] = 1
    assert_array_equal(actual, expected)
    assert_array_equal(actual, Delay(delay_seconds=seconds, feedback=0, mix=1)(audio, 1000))


def test_impulse_spacing_feedback_and_linear_mix():
    audio = np.zeros(1000, np.float32)
    audio[0] = 1
    result = Delay(delay_seconds=0.125)(
        audio, 1000, automation={"feedback": np.full(1000, 0.5), "mix": np.full(1000, 0.75)}
    )
    expected = np.zeros_like(audio)
    expected[0] = 0.25
    for n in range(1, 8):
        expected[n * 125] = 0.75 * 0.5 ** (n - 1)
    assert_array_equal(result, expected)


def test_seconds_use_original_float32_then_integer_truncation():
    audio = np.zeros(1000, np.float32)
    audio[0] = 1
    result = Delay(mix=1)(audio, 48000, automation={"delay_seconds": np.full(1000, 0.01)})
    expected_index = int(float(np.float32(0.01)) * 48000)
    assert expected_index == 479
    assert np.flatnonzero(result).tolist() == [expected_index]


def test_zero_seconds_bypass_pauses_history_and_does_not_record_input():
    effect = Delay(delay_seconds=0.125, feedback=0, mix=1)
    prefix = np.zeros(100, np.float32)
    prefix[0] = 1
    effect(prefix, 1000, buffer_size=1)
    bypass = np.ones(50, np.float32) * 0.25
    assert_array_equal(
        effect(bypass, 1000, reset=False, automation={"delay_seconds": np.zeros(50)}), bypass
    )
    result = effect(
        np.zeros(200, np.float32),
        1000,
        reset=False,
        automation={"delay_seconds": np.full(200, 0.125)},
    )
    expected = np.zeros(200, np.float32)
    expected[25] = 1
    assert_array_equal(result, expected)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("reset", [False, True])
def test_constant_readonly_float64_layout(transpose, reset):
    audio = signal().astype(np.float64)
    if transpose:
        audio = audio.T
    audio.flags.writeable = False
    values = {"delay_seconds": 0.015, "feedback": 0.7, "mix": 0.5}
    curves = {k: np.full(6000, v) for k, v in values.items()}
    for v in curves.values():
        v.flags.writeable = False
    assert_array_equal(
        Delay()(audio, 48000, reset=reset, automation=curves),
        Delay(**values)(audio, 48000, reset=reset),
    )


@pytest.mark.parametrize("rate,channels", [(44100, 2), (48000, 1), (48000, 4), (96000, 2)])
def test_format_changes_reprepare_with_original_semantics(rate, channels):
    effect, scalar = pair(2, 48000, controls(), "warm")
    audio, curves = signal(channels, 1025, rate), controls()
    # Establish a changed channel count with an unambiguous first block. A4x1
    # scalar call still carrying a2-channel hint otherwise means4 mono frames.
    for name, curve in curves.items():
        curve[:8] = curve[0]
        setattr(scalar, name, float(curve[0]))
    start = scalar(audio[:, :8], rate, reset=False)
    remaining = replay(scalar, audio[:, 8:], {k: v[8:] for k, v in curves.items()}, rate, False)
    assert_array_equal(
        effect(audio, rate, reset=False, automation=curves),
        np.concatenate([start, remaining], axis=1),
    )


def test_empty_partial_integer_strided_and_scalar_fallback():
    effect, scalar = pair(2, 48000, controls(), "warm")
    empty = effect(np.empty((2, 0), np.float32), 48000, automation={"mix": np.empty(0)})
    assert empty.shape == (2, 0)
    curve = np.ones(2050, np.int32)[::2]
    curve.flags.writeable = False
    scalar.mix = 1
    assert_array_equal(
        effect(signal(frames=1025), 48000, reset=False, automation={"mix": curve}),
        scalar(signal(frames=1025), 48000, reset=False),
    )
    for mapping in [None, {}]:
        assert_array_equal(Delay()(signal(), 48000, automation=mapping), Delay()(signal(), 48000))
    with pytest.raises(ValueError):
        effect(signal(frames=1025), 48000, automation={"time": np.ones(1025)})


@pytest.mark.parametrize("name,value", [("feedback", 0), ("feedback", 1), ("mix", 0), ("mix", 1)])
def test_endpoints_retain_original_dsp(name, value):
    audio = signal()
    assert_array_equal(
        Delay(delay_seconds=0.015)(audio, 48000, automation={name: np.full(6000, value)}),
        Delay(delay_seconds=0.015, **{name: value})(audio, 48000),
    )


def test_native_calls_serialize_independent_resets():
    audio = signal(frames=1025)
    variants = [{k: v.copy() for k, v in controls().items()} for _ in range(6)]
    for i, curve in enumerate(variants):
        curve["mix"] *= (i + 1) / 6
    expected = [Delay()(audio, 48000, automation=curve) for curve in variants]
    effect = Delay()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda curve: effect(audio, 48000, automation=curve), variants))
    for actual, want in zip(results, expected):
        assert_array_equal(actual, want)
