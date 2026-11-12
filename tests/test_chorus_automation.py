"""Chorus targets preserve LFO phase, native smoothers and private scratch capacity."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import Chorus

NAMES = ["rate_hz", "depth", "centre_delay_ms", "feedback", "mix"]


def signal(channels=2, frames=6000, rate=48000):
    t = np.arange(frames) / rate
    x = 0.35 * np.sin(2 * np.pi * 173 * t) + 0.1 * np.cos(2 * np.pi * 3337 * t)
    return np.stack([x * 0.63**c for c in range(channels)]).astype(np.float32)


def controls(frames=1025):
    return {
        k: np.linspace(a, b, frames, dtype=np.float32)
        for k, a, b in zip(NAMES, [0.5, 0.1, 4, -0.3, 0.2], [3, 0.6, 12, 0.4, 0.7])
    }


def pair(channels, rate, curves, history):
    initial = {k: float(v[0]) for k, v in curves.items()}
    a, b = Chorus(**initial), Chorus(**initial)
    for effect in [a, b]:
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
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
def test_five_targets_lfo_and_full_ramp_future(rate, channels, history, buffer_size):
    curves, audio = controls(), signal(channels, 1025, rate)
    effect, scalar = pair(channels, rate, curves, history)
    reset = history != "warm"
    saved_audio = audio.copy()
    saved = {k: v.copy() for k, v in curves.items()}
    actual = effect(audio, rate, reset=reset, buffer_size=buffer_size, automation=curves)
    assert_array_equal(actual, replay(scalar, audio, curves, rate, reset))
    assert_array_equal(audio, saved_audio)
    for name, curve in curves.items():
        assert getattr(effect, name) == float(curve[-1])
        assert_array_equal(curve, saved[name])
    future = signal(channels, 8192, rate)
    assert future.shape[1] / rate > 0.05
    assert_array_equal(effect(future, rate, reset=False), scalar(future, rate, reset=False))
    silence = np.zeros((channels, 4096), np.float32)
    tail = effect(silence, rate, reset=False)
    assert_array_equal(tail, scalar(silence, rate, reset=False))
    assert np.max(np.abs(tail)) > 1e-6


@pytest.mark.parametrize("shape", ["ramp", "jump"])
def test_tiny_then_large_chunks_and_mapping_order_preserve_phase(shape):
    curves = controls(4096)
    if shape == "jump":
        curves = {
            k: np.where(np.arange(4096) < 113, v[0], v[-1]).astype(np.float32)
            for k, v in curves.items()
        }
    initial = {k: float(v[0]) for k, v in curves.items()}
    effect, scalar = Chorus(**initial), Chorus(**initial)
    prefix = signal(frames=3500)
    effect(prefix, 48000, buffer_size=1)
    scalar(prefix, 48000, buffer_size=16384)
    audio = signal(frames=4096)
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
    assert_array_equal(np.concatenate(parts, axis=1), expected)
    future = signal(frames=8192)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


@pytest.mark.parametrize(
    "name,bad",
    [
        ("rate_hz", -1e-100),
        ("rate_hz", 100.00000001),
        ("rate_hz", np.nan),
        ("depth", -1e-100),
        ("depth", 1.00000001),
        ("depth", np.inf),
        ("centre_delay_ms", np.nan),
        ("centre_delay_ms", np.inf),
        ("centre_delay_ms", 1e100),
        ("feedback", -1.00000001),
        ("feedback", 1.00000001),
        ("feedback", np.nan),
        ("mix", -1e-100),
        ("mix", 1.00000001),
        ("mix", np.inf),
    ],
)
@pytest.mark.parametrize("reset", [False, True])
def test_invalid_late_values_preserve_targets_phase_and_history(name, bad, reset):
    curves = {k: v.astype(np.float64) for k, v in controls().items()}
    effect, scalar = pair(2, 48000, curves, "warm")
    curves[name][-1] = bad
    with pytest.raises(ValueError):
        effect(signal(frames=1025), 48000, reset=reset, automation=curves)
    for k in NAMES:
        assert getattr(effect, k) == getattr(scalar, k)
    future = signal(frames=8192)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("reset", [False, True])
def test_omitted_invalid_scalar_preflight_does_not_reset(name, reset):
    effect, scalar = pair(2, 48000, controls(), "warm")
    old = getattr(effect, name)
    setattr(effect, name, np.nan)
    supplied = "mix" if name != "mix" else "depth"
    with pytest.raises(ValueError):
        effect(signal(frames=1025), 48000, reset=reset, automation={supplied: np.full(1025, 0.9)})
    # The scalar setters themselves retarget native smoothers. Exclude that
    # intentional invalid-scalar excursion from the failed call's atomicity.
    setattr(effect, name, old)
    setattr(scalar, name, np.nan)
    setattr(scalar, name, old)
    future = signal(frames=8192)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


@pytest.mark.parametrize("rate", [0, 1e-100, 99.99, 5e9, 1e12, np.inf, np.nan])
@pytest.mark.parametrize("reset", [False, True])
def test_unsafe_format_rejected_before_reset(rate, reset):
    effect, scalar = pair(2, 48000, controls(), "warm")
    with pytest.raises(ValueError):
        effect(signal(frames=1025), rate, reset=reset, automation={"mix": np.zeros(1025)})
    assert effect.mix == scalar.mix
    future = signal(frames=8192)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


@pytest.mark.parametrize(
    "name,value",
    [
        ("rate_hz", 0),
        ("rate_hz", 100),
        ("depth", 0),
        ("depth", 1),
        ("feedback", -1),
        ("feedback", 1),
        ("mix", 0),
        ("mix", 1),
    ],
)
def test_endpoints_retain_original_dsp(name, value):
    audio = signal()
    result = Chorus()(audio, 48000, automation={name: np.full(6000, value)})
    assert_array_equal(result, Chorus(**{name: value})(audio, 48000))
    assert np.isfinite(result).all()


@pytest.mark.parametrize(
    "requested,effective",
    [
        (-10, 1),
        (0, 1),
        (0.5, 1),
        (1, 1),
        (100, 100),
        (150, 100),
        (float(np.finfo(np.float32).max), 100),
    ],
)
def test_centre_delay_keeps_original_clamp_and_requested_getter(requested, effective):
    audio = signal()
    effect = Chorus()
    actual = effect(audio, 48000, automation={"centre_delay_ms": np.full(6000, requested)})
    assert effect.centre_delay_ms == float(np.float32(requested))
    assert_array_equal(actual, Chorus(centre_delay_ms=effective)(audio, 48000))


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("reset", [False, True])
def test_cold_constants_readonly_float64_and_layout(transpose, reset):
    audio = signal().astype(np.float64)
    if transpose:
        audio = audio.T
    audio.flags.writeable = False
    values = dict(zip(NAMES, [1.2, 0.3, 8, 0.2, 0.5]))
    curves = {k: np.full(6000, v) for k, v in values.items()}
    for v in curves.values():
        v.flags.writeable = False
    assert_array_equal(
        Chorus()(audio, 48000, reset=reset, automation=curves),
        Chorus(**values)(audio, 48000, reset=reset),
    )


@pytest.mark.parametrize(
    "name,value", [("rate_hz", 3), ("depth", 0.7), ("feedback", 0.6), ("mix", 0.8)]
)
def test_warm_targets_keep_original_ramps_instead_of_snapping(name, value):
    curves = controls()
    effect, scalar = pair(2, 48000, curves, "warm")
    audio = signal(frames=8192)
    setattr(scalar, name, value)
    actual = effect(audio, 48000, reset=False, automation={name: np.full(8192, value)})
    assert_array_equal(actual, scalar(audio, 48000, reset=False))
    assert (
        np.max(
            np.abs(
                actual[:, :2000]
                - Chorus(**{**{k: float(v[0]) for k, v in curves.items()}, name: value})(
                    audio, 48000
                )[:, :2000]
            )
        )
        > 1e-4
    )


@pytest.mark.parametrize("rate,channels", [(44100, 2), (48000, 1), (48000, 4), (96000, 2)])
def test_format_changes_prepare_with_original_semantics(rate, channels):
    effect, scalar = pair(2, 48000, controls(), "warm")
    audio, curves = signal(channels, 1025, rate), controls()
    for name, curve in curves.items():
        curve[:8] = curve[0]
        setattr(scalar, name, float(curve[0]))
    first = scalar(audio[:, :8], rate, reset=False)
    rest = replay(scalar, audio[:, 8:], {k: v[8:] for k, v in curves.items()}, rate, False)
    assert_array_equal(
        effect(audio, rate, reset=False, automation=curves), np.concatenate([first, rest], axis=1)
    )


def test_empty_strided_integer_partial_curves_and_scalar_fallback():
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
        assert_array_equal(Chorus()(signal(), 48000, automation=mapping), Chorus()(signal(), 48000))
    with pytest.raises(ValueError):
        effect(signal(frames=1025), 48000, automation={"phase": np.zeros(1025)})


def test_native_calls_serialize_resets():
    audio = signal(frames=1025)
    variants = [{k: v.copy() for k, v in controls().items()} for _ in range(6)]
    for i, curve in enumerate(variants):
        curve["mix"] *= (i + 1) / 6
    expected = [Chorus()(audio, 48000, automation=curve) for curve in variants]
    effect = Chorus()
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(pool.map(lambda curve: effect(audio, 48000, automation=curve), variants))
    for a, b in zip(actual, expected):
        assert_array_equal(a, b)


def test_lowest_supported_rate_and_maximum_lfo_target_match_scalar():
    audio = signal(frames=1000, rate=100)
    assert_array_equal(
        Chorus()(audio, 100, automation={"rate_hz": np.full(1000, 100)}),
        Chorus(rate_hz=100)(audio, 100),
    )
