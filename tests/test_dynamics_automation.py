"""Multi-control dynamics automation must preserve detector and ramp history."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import Compressor, Limiter, NoiseGate

CLASSES = [Compressor, NoiseGate, Limiter]
RANGES = {
    "threshold_db": (-36.0, -8.0),
    "ratio": (1.5, 8.0),
    "attack_ms": (0.1, 15.0),
    "release_ms": (20.0, 180.0),
}


def controls(cls, frames=257):
    return {
        name: np.linspace(low, high, frames, dtype=np.float32)
        for name, (low, high) in RANGES.items()
        if cls is not Limiter or name in ["threshold_db", "release_ms"]
    }


def initial_values(curves):
    return {name: float(curve[0]) for name, curve in curves.items()}


def signal(channels=2, frames=257):
    t = np.arange(frames) / 48000
    wave = np.sin(t * 2 * np.pi * 173) + 0.3 * np.cos(t * 2 * np.pi * 997)
    wave *= np.where((np.arange(frames) // 37) % 2, 0.025, 0.65)
    return np.stack([wave * (0.7**ch) for ch in range(channels)]).astype(np.float32)


def scalar_reference(cls, audio, curves, rate, warm=None):
    effect = cls(**initial_values(curves))
    if warm is not None:
        effect(warm, rate)
    elif len(audio) > 1:
        # Seed layout only; process(reset=True) below clears the detector/ramp.
        effect(np.zeros((audio.shape[0], 8), dtype=np.float32), rate)
    result = []
    for i in range(audio.shape[1]):
        for name, curve in curves.items():
            setattr(effect, name, float(curve[i]))
        result.append(effect(audio[:, i : i + 1], rate, reset=warm is None and i == 0))
    return np.concatenate(result, axis=1), effect


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
@pytest.mark.parametrize("warm", [False, True])
def test_multi_control_matches_scalar(cls, rate, channels, buffer_size, warm):
    audio = signal(channels)
    curves = controls(cls)
    prefix = signal(channels, 1024) if warm else None
    expected, reference = scalar_reference(cls, audio, curves, rate, prefix)
    effect = cls(**initial_values(curves))
    if prefix is not None:
        effect(prefix, rate)
    original, original_curves = audio.copy(), {k: v.copy() for k, v in curves.items()}
    actual = effect(audio, rate, buffer_size=buffer_size, reset=not warm, automation=curves)
    assert_array_equal(actual, expected)
    assert_array_equal(audio, original)
    for name, curve in curves.items():
        assert getattr(effect, name) == float(curve[-1])
        assert_array_equal(curve, original_curves[name])
    # The next audio block proves the detector and makeup ramp also agree.
    future = signal(channels, 512) * 0.1
    assert_array_equal(effect(future, rate, reset=False), reference(future, rate, reset=False))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("reset", [False, True])
def test_cold_constant_matches_original_scalar(cls, rate, transpose, reset):
    curves = {
        name: np.full(257, curve[0], dtype=np.float64) for name, curve in controls(cls).items()
    }
    audio = signal().astype(np.float64)
    if transpose:
        audio = audio.T
    audio.flags.writeable = False
    for curve in curves.values():
        curve.flags.writeable = False
    expected = cls(**initial_values(curves))(audio, rate, reset=reset)
    assert_array_equal(cls()(audio, rate, reset=reset, automation=curves), expected)


@pytest.mark.parametrize("cls", CLASSES)
def test_growing_chunks_preserve_warm_state(cls):
    audio = signal(frames=1024)
    curves = controls(cls, 1024)
    prefix = signal(frames=32)
    expected, reference = scalar_reference(cls, audio, curves, 48000, prefix)
    effect = cls(**initial_values(curves))
    effect(prefix, 48000)
    parts, start = [], 0
    for end, size in [(1, 1), (3, 2), (20, 32), (21, 64), (112, 128), (1024, 8192)]:
        parts.append(
            effect(
                audio[:, start:end],
                48000,
                buffer_size=size,
                reset=False,
                automation={k: v[start:end] for k, v in curves.items()},
            )
        )
        start = end
    assert_array_equal(np.concatenate(parts, axis=1), expected)
    future = signal(frames=1536) * 0.07
    assert_array_equal(effect(future, 48000, reset=False), reference(future, 48000, reset=False))


@pytest.mark.parametrize("cls", CLASSES)
def test_dict_order_and_simultaneous_jumps(cls):
    curves = controls(cls)
    for curve in curves.values():
        curve[:] = np.where(np.arange(257) < 113, curve[0], curve[-1])
    audio = signal()
    a = cls()(audio, 48000, automation=curves)
    b = cls()(audio, 48000, automation=dict(reversed(list(curves.items()))))
    expected, _ = scalar_reference(cls, audio, curves, 48000)
    assert_array_equal(a, b)
    assert_array_equal(a, expected)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "bad",
    [
        "negative",
        "nan",
        "infinite",
        "overflow",
        "length",
        "ndim",
        "scalar",
        "bool",
        "complex",
        "unknown",
        "floor",
    ],
)
def test_invalid_later_curve_preserves_parameters_and_memory(cls, bad):
    curves = controls(cls)
    initial = initial_values(curves)
    actual, reference = cls(**initial), cls(**initial)
    prefix = signal(frames=512)
    actual(prefix, 48000)
    reference(prefix, 48000)
    invalid = {k: v.copy() for k, v in curves.items()}
    name = "release_ms"
    if bad in ["negative", "nan", "infinite", "overflow"]:
        invalid[name][-1] = {
            "negative": -1,
            "nan": np.nan,
            "infinite": np.inf,
            "overflow": np.finfo(np.float32).max,
        }[bad]
        if bad == "overflow":
            invalid[name] = np.full(257, 1e100)
    elif bad == "length":
        invalid[name] = invalid[name][:-1]
    elif bad == "ndim":
        invalid[name] = invalid[name][None, :]
    elif bad == "scalar":
        invalid[name] = 100.0
    elif bad == "bool":
        invalid[name] = invalid[name].astype(bool)
    elif bad == "complex":
        invalid[name] = invalid[name].astype(complex)
    elif bad == "unknown":
        invalid["not_a_control"] = np.zeros(257)
    else:
        invalid["threshold_db"][-1] = -200
    with pytest.raises((ValueError, TypeError)):
        actual(signal(), 48000, automation=invalid)
    assert {k: getattr(actual, k) for k in curves} == initial
    quiet = signal(frames=512) * 0.05
    assert_array_equal(actual(quiet, 48000, reset=False), reference(quiet, 48000, reset=False))


@pytest.mark.parametrize("cls", [Compressor, NoiseGate])
@pytest.mark.parametrize(
    "name,value",
    [("ratio", 0.5), ("attack_ms", -1), ("threshold_db", -np.inf), ("threshold_db", 10000)],
)
def test_reject_unsafe_dynamics_domains(cls, name, value):
    with pytest.raises(ValueError):
        cls()(signal(), 48000, automation={name: np.full(257, value)})


@pytest.mark.parametrize("cls", CLASSES)
def test_omitted_controls_empty_stream_and_noncontiguous_curves(cls):
    effect = cls(threshold_db=-24, release_ms=150)
    reference = cls(threshold_db=-24, release_ms=150)
    prefix = signal(frames=512)
    effect(prefix, 48000)
    reference(prefix, 48000)
    empty = {name: np.empty(0) for name in controls(cls)}
    result = effect(np.empty((2, 0), np.float32), 48000, reset=True, automation=empty)
    assert result.shape == (2, 0)
    assert effect.threshold_db == -24 and effect.release_ms == 150
    audio = signal()
    curve = np.repeat(np.full(257, 150.0, dtype=np.float64), 2)[::2]
    curve.flags.writeable = False
    result = effect(audio, 48000, reset=False, automation={"release_ms": curve})
    assert_array_equal(result, reference(audio, 48000, reset=False))
    assert effect.threshold_db == -24


@pytest.mark.parametrize("cls", CLASSES)
def test_zero_time_and_integer_curves(cls):
    curves = {
        name: np.full(257, curve[0], dtype=np.float32) for name, curve in controls(cls).items()
    }
    curves["release_ms"] = np.zeros(257, dtype=np.int32)
    if cls is not Limiter:
        curves["attack_ms"] = np.zeros(257, dtype=np.int32)
    expected = cls(**initial_values(curves))(signal(), 48000)
    assert_array_equal(cls()(signal(), 48000, automation=curves), expected)


@pytest.mark.parametrize("cls", CLASSES)
def test_concurrent_automated_requests(cls):
    audio = signal()
    variants = []
    for delta in range(6):
        curves = controls(cls)
        curves["threshold_db"] += delta
        variants.append(curves)
    expected = [cls()(audio, 48000, automation=c) for c in variants]
    effect = cls()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda c: effect(audio, 48000, automation=c), variants))
    for result, want in zip(results, expected):
        assert_array_equal(result, want)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("new_rate,new_channels", [(44100, 2), (48000, 1)])
def test_format_changes_still_prepare_dsp(cls, new_rate, new_channels):
    curves = controls(cls)
    effect = cls(**initial_values(curves))
    effect(signal(frames=512), 48000)
    audio = signal(channels=new_channels)
    expected = cls()(audio, new_rate, reset=False, automation=curves)
    actual = effect(audio, new_rate, reset=False, automation=curves)
    assert_array_equal(actual, expected)
