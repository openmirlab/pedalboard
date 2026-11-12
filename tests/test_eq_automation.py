"""Synchronized EQ targets must preserve coefficient arithmetic and filter history."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import HighShelfFilter, LowShelfFilter, PeakFilter

CLASSES = [PeakFilter, LowShelfFilter, HighShelfFilter]


def signal(channels=2, frames=257):
    t = np.arange(frames) / 48000
    wave = 0.4 * np.sin(2 * np.pi * 173 * t) + 0.2 * np.cos(2 * np.pi * 3337 * t)
    return np.stack([wave * 0.63**ch for ch in range(channels)]).astype(np.float32)


def controls(frames=257):
    return {
        "cutoff_frequency_hz": np.geomspace(120, 9000, frames).astype(np.float32),
        "gain_db": np.linspace(-9, 9, frames, dtype=np.float32),
        "q": np.linspace(0.7, 3, frames, dtype=np.float32),
    }


def initial(curves):
    return {name: float(curve[0]) for name, curve in curves.items()}


def scalar_reference(cls, audio, curves, rate, warm=None):
    effect = cls(**initial(curves))
    effect(np.zeros((audio.shape[0], 512), np.float32) if warm is None else warm, rate)
    outputs = []
    for i in range(audio.shape[1]):
        for name, curve in curves.items():
            setattr(effect, name, float(curve[i]))
        outputs.append(
            effect(audio[:, i : i + 1], rate, buffer_size=1, reset=warm is None and i == 0)
        )
    return np.concatenate(outputs, axis=1), effect


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rate", [44100, 48000, 96000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
@pytest.mark.parametrize("warm", [False, True])
def test_all_controls_match_scalar_and_future(cls, rate, channels, buffer_size, warm):
    audio, curves = signal(channels), controls()
    prefix = signal(channels, 512) if warm else None
    expected, reference = scalar_reference(cls, audio, curves, rate, prefix)
    effect = cls(**initial(curves))
    if prefix is not None:
        effect(prefix, rate)
    saved_audio, saved_curves = audio.copy(), {k: v.copy() for k, v in curves.items()}
    actual = effect(audio, rate, reset=not warm, buffer_size=buffer_size, automation=curves)
    assert_array_equal(actual, expected)
    assert_array_equal(audio, saved_audio)
    for name, curve in curves.items():
        assert getattr(effect, name) == getattr(reference, name)
        assert_array_equal(curve, saved_curves[name])
    future = signal(channels, 128)
    assert_array_equal(effect(future, rate, reset=False), reference(future, rate, reset=False))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("reset", [False, True])
def test_cold_constants_and_readonly_float64(cls, rate, transpose, reset):
    audio = signal().astype(np.float64)
    if transpose:
        audio = audio.T
    audio.flags.writeable = False
    values = {"cutoff_frequency_hz": 1000.0, "gain_db": 6.0, "q": 0.707}
    curves = {k: np.full(257, v) for k, v in values.items()}
    for curve in curves.values():
        curve.flags.writeable = False
    assert_array_equal(
        cls()(audio, rate, reset=reset, automation=curves), cls(**values)(audio, rate, reset=reset)
    )


@pytest.mark.parametrize("cls", CLASSES)
def test_simultaneous_jumps_dict_order_and_fragmentation(cls):
    audio, curves = signal(frames=1024), controls(1024)
    curves = {
        k: np.where(np.arange(1024) < 113, v[0], v[-1]).astype(np.float32)
        for k, v in curves.items()
    }
    prefix = signal(frames=32)
    expected, reference = scalar_reference(cls, audio, curves, 48000, prefix)
    effect = cls(**initial(curves))
    effect(prefix, 48000)
    pieces, start = [], 0
    for end, size in [(1, 1), (3, 2), (20, 32), (21, 64), (112, 128), (1024, 8192)]:
        sliced = {k: v[start:end] for k, v in reversed(list(curves.items()))}
        pieces.append(
            effect(audio[:, start:end], 48000, reset=False, buffer_size=size, automation=sliced)
        )
        start = end
    assert_array_equal(np.concatenate(pieces, axis=1), expected)
    assert_array_equal(
        effect(signal(frames=1536), 48000, reset=False),
        reference(signal(frames=1536), 48000, reset=False),
    )
    assert_array_equal(
        cls()(audio, 48000, automation=curves),
        cls()(audio, 48000, automation=dict(reversed(list(curves.items())))),
    )


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize(
    "bad",
    [
        "q_nan",
        "q_zero",
        "q_underflow",
        "q_overflow",
        "frequency_zero",
        "gain_floor",
        "gain_infinite",
        "joint_late",
        "shape",
        "unknown",
    ],
)
def test_invalid_curve_or_joint_combination_does_not_mutate(cls, reset, bad):
    params = {"cutoff_frequency_hz": 1000.0, "gain_db": 6.0, "q": 0.707}
    effect, reference = cls(**params), cls(**params)
    effect(signal(frames=512), 48000)
    reference(signal(frames=512), 48000)
    saved = {k: getattr(effect, k) for k in params}
    curves = controls()
    if bad == "q_nan":
        curves["q"][-1] = np.nan
    elif bad == "q_zero":
        curves["q"][-1] = 0
    elif bad == "q_underflow":
        curves["q"] = np.full(257, 1e-100)
    elif bad == "q_overflow":
        curves["q"] = np.full(257, 1e100)
    elif bad == "frequency_zero":
        curves["cutoff_frequency_hz"][-1] = 0
    elif bad == "gain_floor":
        curves["gain_db"][-1] = -100
    elif bad == "gain_infinite":
        curves["gain_db"][-1] = np.inf
    elif bad == "joint_late":
        curves["q"][-1] = 1e30  # Individually finite, denominator becomes marginal.
    elif bad == "shape":
        curves["gain_db"] = curves["gain_db"][:-1]
    else:
        curves["unknown"] = np.ones(257)
    with pytest.raises((ValueError, TypeError)):
        effect(signal(), 48000, reset=reset, automation=curves)
    assert {k: getattr(effect, k) for k in params} == saved
    assert_array_equal(
        effect(signal(), 48000, reset=False), reference(signal(), 48000, reset=False)
    )


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "name,bad", [("q", 1e30), ("gain_db", -120), ("cutoff_frequency_hz", np.nan)]
)
@pytest.mark.parametrize("reset", [False, True])
def test_joint_validation_includes_omitted_scalar_controls(cls, name, bad, reset):
    params = {"cutoff_frequency_hz": 1000.0, "gain_db": 6.0, "q": 0.707}
    effect, reference = cls(**params), cls(**params)
    effect(signal(frames=512), 48000)
    reference(signal(frames=512), 48000)
    setattr(effect, name, bad)
    supplied = "gain_db" if name != "gain_db" else "cutoff_frequency_hz"
    with pytest.raises(ValueError):
        effect(signal(), 48000, reset=reset, automation={supplied: np.full(257, params[supplied])})
    setattr(effect, name, params[name])
    assert_array_equal(
        effect(signal(), 48000, reset=False), reference(signal(), 48000, reset=False)
    )


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("name", ["cutoff_frequency_hz", "gain_db", "q"])
def test_single_strided_integer_curve_preserves_omitted_controls(cls, name):
    params = {"cutoff_frequency_hz": 1000.0, "gain_db": 6.0, "q": 1.0}
    effect, reference = cls(**params), cls(**params)
    effect(signal(frames=512), 48000)
    reference(signal(frames=512), 48000)
    empty = effect(np.empty((2, 0), np.float32), 48000, automation={name: np.empty(0)})
    assert empty.shape == (2, 0)
    curve = np.repeat(np.full(257, params[name], dtype=np.int32), 2)[::2]
    curve.flags.writeable = False
    assert_array_equal(
        effect(signal(), 48000, reset=False, automation={name: curve}),
        reference(signal(), 48000, reset=False),
    )
    for key in params:
        assert getattr(effect, key) == getattr(reference, key)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("frequency", [50000.0, 1e8])
def test_retains_original_upper_frequency_clamp(cls, frequency):
    params = {"cutoff_frequency_hz": frequency, "gain_db": 6.0, "q": 0.707}
    expected = cls(**params)(signal(), 48000)
    actual = cls()(signal(), 48000, automation={k: np.full(257, v) for k, v in params.items()})
    assert_array_equal(actual, expected)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rate,channels", [(44100, 2), (48000, 1), (96000, 2)])
def test_format_change_uses_current_rate_for_first_targets(cls, rate, channels):
    effect = cls(cutoff_frequency_hz=1000, gain_db=6)
    effect(signal(frames=512), 48000)
    curves = controls()
    if rate == 96000:
        curves["cutoff_frequency_hz"] = np.geomspace(30000, 40000, 257).astype(np.float32)
    assert_array_equal(
        effect(signal(channels), rate, reset=False, automation=curves),
        cls()(signal(channels), rate, reset=False, automation=curves),
    )


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rate", [100, 200])
def test_rejects_rate_incompatible_with_original_clamp(cls, rate):
    with pytest.raises(ValueError):
        cls()(signal(), rate, automation={"gain_db": np.zeros(257)})


@pytest.mark.parametrize("cls", CLASSES)
def test_tone_response_and_high_q_sweep(cls):
    rate = 48000
    t = np.arange(rate) / rate
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    params = {"cutoff_frequency_hz": 1000.0, "gain_db": 6.0, "q": 0.707}
    output = cls()(tone, rate, automation={k: np.full(rate, v) for k, v in params.items()})
    ratio = np.sqrt(np.mean(output[rate // 2 :] ** 2) / np.mean(tone[rate // 2 :] ** 2))
    db = 20 * np.log10(ratio)
    assert db == pytest.approx(6 if cls is PeakFilter else 3, abs=0.01)
    curves = {
        "cutoff_frequency_hz": np.geomspace(200, 8000, 1024).astype(np.float32),
        "gain_db": np.full(1024, 12, dtype=np.float32),
        "q": np.full(1024, 10, dtype=np.float32),
    }
    expected, _ = scalar_reference(cls, tone[None, :1024], curves, rate)
    actual = cls()(tone[:1024], rate, automation=curves)
    assert_array_equal(actual, expected[0])
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("cls", CLASSES)
def test_concurrent_native_processing_requests(cls):
    audio = signal()
    variants = []
    for delta in range(6):
        curves = controls()
        curves["gain_db"] += delta
        variants.append(curves)
    expected = [cls()(audio, 48000, automation=c) for c in variants]
    shared = cls()
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(pool.map(lambda c: shared(audio, 48000, automation=c), variants))
    for result, want in zip(actual, expected):
        assert_array_equal(result, want)
