"""First-order curves must retain original coefficient arithmetic and IIR memory."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import HighpassFilter, LowpassFilter

CLASSES = [LowpassFilter, HighpassFilter]


def signal(channels=2, frames=257):
    t = np.arange(frames) / 48000
    wave = 0.5 * np.sin(2 * np.pi * 173 * t) + 0.2 * np.cos(2 * np.pi * 3337 * t)
    return np.stack([wave * (0.63**ch) for ch in range(channels)]).astype(np.float32)


def reference(cls, audio, curve, rate, warm=None):
    effect = cls(cutoff_frequency_hz=float(curve[0]))
    prefix = np.zeros((audio.shape[0], 512), dtype=np.float32) if warm is None else warm
    effect(prefix, rate)
    parts = []
    for i, value in enumerate(curve):
        effect.cutoff_frequency_hz = float(value)
        parts.append(
            effect(audio[:, i : i + 1], rate, buffer_size=1, reset=warm is None and i == 0)
        )
    return np.concatenate(parts, axis=1), effect


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rate", [44100, 48000, 96000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
@pytest.mark.parametrize("warm", [False, True])
def test_sweep_matches_scalar_and_future(cls, rate, channels, buffer_size, warm):
    audio = signal(channels)
    curve = np.geomspace(30, 18000, audio.shape[1]).astype(np.float32)
    prefix = signal(channels, 512) if warm else None
    expected, scalar = reference(cls, audio, curve, rate, prefix)
    effect = cls(cutoff_frequency_hz=float(curve[0]))
    if prefix is not None:
        effect(prefix, rate)
    saved_audio, saved_curve = audio.copy(), curve.copy()
    actual = effect(
        audio,
        rate,
        reset=not warm,
        buffer_size=buffer_size,
        automation={"cutoff_frequency_hz": curve},
    )
    assert_array_equal(actual, expected)
    assert_array_equal(audio, saved_audio)
    assert_array_equal(curve, saved_curve)
    assert effect.cutoff_frequency_hz == float(curve[-1])
    future = signal(channels, 128)
    assert_array_equal(effect(future, rate, reset=False), scalar(future, rate, reset=False))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("reset", [False, True])
def test_cold_constant_float64_readonly(cls, rate, transpose, reset):
    audio = signal().astype(np.float64)
    if transpose:
        audio = audio.T
    audio.flags.writeable = False
    curve = np.full(257, 1000.0)
    curve.flags.writeable = False
    assert_array_equal(
        cls()(audio, rate, reset=reset, automation={"cutoff_frequency_hz": curve}),
        cls(cutoff_frequency_hz=1000)(audio, rate, reset=reset),
    )


@pytest.mark.parametrize("cls", CLASSES)
def test_jumps_fragmentation_and_block_growth(cls):
    audio = signal(frames=1024)
    curve = np.where(np.arange(1024) < 113, 300.0, 9000.0).astype(np.float32)
    prefix = signal(frames=32)
    expected, scalar = reference(cls, audio, curve, 48000, prefix)
    effect = cls(cutoff_frequency_hz=float(curve[0]))
    effect(prefix, 48000)
    chunks, start = [], 0
    for end, size in [(1, 1), (3, 2), (20, 32), (21, 64), (112, 128), (1024, 8192)]:
        chunks.append(
            effect(
                audio[:, start:end],
                48000,
                reset=False,
                buffer_size=size,
                automation={"cutoff_frequency_hz": curve[start:end]},
            )
        )
        start = end
    assert_array_equal(np.concatenate(chunks, axis=1), expected)
    assert_array_equal(
        effect(signal(frames=1536), 48000, reset=False),
        scalar(signal(frames=1536), 48000, reset=False),
    )


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "bad",
    [
        0,
        -1,
        np.nan,
        np.inf,
        -np.inf,
        24000,
        30000,
        1e100,
        1e-100,
        1e-6,
        np.nextafter(24000.0, 0.0),
    ],
)
def test_reject_unsafe_cutoff_without_mutation(cls, bad):
    effect, scalar = cls(cutoff_frequency_hz=1000), cls(cutoff_frequency_hz=1000)
    effect(signal(frames=512), 48000)
    scalar(signal(frames=512), 48000)
    curve = np.full(257, 2000.0)
    curve[-1] = bad
    with pytest.raises(ValueError):
        effect(signal(), 48000, automation={"cutoff_frequency_hz": curve})
    assert effect.cutoff_frequency_hz == 1000
    assert_array_equal(effect(signal(), 48000, reset=False), scalar(signal(), 48000, reset=False))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "curve",
    [
        np.ones(256),
        np.ones((1, 257)),
        np.ones(257, bool),
        np.ones(257, complex),
        1000.0,
        [1000.0] * 257,
    ],
)
def test_invalid_curve_shape_and_type(cls, curve):
    with pytest.raises((ValueError, TypeError)):
        cls()(signal(), 48000, automation={"cutoff_frequency_hz": curve})


@pytest.mark.parametrize("cls", CLASSES)
def test_cutoff_domain_uses_current_rate(cls):
    curve = np.full(257, 23000.0)
    with pytest.raises(ValueError):
        cls()(signal(), 44100, automation={"cutoff_frequency_hz": curve})
    assert np.isfinite(cls()(signal(), 48000, automation={"cutoff_frequency_hz": curve})).all()


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rate,channels", [(44100, 2), (48000, 1), (96000, 2)])
def test_format_changes_prepare_from_first_target(cls, rate, channels):
    effect = cls(cutoff_frequency_hz=1500)
    effect(signal(frames=512), 48000)
    low, high = (30000, 40000) if rate == 96000 else (300, 9000)
    curve = np.geomspace(low, high, 257).astype(np.float32)
    actual = effect(signal(channels), rate, reset=False, automation={"cutoff_frequency_hz": curve})
    expected = cls()(signal(channels), rate, reset=False, automation={"cutoff_frequency_hz": curve})
    assert_array_equal(actual, expected)


@pytest.mark.parametrize("cls", CLASSES)
def test_empty_strided_integer_and_legacy_fallback(cls):
    effect, scalar = cls(cutoff_frequency_hz=1000), cls(cutoff_frequency_hz=1000)
    effect(signal(frames=512), 48000)
    scalar(signal(frames=512), 48000)
    effect(np.empty((2, 0), np.float32), 48000, automation={"cutoff_frequency_hz": np.empty(0)})
    curve = np.repeat(np.full(257, 1000, dtype=np.int32), 2)[::2]
    curve.flags.writeable = False
    assert_array_equal(
        effect(signal(), 48000, reset=False, automation={"cutoff_frequency_hz": curve}),
        scalar(signal(), 48000, reset=False),
    )
    for mapping in [None, {}]:
        assert_array_equal(cls()(signal(), 48000, automation=mapping), cls()(signal(), 48000))
    with pytest.raises(ValueError):
        effect(signal(), 48000, automation={"gain_db": np.ones(257)})


@pytest.mark.parametrize("cls", CLASSES)
def test_cutoff_response_and_extreme_sweep_are_finite(cls):
    rate, cutoff = 48000, 1000
    t = np.arange(rate) / rate
    tone = np.sin(2 * np.pi * cutoff * t).astype(np.float32)
    output = cls()(tone, rate, automation={"cutoff_frequency_hz": np.full(rate, cutoff)})
    ratio = np.sqrt(np.mean(output[rate // 2 :] ** 2) / np.mean(tone[rate // 2 :] ** 2))
    assert ratio == pytest.approx(1 / np.sqrt(2), rel=1e-4)
    curve = np.where(np.arange(rate) % 2, 1.0, rate / 2 - 100).astype(np.float32)
    rapid = cls()(tone, rate, automation={"cutoff_frequency_hz": curve})
    expected, _ = reference(cls, tone[None, :1024], curve[:1024], rate)
    assert_array_equal(rapid[:1024], expected[0])
    assert np.isfinite(rapid).all()


@pytest.mark.parametrize("cls", CLASSES)
def test_concurrent_calls_remain_isolated(cls):
    audio = signal()
    curves = [np.linspace(100 + 100 * i, 9000, 257, dtype=np.float32) for i in range(6)]
    expected = [cls()(audio, 48000, automation={"cutoff_frequency_hz": c}) for c in curves]
    shared = cls()
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(
            pool.map(lambda c: shared(audio, 48000, automation={"cutoff_frequency_hz": c}), curves)
        )
    for result, want in zip(actual, expected):
        assert_array_equal(result, want)


@pytest.mark.parametrize("cls", CLASSES)
def test_safe_float32_near_nyquist_is_accepted(cls):
    cutoff = float(np.nextafter(np.float32(24000), np.float32(0)))
    audio = signal()
    result = cls()(audio, 48000, automation={"cutoff_frequency_hz": np.full(257, cutoff)})
    assert_array_equal(result, cls(cutoff_frequency_hz=cutoff)(audio, 48000))
    assert np.isfinite(result).all()
