"""Reverb curves preserve original gain ramps, delay memory and setter ordering."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import Reverb

NAMES = ["room_size", "damping", "wet_level", "dry_level", "width"]


def signal(channels=2, frames=12000, rate=48000):
    t = np.arange(frames) / rate
    x = 0.35 * np.sin(2 * np.pi * 173 * t) + 0.1 * np.cos(2 * np.pi * 3337 * t)
    return np.stack([x * 0.63**c for c in range(channels)]).astype(np.float32)


def controls(frames=1025):
    return {
        k: np.linspace(a, b, frames, dtype=np.float32)
        for k, a, b in zip(NAMES, [0.3, 0.2, 0.2, 0.6, 1], [0.85, 0.8, 0.8, 0.1, 0])
    }


def pair(channels, rate, curves, history):
    initial = {k: float(v[0]) for k, v in curves.items()}
    a, b = Reverb(**initial), Reverb(**initial)
    # Establish capacity on the scalar reference before warming its delay lines.
    # A small buffer_size on subsequent calls must not clear those lines.
    for effect in [a, b]:
        effect(np.zeros((channels, 32768), np.float32), rate, buffer_size=32768)
        if history != "cold":
            effect(signal(channels, rate=rate), rate, reset=False)
        if history == "freeze":
            effect.freeze_mode = 1
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
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("history", ["cold", "warm", "warm_reset", "freeze"])
@pytest.mark.parametrize("buffer_size", [1, 31, 32768])
def test_all_targets_and_nonzero_continued_tail(rate, channels, history, buffer_size):
    curves, audio = controls(), signal(channels, 1025, rate)
    effect, scalar = pair(channels, rate, curves, history)
    reset = history in ["cold", "warm_reset"]
    saved = {k: v.copy() for k, v in curves.items()}
    actual = effect(audio, rate, buffer_size=buffer_size, reset=reset, automation=curves)
    assert_array_equal(actual, replay(scalar, audio, curves, rate, reset))
    for name, v in curves.items():
        assert getattr(effect, name) == float(v[-1])
        assert_array_equal(v, saved[name])
    future = np.zeros((channels, 24000), np.float32)
    tail = effect(future, rate, reset=False)
    assert_array_equal(tail, scalar(future, rate, reset=False))
    if history in ["warm", "freeze"]:
        assert np.max(np.abs(tail)) > 1e-5


@pytest.mark.parametrize("channels", [1, 2])
def test_coupled_wet_width_updates_keep_intermediate_ramp_retargeting(channels):
    curves = {k: np.full(1025, v[0], np.float32) for k, v in controls().items()}
    curves["wet_level"][:] = 0.4
    curves["wet_level"][100:] = 0.8
    curves["width"][100:] = 0
    initial = {**curves, "wet_level": np.full(1025, 0.2, np.float32)}
    effect, scalar = pair(channels, 48000, initial, "warm")
    audio = np.zeros((channels, 1025), np.float32)
    expected = replay(scalar, audio, curves, 48000, False)
    actual = effect(audio, 48000, reset=False, automation=dict(reversed(list(curves.items()))))
    assert_array_equal(actual, expected)
    assert np.max(np.abs(actual)) > 1e-4
    # At frame100, wetGain1 returns to the same final target, but the original
    # two setters first change it, then restart its remaining 10ms ramp.
    future = np.zeros((channels, 12000), np.float32)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


def test_fragmentation_and_block_growth_do_not_clear_existing_tail():
    curves = controls(4096)
    prefix = signal(frames=3500)
    effect, scalar = Reverb(), Reverb()
    effect(prefix, 48000, buffer_size=1)
    scalar(np.zeros((2, 32768), np.float32), 48000, buffer_size=32768)
    scalar(prefix, 48000)
    audio = np.zeros((2, 4096), np.float32)
    expected = replay(scalar, audio, curves, 48000, False)
    parts, start = [], 0
    for end, size in [(1, 1), (3, 2), (40, 32), (101, 128), (4096, 32768)]:
        parts.append(
            effect(
                audio[:, start:end],
                48000,
                reset=False,
                buffer_size=size,
                automation={k: v[start:end] for k, v in curves.items()},
            )
        )
        start = end
    actual = np.concatenate(parts, axis=1)
    assert_array_equal(actual, expected)
    assert np.max(np.abs(actual)) > 1e-4
    future = np.zeros((2, 24000), np.float32)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bad", [-1e-8, 1.00000001, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("reset", [False, True])
def test_invalid_late_curve_values_preserve_targets_and_tail(name, bad, reset):
    curves = {k: v.astype(np.float64) for k, v in controls().items()}
    effect, scalar = pair(2, 48000, curves, "warm")
    curves[name][-1] = bad
    with pytest.raises(ValueError):
        effect(signal(frames=1025), 48000, reset=reset, automation=curves)
    for k in NAMES:
        assert getattr(effect, k) == getattr(scalar, k)
    future = np.zeros((2, 12000), np.float32)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


@pytest.mark.parametrize("name", NAMES + ["freeze_mode"])
@pytest.mark.parametrize("reset", [False, True])
def test_preflight_rejects_omitted_nonfinite_scalar_before_reset(name, reset):
    effect, scalar = pair(2, 48000, controls(), "warm")
    previous = getattr(effect, name)
    setattr(effect, name, np.nan)
    supplied = "width" if name != "width" else "dry_level"
    with pytest.raises(ValueError):
        effect(
            signal(frames=1025),
            48000,
            reset=reset,
            automation={supplied: np.full(1025, getattr(scalar, supplied))},
        )
    # Restore in both instances so the intentionally bad scalar setter's own
    # ramp retargeting is excluded from the failed process call's atomicity.
    setattr(effect, name, previous)
    setattr(scalar, name, np.nan)
    setattr(scalar, name, previous)
    future = np.zeros((2, 12000), np.float32)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


@pytest.mark.parametrize("rate,channels", [(100, 2), (1e12, 2), (48000, 3), (196, 3)])
@pytest.mark.parametrize("reset", [False, True])
def test_unsupported_format_rejected_before_tail_or_targets_change(rate, channels, reset):
    effect, scalar = pair(2, 48000, controls(), "warm")
    with pytest.raises(ValueError):
        effect(
            signal(channels, 1025), rate, reset=reset, automation={"room_size": np.full(1025, 0.9)}
        )
    assert effect.room_size == scalar.room_size
    future = np.zeros((2, 12000), np.float32)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("reset", [False, True])
def test_constant_layout_and_readonly_float64(transpose, reset):
    audio = signal().astype(np.float64)
    if transpose:
        audio = audio.T
    audio.flags.writeable = False
    values = dict(zip(NAMES, [0.8, 0.3, 0.5, 0.4, 0.7]))
    curves = {k: np.full(12000, v) for k, v in values.items()}
    for v in curves.values():
        v.flags.writeable = False
    assert_array_equal(
        Reverb()(audio, 48000, reset=reset, automation=curves),
        Reverb(**values)(audio, 48000, reset=reset),
    )


def test_empty_partial_strided_integer_and_scalar_fallback():
    effect, scalar = pair(2, 48000, controls(), "warm")
    empty = effect(np.empty((2, 0), np.float32), 48000, automation={"width": np.empty(0)})
    assert empty.shape == (2, 0)
    curve = np.ones(2050, np.int32)[::2]
    curve.flags.writeable = False
    scalar.width = 1
    assert_array_equal(
        effect(signal(frames=1025), 48000, reset=False, automation={"width": curve}),
        scalar(signal(frames=1025), 48000, reset=False),
    )
    for mapping in [None, {}]:
        assert_array_equal(Reverb()(signal(), 48000, automation=mapping), Reverb()(signal(), 48000))
    with pytest.raises(ValueError):
        effect(signal(frames=1025), 48000, automation={"freeze_mode": np.ones(1025)})


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("value", [0, 1])
def test_endpoints_keep_original_dsp(name, value):
    audio = signal()
    assert_array_equal(
        Reverb()(audio, 48000, automation={name: np.full(12000, value)}),
        Reverb(**{name: value})(audio, 48000),
    )


def test_warm_reset_preserves_original_gain_ramp_instead_of_snapping():
    effect, scalar = pair(2, 48000, controls(), "warm")
    audio = signal(frames=1025)
    scalar.dry_level = 0
    actual = effect(audio, 48000, automation={"dry_level": np.zeros(1025)})
    assert_array_equal(actual, scalar(audio, 48000))
    assert np.max(np.abs(actual[:, :100])) > 1e-3
    assert_array_equal(Reverb(dry_level=0)(audio, 48000)[:, :100], np.zeros((2, 100), np.float32))


@pytest.mark.parametrize("rate,channels", [(44100, 2), (48000, 1), (96000, 2)])
def test_format_changes_reprepare_with_original_semantics(rate, channels):
    effect, scalar = pair(2, 48000, controls(), "warm")
    curves = controls()
    audio = signal(channels, 1025, rate)
    # Original scalar prepare occurs after the first target setters and clears
    # buffers/snaps smoothers on rate/channel change.
    assert_array_equal(
        effect(audio, rate, reset=False, automation=curves),
        replay(scalar, audio, curves, rate, False),
    )


def test_native_calls_are_serialized_with_stable_fixed_targets():
    audio = signal()
    effect = Reverb()
    effect(audio, 48000)
    curves = {k: np.full(12000, getattr(effect, k)) for k in NAMES}
    expected = Reverb()(audio, 48000)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: effect(audio, 48000, automation=curves), range(6)))
    for result in results:
        assert_array_equal(result, expected)


def test_width_also_scales_mono_wet_gain_in_original_juce():
    audio = signal(1)
    narrow = Reverb(dry_level=0)(audio, 48000, automation={"width": np.zeros(12000)})
    wide = Reverb(dry_level=0)(audio, 48000, automation={"width": np.ones(12000)})
    assert np.max(np.abs(wide)) > 1e-4
    assert_array_equal(wide, 2 * narrow)
