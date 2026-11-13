"""Source-clock rate curves preserve duration, partitions and interpolation modes."""

from itertools import pairwise

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import Resample

QUALITIES = list(Resample.Quality.__members__.values())


def signal(frames=2049, channels=2, rate=48000):
    t = np.arange(frames) / rate
    return np.stack(
        [
            (0.3 * np.sin(2 * np.pi * 173 * t) + 0.1 * np.cos(2 * np.pi * 3337 * t)) * 0.63**c
            for c in range(channels)
        ]
    ).astype(np.float32)


def close(actual, expected):
    d = actual.astype(np.float64) - expected
    assert np.max(np.abs(d)) <= 2e-5
    assert np.sqrt(np.mean(d * d)) <= max(
        1e-12, np.sqrt(np.mean(expected.astype(np.float64) ** 2)) * 1e-5
    )


@pytest.mark.parametrize("quality", QUALITIES, ids=lambda q: q.name)
@pytest.mark.parametrize("rate", [22050, 48000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("target_kind", ["low", "noninteger", "native"])
def test_constant_curves_retain_scalar_sound(quality, rate, channels, target_kind):
    target = {"low": rate / 12, "noninteger": rate * 0.271828, "native": rate}[target_kind]
    x = signal(channels=channels, rate=rate)
    expected = Resample(target, quality)(x, rate, buffer_size=128)
    curve = np.full(x.shape[1], target, np.float32)
    actual = Resample(quality=quality)(
        x, rate, buffer_size=31, automation={"target_sample_rate": curve}
    )
    assert actual.shape == x.shape
    assert np.isfinite(actual).all()
    close(actual, expected)


@pytest.mark.parametrize("quality", QUALITIES, ids=lambda q: q.name)
@pytest.mark.parametrize("kind", ["down-step", "up-step", "ramp", "constant"])
def test_buffer_partition_invariance_and_final_getter(quality, kind):
    x = signal(4097)
    n = x.shape[1]
    curve = {
        "down-step": np.where(np.arange(n) < 2000, 24000, 4000),
        "up-step": np.where(np.arange(n) < 2000, 4000, 24000),
        "ramp": np.linspace(24000, 4000, n),
        "constant": np.full(n, 12345.67),
    }[kind].astype(np.float32)
    original = x.copy()
    saved = curve.copy()
    reference = None
    for buffer in [1, 31, 128, 8192]:
        effect = Resample(quality=quality)
        actual = effect(x, 48000, buffer_size=buffer, automation={"target_sample_rate": curve})
        assert effect.target_sample_rate == float(curve[-1])
        assert actual.shape == x.shape
        if reference is None:
            reference = actual
        assert_array_equal(actual, reference)
    assert_array_equal(x, original)
    assert_array_equal(curve, saved)


@pytest.mark.parametrize("quality", QUALITIES, ids=lambda q: q.name)
def test_streaming_with_scalar_hold_drains_exact_latency_and_survives_growth(quality):
    x = signal(4097)
    curve = np.linspace(24000, 4000, 4097, dtype=np.float32)
    whole = Resample(quality=quality)(x, 48000, automation={"target_sample_rate": curve})
    effect = Resample(quality=quality)
    parts = []
    for start, end in pairwise([0, 7, 300, 2049, 4097]):
        parts.append(
            effect(
                x[:, start:end],
                48000,
                buffer_size=8192,
                reset=False,
                automation={"target_sample_rate": curve[start:end]},
            )
        )
    # Sum of returned deficits gives actual fixed latency, even for short input.
    # Probe the bound with a separate constant-rate stream, not a quality table.
    probe = Resample(quality=quality)
    count = 65536
    latency = (
        count
        - probe(
            np.zeros((2, count), np.float32),
            48000,
            reset=False,
            automation={"target_sample_rate": np.full(count, 24000, np.float32)},
        ).shape[1]
    )
    parts.append(effect(np.zeros((2, latency), np.float32), 48000, buffer_size=31, reset=False))
    actual = np.concatenate(parts, axis=1)
    assert actual.shape == whole.shape
    assert_array_equal(actual, whole)


def test_zero_order_hold_matches_independent_integrated_clock_oracle():
    n = 4097
    rate = 48000
    curve = np.where(np.arange(n) < 1501, 6000, 24000).astype(np.float32)
    x = np.random.default_rng(7).uniform(-0.5, 0.5, n).astype(np.float32)
    cycles = np.r_[0, np.cumsum(curve.astype(np.float64) / rate)]
    event_times = np.interp(np.arange(int(np.floor(cycles[-1])) + 1), cycles, np.arange(n + 1))
    indices = np.floor(event_times[np.floor(cycles[:-1]).astype(int)]).astype(int)
    expected = x[indices]
    actual = Resample(quality=Resample.Quality.ZeroOrderHold)(
        x, rate, automation={"target_sample_rate": curve}
    )
    assert_array_equal(actual, expected)


@pytest.mark.parametrize("quality", QUALITIES, ids=lambda q: q.name)
def test_repeated_reset_is_reproducible(quality):
    x = signal(257)
    curve = np.linspace(24000, 4000, 257, dtype=np.float32)
    effect = Resample(quality=quality)
    a = effect(x, 48000, automation={"target_sample_rate": curve})
    b = effect(x, 48000, automation={"target_sample_rate": curve})
    assert_array_equal(a, b)


@pytest.mark.parametrize("value", [0, 749, 48001, np.nan, np.inf])
def test_invalid_curve_preserves_clocked_state(value):
    a, b = Resample(), Resample()
    x = signal(2049)
    for effect in [a, b]:
        effect(x, 48000, reset=False, automation={"target_sample_rate": np.full(2049, 12000)})
    with pytest.raises((ValueError, TypeError)):
        a(x, 48000, automation={"target_sample_rate": np.full(2049, value)})
    assert a.target_sample_rate == b.target_sample_rate
    assert_array_equal(a(x, 48000, reset=False), b(x, 48000, reset=False))


def test_warm_scalar_entry_requires_explicit_reset_without_mutating_history():
    a, b = Resample(12000), Resample(12000)
    x = signal()
    for effect in [a, b]:
        effect(x, 48000)
    with pytest.raises(ValueError, match="reset=True"):
        a(x, 48000, reset=False, automation={"target_sample_rate": np.full(2049, 4000)})
    assert a.target_sample_rate == 12000
    assert_array_equal(a(x, 48000, reset=False), b(x, 48000, reset=False))
    close(
        a(x, 48000, automation={"target_sample_rate": np.full(2049, 4000)}),
        Resample(4000)(x, 48000),
    )


def test_readonly_strided_float64_controls_and_transposed_audio():
    x = signal().T.astype(np.float64)
    x.flags.writeable = False
    curve = np.linspace(4000, 24000, 4098)[::2]
    curve.flags.writeable = False
    actual = Resample()(x, 48000, automation={"target_sample_rate": curve})
    expected = Resample()(x.T, 48000, automation={"target_sample_rate": curve.copy()})
    assert_array_equal(actual, expected.T)


@pytest.mark.parametrize("quality", QUALITIES, ids=lambda q: q.name)
@pytest.mark.parametrize("profile", ["alternating", "random"])
def test_rate_bounds_stress_and_four_channel_timeline(quality, profile):
    n = 8193
    rng = np.random.default_rng(33)
    curve = (
        np.where(np.arange(n) % 2, 48000, 750)
        if profile == "alternating"
        else rng.uniform(750, 48000, n)
    ).astype(np.float32)
    x = signal(n, 4)
    a = Resample(quality=quality)(
        x, 48000, buffer_size=31, automation={"target_sample_rate": curve}
    )
    b = Resample(quality=quality)(
        x, 48000, buffer_size=8192, automation={"target_sample_rate": curve}
    )
    assert np.isfinite(a).all()
    assert_array_equal(a, b)


def test_manual_rate_outside_clocked_range_fails_before_processing():
    effect = Resample()
    x = signal()
    effect(x, 48000, automation={"target_sample_rate": np.full(2049, 12000)})
    effect.target_sample_rate = 96000
    with pytest.raises(ValueError, match="Clocked Resample"):
        effect(x, 48000, reset=False)
    # Scalar restart retains the upstream upsampling range.
    assert np.isfinite(effect(x, 48000, reset=True)).all()


def test_empty_audio_and_empty_dict_do_not_activate_curve_timeline():
    effect = Resample(12000)
    effect(np.empty((2, 0), np.float32), 48000, automation={"target_sample_rate": np.empty(0)})
    assert effect.target_sample_rate == 12000
    x = signal()
    assert_array_equal(effect(x, 48000, automation={}), Resample(12000)(x, 48000))
