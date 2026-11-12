"""Output-clock controls retain lookahead and detector state across calls."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import BrickwallLimiter, Chain


def signal(n=4097, channels=2, rate=48000):
    t = np.arange(n) / rate
    return np.stack(
        [
            (0.85 * np.sin(2 * np.pi * 173 * t) + 0.4 * np.cos(2 * np.pi * 3337 * t)) * 0.63**c
            for c in range(channels)
        ]
    ).astype(np.float32)


def latency(effect, channels, rate):
    n = 32768
    result = effect(np.zeros((channels, n), np.float32), rate, buffer_size=1024, reset=False)
    effect.reset()
    return n - result.shape[1]


@pytest.mark.parametrize("tp", [False, True])
@pytest.mark.parametrize("rate", [44100, 48000, 96000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
def test_output_clock_matches_delayed_scalar_reference(tp, rate, channels, warm, buffer_size):
    a, b = [BrickwallLimiter(ceiling_db=-1, release_ms=20, true_peak=tp) for _ in range(2)]
    delay = latency(a, channels, rate)
    assert delay == latency(b, channels, rate)
    if warm:
        for effect in (a, b):
            effect(signal(6001, channels, rate), rate, buffer_size=1024, reset=False)
    x = signal(2049, channels, rate)
    curves = {
        "ceiling_db": np.linspace(-1, -9, x.shape[1], dtype=np.float32),
        "release_ms": np.linspace(20, 140, x.shape[1], dtype=np.float32),
    }
    actual = a(x, rate, buffer_size=buffer_size, reset=False, automation=curves)
    future = signal(4096, channels, rate)
    actual = np.concatenate(
        [
            actual,
            a(future, rate, buffer_size=8192, reset=False),
            a(np.zeros((channels, delay), np.float32), rate, reset=False),
        ],
        axis=1,
    )
    host = Chain([b])
    host(np.zeros((channels, 32768), np.float32), rate, buffer_size=1024, reset=False)
    host.reset()
    if warm:
        host(signal(6001, channels, rate), rate, buffer_size=1024, reset=False)
    feed = np.concatenate([x, future, np.zeros((channels, delay), np.float32)], axis=1)
    parts = []
    for i, end in [(0, delay)] + [(j, j + 1) for j in range(delay, feed.shape[1])]:
        index = np.clip(i - delay, 0, x.shape[1] - 1)
        b.ceiling_db = float(curves["ceiling_db"][index])
        b.release_ms = float(curves["release_ms"][index])
        parts.append(host(feed[:, i:end], rate, buffer_size=1024, reset=False))
    assert_array_equal(actual, np.concatenate(parts, axis=1))
    assert a.ceiling_db == float(curves["ceiling_db"][-1])
    assert a.release_ms == float(curves["release_ms"][-1])


@pytest.mark.parametrize("tp", [False, True])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
def test_actual_step_is_at_compensated_output_time(tp, buffer_size):
    n, step, rate = 9600, 4800, 48000
    curve = np.where(np.arange(n) < step, 0, -6).astype(np.float32)
    output = BrickwallLimiter(true_peak=tp)(
        np.full(n, 0.8, np.float32), rate, buffer_size=buffer_size, automation={"ceiling_db": curve}
    )
    assert output.shape == (n,)
    assert np.flatnonzero(output < 0.6)[0] == step
    assert np.all(np.abs(output) <= np.power(np.float32(10), curve / 20) + 1e-7)


@pytest.mark.parametrize("tp", [False, True])
def test_chunked_curves_and_tail_equal_one_shot(tp):
    x = signal(6001)
    curve = np.linspace(-1, -12, x.shape[1], dtype=np.float32)
    a, b = BrickwallLimiter(true_peak=tp), BrickwallLimiter(true_peak=tp)
    expected = a(x, 48000, automation={"ceiling_db": curve})
    delay = latency(b, 2, 48000)
    parts = []
    for start, end in [(0, 7), (7, 300), (300, 2049), (2049, 6001)]:
        parts.append(
            b(
                x[:, start:end],
                48000,
                buffer_size=8192,
                reset=False,
                automation={"ceiling_db": curve[start:end]},
            )
        )
    parts.append(b(np.zeros((2, delay), np.float32), 48000, reset=False))
    assert_array_equal(np.concatenate(parts, axis=1), expected)


@pytest.mark.parametrize("tp", [False, True])
@pytest.mark.parametrize(
    "key,value",
    [
        ("ceiling_db", np.nan),
        ("ceiling_db", 10000),
        ("ceiling_db", -10000),
        ("release_ms", 0),
        ("release_ms", 1e30),
        ("lookahead_ms", 4),
        ("true_peak", 1),
    ],
)
def test_invalid_curves_do_not_mutate_warm_state(tp, key, value):
    a, b = BrickwallLimiter(true_peak=tp), BrickwallLimiter(true_peak=tp)
    x = signal()
    for effect in (a, b):
        effect(x, 48000)
    with pytest.raises((ValueError, TypeError, OverflowError)):
        a(x, 48000, automation={key: np.full(x.shape[1], value)})
    assert a.ceiling_db == b.ceiling_db
    assert a.release_ms == b.release_ms
    assert_array_equal(a(x, 48000, reset=False), b(x, 48000, reset=False))


@pytest.mark.parametrize("tp", [False, True])
def test_constant_curves_and_scalar_reset_paths(tp):
    x = signal()
    a, b = BrickwallLimiter(-3, 80, true_peak=tp), BrickwallLimiter(-3, 80, true_peak=tp)
    assert_array_equal(
        a(
            x,
            48000,
            automation={
                "ceiling_db": np.full(x.shape[1], -3),
                "release_ms": np.full(x.shape[1], 80),
            },
        ),
        b(x, 48000),
    )
    assert_array_equal(a(x, 48000, automation={}), b(x, 48000))
    assert_array_equal(a(x, 48000, automation=None), b(x, 48000))
