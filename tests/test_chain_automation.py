"""Graph curves follow effect identity and valid source frames, including latency."""

import gc
import weakref

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import (
    Bitcrush,
    BrickwallLimiter,
    Chain,
    Chorus,
    Clipping,
    Compressor,
    Convolution,
    Delay,
    Distortion,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    Invert,
    LadderFilter,
    Limiter,
    LowpassFilter,
    LowShelfFilter,
    Mix,
    NoiseGate,
    PeakFilter,
    Pedalboard,
    Phaser,
    PitchShift,
    Resample,
    Reverb,
)
from pedalboard_native._internal import AddLatency

RATE = 48000


def signal(n=1025, channels=2):
    return np.random.default_rng(184).uniform(-0.8, 0.8, (channels, n)).astype(np.float32)


def curves(n):
    return np.linspace(-18, 4, n, dtype=np.float32), np.linspace(-15, -1, n, dtype=np.float32)


@pytest.mark.parametrize("container", [Pedalboard, Chain])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
@pytest.mark.parametrize("layout", ["mono", "planar", "interleaved"])
def test_serial_matches_independent_nodes(container, buffer_size, layout):
    x = signal()
    if layout == "mono":
        x = x[0]
    elif layout == "interleaved":
        x = x.T
    a, b = curves(1025)
    gain, clip = Gain(), Clipping()
    board = container([gain, clip])
    actual = board(
        x, RATE, buffer_size, automation={clip: {"threshold_db": b}, gain: {"gain_db": a}}
    )
    expected = Clipping()(
        Gain()(x, RATE, buffer_size, automation={"gain_db": a}),
        RATE,
        buffer_size,
        automation={"threshold_db": b},
    )
    assert_array_equal(actual, expected)
    assert board[0] is gain and board[1] is clip
    assert_array_equal(a, curves(1025)[0])


@pytest.mark.parametrize("latency", [1, 57, 2048])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
def test_serial_delay_does_not_advance_downstream_curve(latency, buffer_size):
    x = signal()
    a, b = curves(x.shape[1])
    gain, clip = Gain(), Clipping()
    board = Pedalboard([AddLatency(latency), gain, Chain([AddLatency(latency), clip])])
    actual = board(
        x, RATE, buffer_size, automation={gain: {"gain_db": a}, clip: {"threshold_db": b}}
    )
    expected = Clipping()(
        Gain()(x, RATE, buffer_size, automation={"gain_db": a}),
        RATE,
        buffer_size,
        automation={"threshold_db": b},
    )
    assert_array_equal(actual, expected)


def make_parallel(latency=57):
    a, b, c = Gain(), Clipping(), Gain()
    mix = Mix([Chain([AddLatency(latency), a]), b, None])
    return Pedalboard([mix, c]), (a, b, c)


@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
@pytest.mark.parametrize("latency", [1, 57, 2048])
def test_parallel_with_downstream_automation(buffer_size, latency):
    x = signal()
    a, b = curves(x.shape[1])
    board, (gain, clip, post) = make_parallel(latency)
    actual = board(
        x,
        RATE,
        buffer_size,
        automation={gain: {"gain_db": a}, clip: {"threshold_db": b}, post: {"gain_db": b}},
    )
    parallel = Gain()(x, RATE, buffer_size, automation={"gain_db": a})
    parallel += Clipping()(x, RATE, buffer_size, automation={"threshold_db": b})
    parallel += x
    expected = Gain()(parallel, RATE, buffer_size, automation={"gain_db": b})
    assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["serial", "parallel", "nested"])
@pytest.mark.parametrize("chunk", [1, 17, 257, 2048])
def test_streaming_queues_are_partition_invariant(kind, chunk):
    n = 4097
    x = signal(n, 1)[0]
    a, b = curves(n)

    def build():
        gain, clip = Gain(), Clipping()
        if kind == "serial":
            board = Pedalboard([AddLatency(301), gain, clip])
        elif kind == "parallel":
            board = Mix([Chain([AddLatency(301), gain]), clip])
        else:
            board = Pedalboard([Mix([Chain([AddLatency(301), gain]), None]), clip])
        return board, gain, clip

    board, gain, clip = build()
    expected = board(x, RATE, 31, automation={gain: {"gain_db": a}, clip: {"threshold_db": b}})
    board, gain, clip = build()
    board.reset()
    parts = []
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        parts.append(
            board(
                x[start:end],
                RATE,
                31,
                reset=False,
                automation={gain: {"gain_db": a[start:end]}, clip: {"threshold_db": b[start:end]}},
            )
        )
    # Advance real root input time with final targets to drain delayed source audio.
    parts.append(board(np.zeros(602, np.float32), RATE, 31, reset=False))
    assert_array_equal(np.concatenate(parts)[:n], expected)


CASES = [
    (Gain, "gain_db", -12, 3),
    (Clipping, "threshold_db", -18, -1),
    (Distortion, "drive_db", 0, 18),
    (Bitcrush, "bit_depth", 3, 12),
    (Compressor, "threshold_db", -24, -6),
    (NoiseGate, "threshold_db", -40, -18),
    (Limiter, "threshold_db", -18, -1),
    (LowpassFilter, "cutoff_frequency_hz", 300, 8000),
    (HighpassFilter, "cutoff_frequency_hz", 100, 2000),
    (PeakFilter, "gain_db", -9, 9),
    (LowShelfFilter, "gain_db", -9, 9),
    (HighShelfFilter, "gain_db", -9, 9),
    (LadderFilter, "cutoff_hz", 300, 8000),
    (Reverb, "wet_level", 0.1, 0.8),
    (Delay, "mix", 0.1, 0.8),
    (Chorus, "mix", 0.1, 0.8),
    (Phaser, "mix", 0.1, 0.8),
    (lambda: Convolution(np.array([1, 0.3, -0.2], np.float32), sample_rate=RATE), "mix", 0.1, 0.8),
    (BrickwallLimiter, "ceiling_db", -12, -1),
    (PitchShift, "semitones", -3, 7),
    (Resample, "target_sample_rate", 4000, 24000),
]


@pytest.mark.parametrize("factory,name,lo,hi", CASES)
@pytest.mark.parametrize("buffer_size", [31, 8192])
def test_all_supported_builtins_reuse_individual_scheduler(factory, name, lo, hi, buffer_size):
    x = signal()
    curve = np.linspace(lo, hi, x.shape[1], dtype=np.float32)
    effect = factory()
    actual = Pedalboard([AddLatency(57), effect])(
        x, RATE, buffer_size, automation={effect: {name: curve}}
    )
    expected = factory()(x, RATE, buffer_size, automation={name: curve})
    if name == "mix" and isinstance(effect, Convolution):
        # Different FFT block partitions may round at float32 precision.
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    else:
        assert_array_equal(actual, expected)


@pytest.mark.parametrize("factory,name,lo,hi", [CASES[i] for i in [13, 17, 18, 19, 20]])
def test_buffered_and_tail_effects_before_another_control(factory, name, lo, hi):
    n = 5001
    x = signal(n, 1)[0]
    curve = np.linspace(lo, hi, n, dtype=np.float32)
    target = np.linspace(-18, 0, n, dtype=np.float32)
    effect, gain = factory(), Gain()
    actual = Pedalboard([effect, gain])(
        x, RATE, 257, automation={effect: {name: curve}, gain: {"gain_db": target}}
    )
    expected = Gain()(
        factory()(x, RATE, 257, automation={name: curve}),
        RATE,
        257,
        automation={"gain_db": target},
    )
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "bad", ["foreign", "position", "unknown", "nan", "length", "list", "unsupported", "container"]
)
def test_invalid_mapping_preserves_warm_graph(bad):
    a, b = Gain(), Gain()
    gate_a, gate_b = Reverb(), Reverb()
    extra = Invert()
    left, right = Pedalboard([a, gate_a, extra]), Pedalboard([b, gate_b, Invert()])
    x = signal()
    initial = np.linspace(-6, 0, x.shape[1])
    for board, gain in [(left, a), (right, b)]:
        board(x, RATE, automation={gain: {"gain_db": initial}})
    mapping = {a: {"gain_db": np.full(x.shape[1], -12.0)}}
    if bad == "foreign":
        mapping[Gain()] = {"gain_db": initial}
    elif bad == "position":
        mapping[0] = {"gain_db": initial}
    elif bad == "unknown":
        mapping[a]["missing"] = initial
    elif bad == "nan":
        mapping[gate_a] = {"wet_level": np.full(x.shape[1], np.nan)}
    elif bad == "length":
        mapping[gate_a] = {"wet_level": initial[:-1]}
    elif bad == "list":
        mapping[gate_a] = {"wet_level": [0.2] * x.shape[1]}
    elif bad == "unsupported":
        mapping[extra] = {"mix": initial}
    else:
        mapping[left] = {"gain_db": initial}
    with pytest.raises((ValueError, TypeError)):
        left(x, RATE, automation=mapping)
    assert a.gain_db == b.gain_db
    assert_array_equal(left(x, RATE, reset=False), right(x, RATE, reset=False))


def test_joint_native_validation_does_not_latch_other_effects():
    x = signal()
    for factory, name, lo, hi in [CASES[i] for i in [18, 19, 20]]:
        effect, reference = factory(), factory()
        reverb = Reverb()
        reverb.freeze_mode = float("nan")
        board = Pedalboard([effect, reverb])
        with pytest.raises(ValueError):
            board(
                x,
                RATE,
                automation={
                    effect: {name: np.full(1025, lo)},
                    reverb: {"wet_level": np.full(1025, 0.3)},
                },
            )
        # Invalid graph validation must not silently enable clocked Resample,
        # pitch fine scheduling or output-aligned limiter automation.
        assert_array_equal(effect(x, RATE), reference(x, RATE))


def test_identity_follows_reordering_and_distinguishes_same_class():
    x = signal()
    a, b = Gain(), Gain()
    board = Pedalboard([a, b])
    target_a, target_b = curves(1025)
    expected = board(x, RATE, automation={a: {"gain_db": target_a}, b: {"gain_db": target_b}})
    board[0], board[1] = b, a
    actual = board(x, RATE, automation={a: {"gain_db": target_a}, b: {"gain_db": target_b}})
    # Gain multiplications commute to floating precision, and each instance
    # retains its own final target regardless of its new index.
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
    assert a.gain_db == float(target_a[-1])
    assert b.gain_db == float(target_b[-1])


def test_stream_boundaries_and_explicit_reset():
    x = signal()
    gain, clip = Gain(), Clipping()
    board = Pedalboard([gain, clip])
    board(x, RATE)
    with pytest.raises(ValueError, match="reset"):
        board(x, RATE, reset=False, automation={gain: {"gain_db": np.zeros(1025)}})
    board.reset()
    board(x, RATE, reset=False, automation={gain: {"gain_db": np.zeros(1025)}})
    with pytest.raises(ValueError, match="lanes"):
        board(x, RATE, reset=False, automation={clip: {"threshold_db": np.zeros(1025)}})
    with pytest.raises(ValueError, match="format"):
        board(x, 44100, reset=False)
    board.append(Gain())
    with pytest.raises(ValueError, match="topology"):
        board(x, RATE, reset=False)
    board(x, RATE, automation={clip: {"threshold_db": np.zeros(1025)}})


@pytest.mark.parametrize("cycle", [False, True])
def test_duplicate_and_cycle_errors_are_bounded(cycle):
    gain = Gain()
    board = Pedalboard([gain])
    board.append(board if cycle else gain)
    with pytest.raises(ValueError, match="cycles or duplicate"):
        board(signal(), RATE, automation={gain: {"gain_db": np.zeros(1025)}})
    if cycle:
        del board[1]


def test_session_has_no_root_reference_cycle():
    gain = Gain()
    board = Pedalboard([Chain([gain])])
    ref = weakref.ref(board)
    board(signal(), RATE, automation={gain: {"gain_db": np.zeros(1025)}})
    del board
    gc.collect()
    assert ref() is None


def test_mix_list_edits_resize_internal_branch_storage():
    x = signal()
    mix = Mix([])
    mix(x, RATE)
    mix.append(Gain(0))
    assert_array_equal(mix(x, RATE), x)
    mix.append(None)
    assert_array_equal(mix(x, RATE), x + x)
    del mix[0]
    assert_array_equal(mix(x, RATE), x)


def test_omitted_parameter_lane_holds_last_scheduled_target():
    x = signal(1537, 1)[0]

    def run(omit):
        compressor = Compressor()
        gain = Gain()
        board = Pedalboard([AddLatency(701), compressor, gain])
        board.reset()
        first = board(
            x[:513],
            RATE,
            31,
            reset=False,
            automation={
                compressor: {
                    "threshold_db": np.linspace(-30, -12, 513),
                    "ratio": np.linspace(1, 5, 513),
                },
                gain: {"gain_db": np.linspace(-12, -6, 513)},
            },
        )
        parameters = {"threshold_db": np.linspace(-12, -24, 1024)}
        if not omit:
            parameters["ratio"] = np.full(1024, 5)
        second = board(x[513:], RATE, 31, reset=False, automation={compressor: parameters})
        tail = board(np.zeros(1402, np.float32), RATE, 31, reset=False)
        return np.concatenate([first, second, tail])[: len(x)]

    assert_array_equal(run(True), run(False))


def test_pending_curves_are_owned_and_empty_input_does_not_flush():
    x = signal(1025, 1)[0]

    def run(mutate):
        gain = Gain()
        board = Pedalboard([AddLatency(701), gain])
        board.reset()
        target = np.linspace(-18, 0, len(x), dtype=np.float32)
        first = board(x[:513], RATE, 31, reset=False, automation={gain: {"gain_db": target[:513]}})
        assert first.size == 0
        assert board(np.zeros(0, np.float32), RATE, reset=False).size == 0
        if mutate:
            target[:513] = 60
        second = board(x[513:], RATE, 31, reset=False, automation={gain: {"gain_db": target[513:]}})
        tail = board(np.zeros(1402, np.float32), RATE, 31, reset=False)
        return np.concatenate([first, second, tail])[: len(x)]

    assert_array_equal(run(True), run(False))


def test_native_preflight_failure_preserves_pending_stream():
    x = signal(1537, 1)[0]

    def build():
        gain, reverb = Gain(), Reverb()
        board = Pedalboard([AddLatency(701), gain, reverb])
        board.reset()
        board(
            x[:513],
            RATE,
            31,
            reset=False,
            automation={
                gain: {"gain_db": np.linspace(-18, 0, 513)},
                reverb: {"wet_level": np.full(513, 0.3)},
            },
        )
        return board, gain, reverb

    left, gain, reverb = build()
    right, _, _ = build()
    reverb.freeze_mode = float("nan")
    with pytest.raises(ValueError):
        left(x[513:], RATE, 31, reset=False, automation={gain: {"gain_db": np.full(1024, -48)}})
    reverb.freeze_mode = 0
    assert_array_equal(left(x[513:], RATE, 31, reset=False), right(x[513:], RATE, 31, reset=False))


def test_processing_routes_are_thread_local_and_locked():
    from concurrent.futures import ThreadPoolExecutor

    x = signal()

    def process_one(db):
        gain = Gain()
        board = Pedalboard([Mix([gain, None])])
        target = np.full(1025, db, np.float32)
        expected = Gain()(x, RATE, automation={"gain_db": target}) + x
        for _ in range(5):
            assert_array_equal(board(x, RATE, 31, automation={gain: {"gain_db": target}}), expected)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(process_one, db) for db in [-18, -6, 0, 6]]
        for future in futures:
            future.result(timeout=15)


def test_changing_buffer_size_keeps_delayed_stream_memory():
    x = signal(4097, 1)[0]
    target = np.linspace(-18, 0, len(x), dtype=np.float32)
    gain = Gain()
    board = Mix([Chain([AddLatency(701), gain]), None])
    expected = board(x, RATE, 31, automation={gain: {"gain_db": target}})
    board.reset()
    parts = []
    start = 0
    for count, size in [(1, 31), (17, 1), (2048, 8192), (2031, 257)]:
        end = start + count
        parts.append(
            board(
                x[start:end],
                RATE,
                size,
                reset=False,
                automation={gain: {"gain_db": target[start:end]}},
            )
        )
        start = end
    parts.append(board(np.zeros(1402, np.float32), RATE, 8192, reset=False))
    assert_array_equal(np.concatenate(parts)[: len(x)], expected)


@pytest.mark.parametrize("factory,name,lo,hi", [CASES[i] for i in [18, 19, 20]])
@pytest.mark.parametrize("parallel", [False, True])
def test_real_buffered_effect_streams_keep_source_controls(factory, name, lo, hi, parallel):
    n = 5001
    x = signal(n, 1)[0]
    target = np.linspace(lo, hi, n, dtype=np.float32)
    gain_target = np.linspace(-18, 0, n, dtype=np.float32)

    def build():
        effect, gain = factory(), Gain()
        before = Mix([effect, None]) if parallel else effect
        board = Pedalboard([before, gain])
        return board, effect, gain

    board, effect, gain = build()
    expected = board(
        x, RATE, 257, automation={effect: {name: target}, gain: {"gain_db": gain_target}}
    )
    board, effect, gain = build()
    board.reset()
    parts = []
    for start in range(0, n, 513):
        end = min(n, start + 513)
        parts.append(
            board(
                x[start:end],
                RATE,
                257,
                reset=False,
                automation={
                    effect: {name: target[start:end]},
                    gain: {"gain_db": gain_target[start:end]},
                },
            )
        )
    parts.append(board(np.zeros(RATE * 2, np.float32), RATE, 257, reset=False))
    assert_array_equal(np.concatenate(parts)[:n], expected)


def test_concurrent_reset_and_render_share_tree_locks():
    from concurrent.futures import ThreadPoolExecutor

    gain = Gain()
    board = Pedalboard([Mix([Chain([AddLatency(57), gain]), None])])
    x = signal()
    target = np.linspace(-18, 0, 1025, dtype=np.float32)

    expected = Gain()(x, RATE, 31, automation={"gain_db": target}) + x

    def render():
        for _ in range(20):
            result = board(x, RATE, 31, automation={gain: {"gain_db": target}})
            assert_array_equal(result, expected)

    def reset():
        for _ in range(20):
            board.reset()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(render), pool.submit(reset)]
        for future in futures:
            future.result(timeout=20)


def test_resetting_root_clears_nested_sessions_and_restores_scalar_path():
    x = signal()
    gain = Gain()
    nested = Chain([gain])
    board = Pedalboard([nested])
    nested(x, RATE, automation={gain: {"gain_db": np.zeros(1025)}})
    board.reset()
    gain.gain_db = -12
    assert_array_equal(nested(x, RATE, reset=False), Gain(-12)(x, RATE))


def test_dropping_automation_with_reset_true_returns_to_scalar_processing():
    x = signal()
    gain = Gain()
    board = Pedalboard([gain])
    board(x, RATE, automation={gain: {"gain_db": np.zeros(1025)}})
    gain.gain_db = -12
    assert_array_equal(board(x, RATE), Gain(-12)(x, RATE))
    gain.gain_db = -6
    assert_array_equal(board(x, RATE, reset=False), Gain(-6)(x, RATE))


@pytest.mark.parametrize("factory,name,lo,hi", [CASES[i] for i in [13, 14, 17]])
def test_explicit_silence_keeps_effect_tails_and_held_controls(factory, name, lo, hi):
    n, tail = 513, 8192
    x = np.zeros(n + tail, np.float32)
    x[0] = 0.8
    x[n - 1] = 0.4
    target = np.r_[np.linspace(lo, hi, n, dtype=np.float32), np.full(tail, hi, np.float32)]

    def build():
        effect = factory()
        if isinstance(effect, Delay):
            effect.delay_seconds = 0.01
            effect.feedback = 0.2
        return Pedalboard([effect]), effect

    board, effect = build()
    expected = board(x, RATE, 257, automation={effect: {name: target}})
    board, effect = build()
    board.reset()
    first = board(x[:n], RATE, 257, reset=False, automation={effect: {name: target[:n]}})
    middle = board(x[n : n + tail // 2], RATE, 257, reset=False)
    last = board(x[n + tail // 2 :], RATE, 257, reset=False)
    actual = np.concatenate([first, middle, last])
    if isinstance(effect, Convolution):
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    else:
        assert_array_equal(actual, expected)
    assert np.any(actual[n:] != 0)


def test_large_requested_buffer_keeps_internal_capacity_bounded():
    gain = Gain()
    board = Mix([gain, None])
    x = np.array([0.125], np.float32)
    output = board(
        x, RATE, np.iinfo(np.int32).max, automation={gain: {"gain_db": np.zeros(1, np.float32)}}
    )
    assert_array_equal(output, x + x)


@pytest.mark.parametrize("parent_automated", [False, True])
def test_parent_render_reset_discards_old_nested_automation(parent_automated):
    x = signal()
    gain, post = Gain(), Gain(0)
    nested = Chain([gain])
    nested(x, RATE, automation={gain: {"gain_db": np.zeros(1025)}})
    board = Pedalboard([nested, post])
    mapping = {post: {"gain_db": np.zeros(1025)}} if parent_automated else None
    board(x, RATE, reset=True, automation=mapping)
    # The parent reset already restarted the child. Its next scalar call must
    # not resurrect automation that belonged to the child's earlier session.
    gain.gain_db = -12
    assert_array_equal(nested(x, RATE, reset=False), Gain(-12)(x, RATE))
