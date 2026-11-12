"""Ladder targets must retain native smoothers and nonlinear state in every mode."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import LadderFilter

MODES = [
    LadderFilter.LPF12,
    LadderFilter.LPF24,
    LadderFilter.HPF12,
    LadderFilter.HPF24,
    LadderFilter.BPF12,
    LadderFilter.BPF24,
]


def signal(channels=2, frames=513):
    t = np.arange(frames) / 48000
    wave = 0.4 * np.sin(2 * np.pi * 173 * t) + 0.15 * np.cos(2 * np.pi * 3337 * t)
    return np.stack([wave * 0.63**ch for ch in range(channels)]).astype(np.float32)


def controls(frames=513):
    return {
        "cutoff_hz": np.geomspace(300, 6000, frames).astype(np.float32),
        "resonance": np.linspace(0.1, 0.85, frames, dtype=np.float32),
        "drive": np.linspace(1, 3, frames, dtype=np.float32),
    }


def initial(curves):
    return {k: float(v[0]) for k, v in curves.items()}


def reference(mode, audio, curves, rate, warm=None):
    effect = LadderFilter(mode=mode, **initial(curves))
    effect(np.zeros((audio.shape[0], 4096), np.float32) if warm is None else warm, rate)
    parts = []
    for i in range(audio.shape[1]):
        for name, curve in curves.items():
            setattr(effect, name, float(curve[i]))
        parts.append(
            effect(audio[:, i : i + 1], rate, buffer_size=1, reset=warm is None and i == 0)
        )
    return np.concatenate(parts, axis=1), effect


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
@pytest.mark.parametrize("warm", [False, True])
def test_targets_smoothers_and_future_match_scalar(mode, rate, channels, buffer_size, warm):
    audio, curves = signal(channels), controls()
    prefix = signal(channels, 4096) if warm else None
    expected, scalar = reference(mode, audio, curves, rate, prefix)
    effect = LadderFilter(mode=mode, **initial(curves))
    if prefix is not None:
        effect(prefix, rate)
    saved_audio, saved_curves = audio.copy(), {k: v.copy() for k, v in curves.items()}
    actual = effect(audio, rate, buffer_size=buffer_size, reset=not warm, automation=curves)
    assert_array_equal(actual, expected)
    assert_array_equal(audio, saved_audio)
    assert effect.mode == mode
    for name, curve in curves.items():
        assert getattr(effect, name) == float(curve[-1])
        assert_array_equal(curve, saved_curves[name])
    future = signal(channels, 4096)
    assert_array_equal(effect(future, rate, reset=False), scalar(future, rate, reset=False))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("reset", [False, True])
def test_cold_constants_layout_and_float64(mode, transpose, reset):
    audio = signal().astype(np.float64)
    if transpose:
        audio = audio.T
    audio.flags.writeable = False
    values = {"cutoff_hz": 1700.0, "resonance": 0.6, "drive": 2.0}
    curves = {k: np.full(513, v) for k, v in values.items()}
    for v in curves.values():
        v.flags.writeable = False
    assert_array_equal(
        LadderFilter(mode=mode)(audio, 48000, reset=reset, automation=curves),
        LadderFilter(mode=mode, **values)(audio, 48000, reset=reset),
    )


@pytest.mark.parametrize("mode", MODES)
def test_jump_fragmentation_and_block_growth(mode):
    audio, curves = signal(frames=1024), controls(1024)
    curves = {
        k: np.where(np.arange(1024) < 113, v[0], v[-1]).astype(np.float32)
        for k, v in curves.items()
    }
    prefix = signal(frames=32)
    expected, scalar = reference(mode, audio, curves, 48000, prefix)
    effect = LadderFilter(mode=mode, **initial(curves))
    effect(prefix, 48000)
    parts, start = [], 0
    for end, size in [(1, 1), (3, 2), (20, 32), (21, 64), (112, 128), (1024, 8192)]:
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
    future = signal(frames=4096)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))
    assert_array_equal(
        LadderFilter(mode=mode)(audio, 48000, automation=curves),
        LadderFilter(mode=mode)(audio, 48000, automation=dict(reversed(list(curves.items())))),
    )


@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize(
    "name,bad",
    [
        ("cutoff_hz", 0),
        ("cutoff_hz", np.nan),
        ("cutoff_hz", 1e-100),
        ("cutoff_hz", np.inf),
        ("resonance", -1e-6),
        ("resonance", 1.01),
        ("resonance", np.nan),
        ("drive", 0.99),
        ("drive", 1e100),
        ("drive", np.inf),
    ],
)
def test_invalid_late_values_do_not_reset_or_mutate(name, bad, reset):
    values = {"cutoff_hz": 1000.0, "resonance": 0.2, "drive": 2.0}
    effect, scalar = LadderFilter(**values), LadderFilter(**values)
    effect(signal(frames=4096), 48000)
    scalar(signal(frames=4096), 48000)
    curves = {k: v.astype(np.float64) for k, v in controls().items()}
    curves[name][-1] = bad
    with pytest.raises(ValueError):
        effect(signal(), 48000, reset=reset, automation=curves)
    assert {k: getattr(effect, k) for k in values} == {k: getattr(scalar, k) for k in values}
    assert_array_equal(
        effect(signal(frames=4096), 48000, reset=False),
        scalar(signal(frames=4096), 48000, reset=False),
    )


@pytest.mark.parametrize("name", ["cutoff_hz", "resonance", "drive"])
@pytest.mark.parametrize("reset", [False, True])
def test_preflight_checks_omitted_scalar_controls(name, reset):
    values = {"cutoff_hz": 1000.0, "resonance": 0.2, "drive": 2.0}
    effect, scalar = LadderFilter(**values), LadderFilter(**values)
    effect(signal(frames=4096), 48000)
    scalar(signal(frames=4096), 48000)
    setattr(effect, name, np.nan)
    supplied = "drive" if name != "drive" else "cutoff_hz"
    with pytest.raises(ValueError):
        effect(signal(), 48000, reset=reset, automation={supplied: np.full(513, values[supplied])})
    setattr(effect, name, values[name])
    assert_array_equal(
        effect(signal(frames=4096), 48000, reset=False),
        scalar(signal(frames=4096), 48000, reset=False),
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("resonance", [0.0, 1.0])
@pytest.mark.parametrize("drive", [1.0, 8.0])
def test_resonance_endpoints_and_saturation_match(mode, resonance, drive):
    values = {"cutoff_hz": 3000.0, "resonance": resonance, "drive": drive}
    audio = signal(frames=4096)
    actual = LadderFilter(mode=mode)(
        audio, 48000, automation={k: np.full(4096, v) for k, v in values.items()}
    )
    expected = LadderFilter(mode=mode, **values)(audio, 48000)
    assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize(
    "name,value",
    [("cutoff_hz", 1e-20), ("cutoff_hz", 30000.0), ("drive", float(np.finfo(np.float32).max))],
)
def test_retains_original_finite_domain_without_nyquist_clamp(name, value):
    values = {"cutoff_hz": 1000.0, "resonance": 0.2, "drive": 1.0}
    values[name] = value
    audio = signal()
    result = LadderFilter()(
        audio, 48000, automation={k: np.full(513, v) for k, v in values.items()}
    )
    assert_array_equal(result, LadderFilter(**values)(audio, 48000))
    assert np.isfinite(result).all()


@pytest.mark.parametrize("name,value", [("cutoff_hz", 6000.0), ("resonance", 0.85)])
def test_warm_targets_keep_original_smoothing_instead_of_snapping(name, value):
    values = {"cutoff_hz": 300.0, "resonance": 0.1, "drive": 1.0}
    effect, scalar = LadderFilter(**values), LadderFilter(**values)
    prefix = signal(frames=4096)
    effect(prefix, 48000)
    scalar(prefix, 48000)
    audio = signal(frames=4096)
    setattr(scalar, name, value)
    actual = effect(audio, 48000, reset=False, automation={name: np.full(4096, value)})
    assert_array_equal(actual, scalar(audio, 48000, reset=False))
    # reset=True snaps the existing smoother to its target and clears memory;
    # the warm path must not accidentally behave like that reset path.
    fresh_values = {**values, name: value}
    snapped = LadderFilter(**fresh_values)(audio, 48000)
    assert np.max(np.abs(actual[:, :64] - snapped[:, :64])) > 1e-4


@pytest.mark.parametrize("rate,channels", [(44100, 2), (48000, 1), (96000, 2)])
def test_format_transition_prepares_current_rate(rate, channels):
    effect = LadderFilter(cutoff_hz=1000, resonance=0.2, drive=2)
    effect(signal(frames=4096), 48000)
    curves = controls()
    if rate == 96000:
        curves["cutoff_hz"] = np.geomspace(30000, 40000, 513).astype(np.float32)
    assert_array_equal(
        effect(signal(channels), rate, reset=False, automation=curves),
        LadderFilter()(signal(channels), rate, reset=False, automation=curves),
    )


@pytest.mark.parametrize("rate", [1e100, 1e-100, 1e12])
def test_unrepresentable_smoother_rate_is_rejected_before_reset(rate):
    effect, scalar = LadderFilter(), LadderFilter()
    effect(signal(frames=4096), 48000)
    scalar(signal(frames=4096), 48000)
    with pytest.raises(ValueError):
        effect(signal(), rate, automation={"drive": np.ones(513)})
    assert_array_equal(effect(signal(), 48000, reset=False), scalar(signal(), 48000, reset=False))


def test_empty_strided_integer_partial_controls_and_mode_rejection():
    values = {"cutoff_hz": 1000.0, "resonance": 0.2, "drive": 2.0}
    effect, scalar = LadderFilter(**values), LadderFilter(**values)
    effect(signal(frames=4096), 48000)
    scalar(signal(frames=4096), 48000)
    empty = effect(np.empty((2, 0), np.float32), 48000, automation={k: np.empty(0) for k in values})
    assert empty.shape == (2, 0)
    curve = np.repeat(np.full(513, 2, np.int32), 2)[::2]
    curve.flags.writeable = False
    assert_array_equal(
        effect(signal(), 48000, reset=False, automation={"drive": curve}),
        scalar(signal(), 48000, reset=False),
    )
    for mapping in [None, {}]:
        assert_array_equal(
            LadderFilter()(signal(), 48000, automation=mapping), LadderFilter()(signal(), 48000)
        )
    with pytest.raises(ValueError):
        effect(signal(), 48000, automation={"mode": np.zeros(513, dtype=np.int32)})
    assert {k: getattr(effect, k) for k in values} == {k: getattr(scalar, k) for k in values}


def test_concurrent_native_calls_preserve_independent_reset_results():
    curves_list = []
    for i in range(6):
        curves = controls()
        curves["cutoff_hz"] += 100 * i
        curves["resonance"] += i * 0.01
        curves_list.append(curves)
    audio = signal()
    expected = [LadderFilter()(audio, 48000, automation=c) for c in curves_list]
    effect = LadderFilter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda c: effect(audio, 48000, automation=c), curves_list))
    for actual, want in zip(results, expected):
        assert_array_equal(actual, want)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("warm", [False, True])
def test_96k_continuation_covers_entire_smoothing_ramp(mode, warm):
    # 4096 frames are less than 50 ms at 96 kHz; retain a longer future here.
    rate = 96000
    audio, curves = signal(), controls()
    prefix = signal(frames=4096) if warm else None
    expected, scalar = reference(mode, audio, curves, rate, prefix)
    effect = LadderFilter(mode=mode, **initial(curves))
    if prefix is not None:
        effect(prefix, rate)
    assert_array_equal(effect(audio, rate, reset=not warm, automation=curves), expected)
    future = signal(frames=8192)
    assert len(future[0]) / rate > 0.05
    assert_array_equal(effect(future, rate, reset=False), scalar(future, rate, reset=False))
