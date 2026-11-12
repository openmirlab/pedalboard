"""Dense native automation for the four stateless effects in the first batch."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import Bitcrush, Clipping, Distortion, Gain

CASES = [
    (Gain, "gain_db", -24.0, 6.0),
    (Clipping, "threshold_db", -24.0, -1.0),
    (Distortion, "drive_db", -6.0, 24.0),
    (Bitcrush, "bit_depth", 2.5, 12.5),
]


def signal(channels=2, frames=257):
    return np.random.default_rng(814).uniform(-1.5, 1.5, (channels, frames)).astype(np.float32)


def scalar_reference(cls, name, audio, curve, rate):
    # The effects are channel-independent. Mono calls avoid upstream's ambiguous
    # single-frame stereo layout while keeping exactly the scalar DSP arithmetic.
    rows = []
    for channel in audio:
        effect = cls()
        row = []
        for i, value in enumerate(curve):
            setattr(effect, name, float(value))
            row.append(effect(channel[i : i + 1], rate, reset=i == 0))
        rows.append(np.concatenate(row))
    return np.stack(rows)


@pytest.mark.parametrize("cls,name,low,high", CASES)
@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
def test_dense_curve_matches_scalar_dsp(cls, name, low, high, rate, channels, buffer_size):
    audio = signal(channels)
    curve = np.linspace(low, high, audio.shape[1], dtype=np.float32)
    curve[50:80] = curve[50]
    original = audio.copy()
    controls = curve.copy()
    effect = cls()
    actual = effect.process(audio, rate, buffer_size, automation={name: curve})
    assert_array_equal(actual, scalar_reference(cls, name, audio, curve, rate))
    assert getattr(effect, name) == float(curve[-1])
    assert_array_equal(audio, original)
    assert_array_equal(curve, controls)


@pytest.mark.parametrize("cls,name,low,high", CASES)
@pytest.mark.parametrize("layout", ["mono", "planar", "interleaved"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_constant_layouts_and_dtype(cls, name, low, high, layout, dtype):
    audio = signal().astype(dtype)
    if layout == "mono":
        audio = audio[0]
    elif layout == "interleaved":
        audio = audio.T
    curve = np.full(257, low, dtype=dtype)
    # Read-only, strided inputs must remain valid and unmodified.
    backing = np.repeat(curve, 2)
    curve = backing[::2]
    audio.flags.writeable = False
    curve.flags.writeable = False
    expected = cls(**{name: low})(audio, 48000)
    assert_array_equal(cls()(audio, 48000, automation={name: curve}), expected)


@pytest.mark.parametrize("cls,name,low,high", CASES)
def test_irregular_stream_and_following_scalar(cls, name, low, high):
    audio = signal()
    curve = np.linspace(low, high, 257, dtype=np.float32)
    effect = cls()
    # Establish stereo layout before tiny/square fragments.
    effect(audio[:, :8], 48000)
    parts = []
    start = 0
    for end in [1, 3, 20, 21, 112, 257]:
        parts.append(
            effect(audio[:, start:end], 48000, reset=False, automation={name: curve[start:end]})
        )
        start = end
    assert_array_equal(np.concatenate(parts, axis=1), cls()(audio, 48000, automation={name: curve}))
    assert_array_equal(effect(audio, 48000, reset=False), cls(**{name: high})(audio, 48000))


@pytest.mark.parametrize("cls,name,low,high", CASES)
@pytest.mark.parametrize(
    "bad",
    [
        "unknown",
        "length",
        "ndim",
        "nan",
        "infinity",
        "overflow",
        "complex",
        "bool",
        "list",
        "scalar",
        "mapping",
    ],
)
def test_invalid_controls_do_not_change_parameter(cls, name, low, high, bad):
    effect = cls(**{name: low})
    audio = signal()
    curve = np.full(257, high)
    if bad == "length":
        curve = curve[:-1]
    elif bad == "ndim":
        curve = curve[None, :]
    elif bad in ["nan", "infinity", "overflow"]:
        curve[-1] = {"nan": np.nan, "infinity": np.inf, "overflow": 1e100}[bad]
    elif bad == "complex":
        curve = curve.astype(complex)
    elif bad == "bool":
        curve = curve.astype(bool)
    elif bad == "list":
        curve = curve.tolist()
    elif bad == "scalar":
        curve = high
    automation = {"not_a_parameter" if bad == "unknown" else name: curve}
    if bad == "mapping":
        automation = []
    with pytest.raises((ValueError, TypeError)):
        effect(audio, 48000, automation=automation)
    assert getattr(effect, name) == low


@pytest.mark.parametrize("cls,name,low,high", CASES)
def test_empty_and_legacy_routes(cls, name, low, high):
    effect = cls(**{name: low})
    assert effect(np.empty(0, np.float32), 48000, automation={name: np.empty(0)}).shape == (0,)
    assert getattr(effect, name) == low
    audio = signal()
    expected = cls(**{name: low})(audio, 48000)
    assert_array_equal(effect(audio, 48000, automation={}), expected)
    assert_array_equal(effect.process(audio, 48000, automation=None), expected)


@pytest.mark.parametrize("cls,name,low,high", CASES)
@pytest.mark.parametrize("rate,size", [(0, 64), (-1, 64), (np.nan, 64), (np.inf, 64), (48000, 0)])
def test_invalid_processing_options(cls, name, low, high, rate, size):
    effect = cls(**{name: low})
    with pytest.raises(ValueError):
        effect(signal(), rate, buffer_size=size, automation={name: np.full(257, high)})
    assert getattr(effect, name) == low


@pytest.mark.parametrize(
    "cls,name", [(Gain, "gain_db"), (Clipping, "threshold_db"), (Distortion, "drive_db")]
)
def test_negative_infinity_db_is_silence(cls, name):
    result = cls()(signal(), 48000, automation={name: np.full(257, -np.inf)})
    assert_array_equal(result, np.zeros_like(result))


@pytest.mark.parametrize("cls,name,low,high", CASES)
def test_concurrent_curves_on_one_effect_are_isolated(cls, name, low, high):
    from concurrent.futures import ThreadPoolExecutor

    audio = signal(frames=1024)
    effect = cls()
    curves = [np.full(1024, value, dtype=np.float32) for value in np.linspace(low, high, 8)]
    expected = [cls()(audio, 48000, automation={name: curve}) for curve in curves]

    def render(curve):
        return effect(audio, 48000, automation={name: curve})

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(render, curves))
    for result, reference in zip(results, expected):
        assert_array_equal(result, reference)


@pytest.mark.parametrize("depth", [0.0, 32.0])
def test_bit_depth_curve_endpoints(depth):
    audio = signal()
    assert_array_equal(
        Bitcrush()(audio, 48000, automation={"bit_depth": np.full(257, depth)}),
        Bitcrush(depth)(audio, 48000),
    )


@pytest.mark.parametrize("cls,name,low,high", CASES)
def test_integer_control_arrays(cls, name, low, high):
    audio = signal()
    curve = np.arange(257, dtype=np.uint16) % 7
    assert_array_equal(
        cls()(audio, 48000, automation={name: curve}),
        scalar_reference(cls, name, audio, curve, 48000),
    )
