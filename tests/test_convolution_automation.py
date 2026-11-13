"""Mix curves retain convolution history while FFT work stays block based."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import Convolution
from pedalboard.io import AudioFile


def signal(channels=2, frames=1025, rate=48000):
    t = np.arange(frames) / rate
    x = 0.35 * np.sin(2 * np.pi * 173 * t) + 0.1 * np.cos(2 * np.pi * 3337 * t)
    return np.stack([x * 0.63**c for c in range(channels)]).astype(np.float32)


def impulse(kind="long"):
    if kind == "short":
        return np.array([1, 0.4, -0.2], np.float32)
    rng = np.random.default_rng(52)
    ir = (
        rng.normal(0, 0.06, (2 if kind == "stereo" else 1, 8193 if kind == "stereo" else 4097))
        * np.exp(-np.arange(8193 if kind == "stereo" else 4097) / 1700)
    ).astype(np.float32)
    ir[:, 0] = 1
    return ir if kind == "stereo" else ir[0]


def audio_close(actual, expected):
    # FFT roundoff varies with partial-call shape. These bounds precede results.
    diff = actual.astype(np.float64) - expected
    assert np.max(np.abs(diff)) <= 2e-5
    error = np.sqrt(np.mean(diff**2))
    level = np.sqrt(np.mean(expected.astype(np.float64) ** 2))
    assert error <= max(1e-12, level * 1e-5)


def pair(ir, channels, rate, first, warm):
    a, b = Convolution(ir, first, sample_rate=48000), Convolution(ir, first, sample_rate=48000)
    for effect in [a, b]:
        effect(np.zeros((channels, 2048), np.float32), rate, buffer_size=128)
        if warm:
            effect(signal(channels, 2048, rate), rate, buffer_size=128)
    return a, b


def replay(effect, audio, curve, rate, reset):
    parts = []
    for i in range(audio.shape[1]):
        effect.mix = float(curve[i])
        parts.append(effect(audio[:, i : i + 1], rate, buffer_size=1, reset=reset and i == 0))
    return np.concatenate(parts, axis=1)


@pytest.mark.parametrize("rate", [44100, 48000, 96000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("kind", ["short", "long", "stereo"])
@pytest.mark.parametrize("history", ["cold", "warm", "warm_reset"])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
def test_mix_ramps_and_nonzero_tail_match_scalar(rate, channels, kind, history, buffer_size):
    ir = impulse(kind)
    audio = signal(channels, rate=rate)
    curve = np.linspace(0.15, 0.85, 1025, dtype=np.float32)
    effect, scalar = pair(ir, channels, rate, float(curve[0]), history != "cold")
    reset = history != "warm"
    saved_audio, saved_curve, saved_ir = audio.copy(), curve.copy(), ir.copy()
    actual = effect(audio, rate, reset=reset, buffer_size=buffer_size, automation={"mix": curve})
    audio_close(actual, replay(scalar, audio, curve, rate, reset))
    assert effect.mix == float(curve[-1])
    assert_array_equal(audio, saved_audio)
    assert_array_equal(curve, saved_curve)
    assert_array_equal(ir, saved_ir)
    future = np.zeros((channels, 8192), np.float32)
    tail = effect(future, rate, buffer_size=128, reset=False)
    audio_close(tail, scalar(future, rate, buffer_size=128, reset=False))
    if kind != "short":
        assert np.max(np.abs(tail[:, :1024])) > 1e-6


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("reset", [False, True])
def test_cold_constant_same_partition_and_readonly_float64(transpose, reset):
    audio = signal().astype(np.float64)
    if transpose:
        audio = audio.T
    audio.flags.writeable = False
    curve = np.full(1025, 0.3)
    curve.flags.writeable = False
    ir = impulse("stereo")
    assert_array_equal(
        Convolution(ir, sample_rate=48000)(audio, 48000, reset=reset, automation={"mix": curve}),
        Convolution(ir, 0.3, sample_rate=48000)(audio, 48000, reset=reset),
    )


def test_growing_chunks_keep_long_tail_and_mixer_progress():
    ir = impulse("stereo")
    audio = np.zeros((2, 4096), np.float32)
    curve = np.linspace(0.1, 0.9, 4096, dtype=np.float32)
    effect, scalar = Convolution(ir, 0.1, 48000), Convolution(ir, 0.1, 48000)
    prefix = signal(frames=2048)
    effect(prefix, 48000, buffer_size=1)
    scalar(prefix, 48000, buffer_size=128)
    expected = replay(scalar, audio, curve, 48000, False)
    parts = []
    start = 0
    for end, size in [(1, 1), (3, 2), (40, 32), (101, 128), (4096, 16384)]:
        parts.append(
            effect(
                audio[:, start:end],
                48000,
                reset=False,
                buffer_size=size,
                automation={"mix": curve[start:end]},
            )
        )
        start = end
    actual = np.concatenate(parts, axis=1)
    audio_close(actual, expected)
    assert np.max(np.abs(actual)) > 1e-5
    future = np.zeros((2, 8192), np.float32)
    audio_close(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


def test_wet_history_keeps_running_while_mix_is_zero():
    # IR taps arrive after the input impulse; hiding wet must not stop its engine.
    ir = np.zeros(4097, np.float32)
    ir[0] = 1
    ir[1800] = 0.5
    ir[3000] = 0.25
    effect = Convolution(ir, 0, sample_rate=48000)
    audio = np.zeros(4096, np.float32)
    audio[0] = 1
    mix = np.zeros(4096, np.float32)
    mix[100:] = 1
    result = effect(audio, 48000, automation={"mix": mix})
    assert abs(result[1800]) > 1e-3
    assert abs(result[3000]) > 1e-3
    assert np.max(np.abs(result[200:1700])) < 1e-6


def test_one_tap_ir_has_original_linear_fifty_ms_mixer_ramp():
    # Original normalization turns a unit IR into gain0.125. No FFT timing delay.
    effect = Convolution(np.ones(1, np.float32), 0, sample_rate=48000)
    effect(np.zeros(1024, np.float32), 48000)
    audio = np.ones(3000, np.float32) * 0.2
    result = effect(audio, 48000, reset=False, automation={"mix": np.ones(3000)})
    ramp = np.minimum((np.arange(3000) + 1) / 2400, 1)
    expected = 0.2 * (1 - ramp) + 0.025 * ramp
    np.testing.assert_allclose(result, expected, atol=1e-5, rtol=0)
    assert result[0] > 0.19
    assert abs(result[-1] - 0.025) < 1e-7


def test_impulse_leading_silence_is_ir_content_not_engine_latency():
    ir = np.zeros(257, np.float32)
    ir[113] = 1
    audio = np.zeros(1024, np.float32)
    audio[0] = 1
    output = Convolution(ir, sample_rate=48000)(audio, 48000, automation={"mix": np.ones(1024)})
    expected = np.zeros(1024, np.float32)
    expected[113] = 0.125
    np.testing.assert_allclose(output, expected, atol=1e-7, rtol=0)


@pytest.mark.parametrize("bad", [-1e-100, 1.00000001, np.nan, np.inf])
@pytest.mark.parametrize("reset", [False, True])
def test_invalid_late_curve_preserves_tail_and_target(bad, reset):
    effect, scalar = pair(impulse(), 2, 48000, 0.3, True)
    curve = np.linspace(0.2, 0.8, 1025)
    curve[-1] = bad
    with pytest.raises(ValueError):
        effect(signal(), 48000, reset=reset, automation={"mix": curve})
    assert effect.mix == scalar.mix
    future = np.zeros((2, 8192), np.float32)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


@pytest.mark.parametrize(
    "rate,channels", [(99.99, 2), (1e-100, 2), (1e10, 2), (1e12, 2), (np.inf, 2), (48000, 3)]
)
@pytest.mark.parametrize("reset", [False, True])
def test_unsafe_format_rejects_before_tail_reset(rate, channels, reset):
    effect, scalar = pair(impulse(), 2, 48000, 0.3, True)
    with pytest.raises(ValueError):
        effect(signal(channels), rate, reset=reset, automation={"mix": np.ones(1025)})
    assert effect.mix == scalar.mix
    future = np.zeros((2, 8192), np.float32)
    assert_array_equal(effect(future, 48000, reset=False), scalar(future, 48000, reset=False))


@pytest.mark.parametrize("rate,channels", [(44100, 2), (48000, 1), (96000, 2)])
def test_format_changes_reset_for_new_ir_resampling(rate, channels):
    ir = impulse("stereo")
    effect, _ = pair(ir, 2, 48000, 0.3, True)
    audio = signal(channels, rate=rate)
    curve = np.linspace(0.2, 0.8, 1025, dtype=np.float32)
    assert_array_equal(
        effect(audio, rate, reset=False, automation={"mix": curve}),
        Convolution(ir, sample_rate=48000)(audio, rate, automation={"mix": curve}),
    )


def test_file_and_array_ir_metadata_and_resampling(tmp_path):
    ir = impulse("stereo")
    path = tmp_path / "impulse.wav"
    with AudioFile(str(path), "w", 44100, 2, bit_depth=32) as file:
        file.write(ir)
    file = Convolution(str(path))
    array = Convolution(ir, sample_rate=44100)
    curve = np.linspace(0, 1, 1025, dtype=np.float32)
    audio_close(
        file(signal(), 48000, automation={"mix": curve}),
        array(signal(), 48000, automation={"mix": curve}),
    )
    assert file.impulse_response_filename == str(path)
    assert file.impulse_response is None
    assert array.impulse_response_filename is None
    assert_array_equal(array.impulse_response, ir)


def test_empty_strided_integer_controls_and_unsupported_ir_curve():
    effect, scalar = pair(impulse(), 2, 48000, 0.3, True)
    empty = effect(np.empty((2, 0), np.float32), 48000, automation={"mix": np.empty(0)})
    assert empty.shape == (2, 0)
    curve = np.ones(2050, np.int32)[::2]
    curve.flags.writeable = False
    scalar.mix = 1
    audio_close(
        effect(signal(), 48000, reset=False, automation={"mix": curve}),
        scalar(signal(), 48000, reset=False),
    )
    with pytest.raises(ValueError):
        effect(signal(), 48000, automation={"impulse_response": np.ones(1025)})
    for mapping in [None, {}]:
        assert_array_equal(
            Convolution(impulse(), 0.3, 48000)(signal(), 48000, automation=mapping),
            Convolution(impulse(), 0.3, 48000)(signal(), 48000),
        )


@pytest.mark.parametrize("mix", [0, 1])
def test_mix_endpoints_match_scalar(mix):
    ir = impulse()
    audio = signal()
    assert_array_equal(
        Convolution(ir, sample_rate=48000)(audio, 48000, automation={"mix": np.full(1025, mix)}),
        Convolution(ir, mix, 48000)(audio, 48000),
    )
    if mix == 0:
        assert_array_equal(
            audio,
            Convolution(ir, sample_rate=48000)(audio, 48000, automation={"mix": np.zeros(1025)}),
        )


def test_native_calls_serialize_independent_resets():
    ir = impulse()
    audio = signal()
    curves = [np.linspace(0.1, 0.3 + i * 0.1, 1025, dtype=np.float32) for i in range(6)]
    expected = [
        Convolution(ir, sample_rate=48000)(audio, 48000, automation={"mix": v}) for v in curves
    ]
    effect = Convolution(ir, sample_rate=48000)
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(pool.map(lambda v: effect(audio, 48000, automation={"mix": v}), curves))
    for a, b in zip(actual, expected):
        assert_array_equal(a, b)


def test_lowest_supported_rate_has_original_unit_ir_mix():
    audio = np.ones(1024, np.float32) * 0.2
    result = Convolution(np.ones(1, np.float32), sample_rate=100)(
        audio, 100, automation={"mix": np.full(1024, 0.25)}
    )
    np.testing.assert_allclose(result, audio * 0.78125, atol=1e-7, rtol=0)
