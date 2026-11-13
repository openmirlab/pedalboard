"""Actual plugin bus routing, source ownership and stream boundaries."""

import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import VST3Plugin

from .test_au_automation import au_probe as _au_probe  # noqa: F401
from .test_vst3_automation import vst3_probe as _vst3_probe  # noqa: F401


@pytest.fixture(scope="session")
def vst3_probe(request):
    return request.getfixturevalue("_vst3_probe")


@pytest.fixture(scope="session")
def au_probe(request):
    return request.getfixturevalue("_au_probe")


def load(path):
    return VST3Plugin(path, plugin_name="Automation Probe Buses")


def signals(n=259, channels=2, key_channels=2):
    main = np.random.default_rng(92).uniform(-0.4, 0.4, (channels, n)).astype(np.float32)
    key = np.zeros((key_channels, n), np.float32)
    key[:, 31:129] = 1
    return main, key


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192, 65536])
@pytest.mark.parametrize("channels,key_channels,aux_channels", [(1, 2, 1), (2, 1, 2), (2, 2, 1)])
def test_independent_main_sidechain_and_aux(
    vst3_probe, rate, buffer_size, channels, key_channels, aux_channels
):
    p = load(vst3_probe)
    main, key = signals(channels=channels, key_channels=key_channels)
    original_main, original_key = main.copy(), key.copy()
    result = p.process_buses(
        {0: main, 1: key},
        rate,
        output_buses={0: channels, 1: aux_channels},
        buffer_size=buffer_size,
    )
    assert list(result) == [0, 1]
    assert_array_equal(result[0], main * np.where(key[:1] > 0.5, np.float32(0.25), np.float32(1)))
    assert_array_equal(result[1], 0.5 * main[np.arange(aux_channels) % channels])
    assert_array_equal(main, original_main)
    assert_array_equal(key, original_key)
    assert not np.shares_memory(result[0], main)
    assert not np.shares_memory(result[0], result[1])
    buses = p.audio_buses
    assert [bus["channels"] for bus in buses["inputs"]] == [channels, key_channels]
    assert [bus["channels"] for bus in buses["outputs"]] == [channels, aux_channels]
    assert buses["inputs"][1]["index"] == 1
    assert buses["inputs"][1]["name"] == "External Key"
    assert buses["outputs"][1]["name"] == "Aux Output"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_absent_detector_is_disabled_and_arrays_are_explicit_planar(vst3_probe, dtype):
    p = load(vst3_probe)
    main, _ = signals()
    source = np.array(main[:, ::2], dtype=dtype, order="F")
    result = p.process_buses({0: source}, 48000, output_buses={0: 2})
    assert_array_equal(result[0], source.astype(np.float32))
    assert result[0].dtype == np.float32
    assert p.audio_buses["inputs"][1]["channels"] == 0
    assert not p.audio_buses["outputs"][1]["enabled"]


def test_split_stream_changes_partition_without_repreparing(vst3_probe):
    main, key = signals(17003)
    p = load(vst3_probe)
    whole = p.process_buses({0: main, 1: key}, 44100, output_buses={0: 2, 1: 2})
    partial = {0: [], 1: []}
    for start, end, size in [(0, 1, 1), (1, 119, 31), (119, 17003, 65536)]:
        output = p.process_buses(
            {0: main[:, start:end], 1: key[:, start:end]},
            44100,
            output_buses={0: 2, 1: 2},
            reset=start == 0,
            buffer_size=size,
        )
        for index, chunks in partial.items():
            chunks.append(output[index])
    for index, chunks in partial.items():
        assert_array_equal(np.concatenate(chunks, axis=1), whole[index])
    # Preparation resets the diagnostic block counter. Eight actual spans survive.
    assert int(p._get_parameter("Blocks").string_value) == 1 + 4 + 3


@pytest.mark.parametrize("change", ["layout", "rate", "api"])
def test_switching_live_bus_stream_requires_reset(vst3_probe, change):
    p = load(vst3_probe)
    main, key = signals()
    p.process_buses({0: main, 1: key}, 48000, output_buses={0: 2})
    if change == "api":
        with pytest.raises(ValueError, match="Reset"):
            p(main, 48000, reset=False)
    else:
        with pytest.raises(ValueError, match="Reset"):
            p.process_buses(
                {0: main} if change == "layout" else {0: main, 1: key},
                44100 if change == "rate" else 48000,
                output_buses={0: 2},
                reset=False,
            )
    result = p.process_buses({0: main, 1: key}, 48000, output_buses={0: 2}, reset=False)
    assert result[0].shape == main.shape
    # Explicit reset restores the original main-only convenience route.
    assert_array_equal(p(main, 48000, reset=True), main)
    assert not p.audio_buses["inputs"][1]["enabled"]
    with pytest.raises(ValueError, match="Reset"):
        p.process_buses({0: main}, 48000, output_buses={0: 2}, reset=False)


@pytest.mark.parametrize(
    "inputs,outputs,extra,error",
    [
        ({2: np.zeros((2, 10), np.float32)}, {0: 2}, {}, "exist"),
        ({0: np.zeros((2, 10), np.float32)}, {2: 2}, {}, "exist"),
        ({0: np.zeros((2, 10), np.float32)}, {}, {}, "output"),
        ({0: np.zeros((2, 10), np.float32)}, {0: 2}, {"num_frames": 9}, "share"),
        ({0: np.zeros((2, 10), np.float32)}, {0: 65}, {}, "64"),
        ({0: np.zeros((2, 10), np.float32)}, {0: 2}, {"buffer_size": 0}, "Unsafe"),
        ({0: np.zeros((2, 10), np.float32)}, {0: 2}, {"sample_rate": float("nan")}, "Unsafe"),
    ],
)
def test_invalid_request_preserves_active_stream(vst3_probe, inputs, outputs, extra, error):
    p = load(vst3_probe)
    main, key = signals()
    kwargs = {"sample_rate": 48000, "output_buses": outputs, **extra}
    p.process_buses({0: main, 1: key}, 48000, output_buses={0: 2})
    before = int(p._get_parameter("Blocks").string_value)
    buses = p.audio_buses
    with pytest.raises(ValueError, match=error):
        p.process_buses(inputs, **kwargs)
    assert int(p._get_parameter("Blocks").string_value) == before
    assert p.audio_buses == buses
    p.process_buses({0: main, 1: key}, 48000, output_buses={0: 2}, reset=False)


@pytest.mark.parametrize("array", [np.zeros(5), np.zeros((2, 5), np.int32), [[0.0] * 5]])
def test_invalid_array(vst3_probe, array):
    with pytest.raises((TypeError, ValueError)):
        load(vst3_probe).process_buses({0: array}, 48000, output_buses={0: 2})


def test_zero_frames_preserves_live_stream(vst3_probe):
    p = load(vst3_probe)
    main, _ = signals()
    p.process_buses({0: main}, 48000, output_buses={0: 2})
    before = int(p._get_parameter("Blocks").string_value)
    result = p.process_buses({0: main[:, :0]}, 48000, output_buses={0: 2})
    assert result[0].shape == (2, 0)
    assert int(p._get_parameter("Blocks").string_value) == before


def test_no_input_requires_duration_and_supported_layout(vst3_probe):
    p = VST3Plugin(vst3_probe, plugin_name="Automation Probe Instrument")
    with pytest.raises(ValueError, match="num_frames"):
        p.process_buses({}, 48000, output_buses={0: 1})
    result = p.process_buses({}, 48000, output_buses={0: 1}, num_frames=259, buffer_size=1)
    assert_array_equal(result[0], np.zeros((1, 259), np.float32))


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
@pytest.mark.parametrize("channels,key_channels,aux_channels", [(1, 2, 1), (2, 1, 2), (2, 2, 1)])
def test_same_loaded_binary_matches_independent_native_host(
    vst3_probe, tmp_path, rate, buffer_size, channels, key_channels, aux_channels
):
    bundle = Path(vst3_probe)
    if platform.system() == "Darwin":
        binary = bundle / "Contents/MacOS/AutomationProbe"
        host = bundle.parent / "NativeBusHost"
    elif platform.system() == "Windows":
        binary = bundle
        host = bundle.parent / "Release/NativeBusHost.exe"
    else:
        binary = next(bundle.glob("Contents/*/AutomationProbe.so"))
        host = bundle.parent / "NativeBusHost"
    output = tmp_path / "native.f32"
    n = 259
    command = [
        str(host),
        str(binary),
        str(output),
        *map(str, [rate, buffer_size, channels, key_channels, aux_channels, n]),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    native = np.fromfile(output, np.float32).reshape(channels + aux_channels, n)
    main = np.array(
        [
            np.float32(-0.1 if c else 0.2) + np.arange(n, dtype=np.float32) * np.float32(1 / 65536)
            for c in range(channels)
        ]
    )
    _, key = signals(n, channels, key_channels)
    # First require the native positive control to satisfy the intended routing.
    expected_main = main * np.where(key[:1] > 0.5, np.float32(0.25), np.float32(1))
    assert_array_equal(native[:channels], expected_main)
    assert_array_equal(native[channels:], main[np.arange(aux_channels) % channels] * 0.5)
    actual = load(vst3_probe).process_buses(
        {0: main, 1: key},
        rate,
        output_buses={0: channels, 1: aux_channels},
        buffer_size=buffer_size,
    )
    assert_array_equal(actual[0], native[:channels])
    assert_array_equal(actual[1], native[channels:])


@pytest.mark.skipif(platform.system() != "Darwin", reason="Audio Units require macOS")
@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192, 65536])
@pytest.mark.parametrize("channels,key_channels,aux_channels", [(1, 2, 1), (2, 1, 2), (2, 2, 1)])
def test_au_multiple_buses(au_probe, rate, buffer_size, channels, key_channels, aux_channels):
    from pedalboard import AudioUnitPlugin

    p = AudioUnitPlugin(au_probe, plugin_name="Automation AU Buses")
    main, key = signals(channels=channels, key_channels=key_channels)
    output = p.process_buses(
        {0: main, 1: key},
        rate,
        output_buses={0: channels, 1: aux_channels},
        buffer_size=buffer_size,
    )
    assert_array_equal(output[0], main * np.where(key[:1] > 0.5, np.float32(0.25), np.float32(1)))
    assert_array_equal(output[1], main[np.arange(aux_channels) % channels] * 0.5)
    # AU cannot disable buses in this JUCE host: omitted detector gets silence.
    silent_detector = p.process_buses({0: main}, rate, output_buses={0: channels})
    assert_array_equal(silent_detector[0], main)
    assert p.audio_buses["inputs"][1]["enabled"]
    # AU required sidechain buses survive a convenience reset. Main-only calls
    # must allocate silent input storage even when there are fewer outputs.
    assert_array_equal(p(main, rate), main)


@pytest.mark.parametrize("fmt", ["vst3", "au"])
@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192])
def test_explicit_audio_buses_preserve_raw_latency(request, fmt, rate, buffer_size):
    if fmt == "au":
        if platform.system() != "Darwin":
            pytest.skip("Audio Units require macOS")
        from pedalboard import AudioUnitPlugin

        p = AudioUnitPlugin(
            request.getfixturevalue("au_probe"), plugin_name="Automation AU Delayed"
        )
    else:
        p = VST3Plugin(
            request.getfixturevalue("vst3_probe"), plugin_name="Automation Probe Delayed"
        )
    p._get_parameter("Frequency").raw_value = 1
    main, _ = signals()
    expected = np.pad(main, ((0, 0), (20, 0)))[:, : main.shape[1]]
    output = p.process_buses({0: main}, rate, output_buses={0: 2}, buffer_size=buffer_size)
    assert p.reported_latency_samples == 20
    assert_array_equal(output[0], expected)
    # The legacy path still performs its original whole-render compensation.
    assert_array_equal(p(main, rate), main)


@pytest.mark.parametrize(
    "inputs,outputs",
    [({1: np.zeros((2, 10), np.float32)}, {0: 2}), ({0: np.zeros((2, 10), np.float32)}, {0: 1})],
)
def test_unsupported_joint_layout_is_explicit(vst3_probe, inputs, outputs):
    with pytest.raises(ValueError, match="layout"):
        load(vst3_probe).process_buses(inputs, 48000, output_buses=outputs)


@pytest.mark.parametrize("fmt", ["vst3", "au"])
@pytest.mark.parametrize("rate", [44100, 48000])
def test_real_chow_bus_output_matches_legacy_compensated_audio(fmt, rate):
    folder = Path(__file__).parent / "plugins/effect" / platform.system()
    if fmt == "au":
        if platform.system() != "Darwin":
            pytest.skip("Audio Units require macOS")
        from pedalboard import AudioUnitPlugin

        from .au_automation_plugin.register import register_bundle

        p = AudioUnitPlugin(register_bundle(folder / "CHOWTapeModel.component"))
        reference = AudioUnitPlugin(register_bundle(folder / "CHOWTapeModel.component"))
    else:
        path = folder / "CHOWTapeModel.vst3"
        if not path.exists():
            pytest.skip("Platform-specific CHOW fixture unavailable")
        p = VST3Plugin(str(path))
        reference = VST3Plugin(str(path))
    main, _ = signals(1001)
    # The diagnostic and actual effect paths have different PDC contracts.
    # Supply the exact extra source silence needed to compare the returned audio.
    legacy = reference(main, rate)
    latency = reference.reported_latency_samples
    # CHOW's processing depends on native partition boundaries. Match the
    # legacy source block and separate drain, rather than assuming invariance.
    raw = p.process_buses(
        {0: np.pad(main, ((0, 0), (0, latency)))},
        rate,
        output_buses={0: 2},
        buffer_size=main.shape[1],
    )[0]
    assert p.reported_latency_samples == latency
    assert_array_equal(raw[:, latency : latency + main.shape[1]], legacy)


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("buffer_size", [1, 31, 8192, 65536])
@pytest.mark.parametrize("with_key", [False, True])
def test_unmodified_juce_noisegate_matches_native_host(tmp_path, rate, buffer_size, with_key):
    configured = os.environ.get("PEDALBOARD_TEST_JUCE_SIDECHAIN_BUILD")
    if not configured:
        pytest.skip(
            "Optional JUCE NoiseGate build is not configured; see tests/juce_sidechain_plugin"
        )
    build = Path(configured).resolve()
    bundle = next((build / "SidechainAcceptance_artefacts/Release/VST3").glob("*.vst3"))
    host = build / "NativeSidechainHost_artefacts/Release/NativeSidechainHost"
    if platform.system() == "Windows":
        host = host.with_suffix(".exe")
    target = tmp_path / "native-noisegate.f32"
    native = subprocess.run(
        [str(host), str(bundle), str(target), str(rate), str(buffer_size), str(int(with_key))],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert native.returncode == 0, native.stdout + native.stderr
    n = rate + 160
    main = np.empty((2, n), np.float32)
    main[0] = 0.2
    main[1] = -0.1
    key = np.zeros_like(main)
    if with_key:
        key[:, 31] = 1
    expected = np.zeros_like(main)
    if with_key:
        expected[:, 31 : 31 + rate] = main[:, 31 : 31 + rate]
    native_audio = np.fromfile(target, np.float32).reshape(2, n)
    assert_array_equal(native_audio, expected)
    p = VST3Plugin(str(bundle))
    p._get_parameter("Threshold").raw_value = 0.5
    p._get_parameter("Alpha").raw_value = 0
    output = p.process_buses({0: main, 1: key}, rate, output_buses={0: 2}, buffer_size=buffer_size)
    assert_array_equal(output[0], native_audio)
