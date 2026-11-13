"""Combined source audio, sidechain, MIDI and physical parameter curves."""

import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pedalboard import VST3Plugin

from .test_au_automation import au_probe as _au_probe  # noqa: F401
from .test_hosted_audio_buses import signals
from .test_vst3_automation import expected
from .test_vst3_automation import vst3_probe as _vst3_probe  # noqa: F401


@pytest.fixture(params=["vst3", "au"])
def host(request):
    fmt = request.param
    if fmt == "au":
        if platform.system() != "Darwin":
            pytest.skip("Audio Units require macOS")
        from pedalboard import AudioUnitPlugin

        cls, prefix, path = AudioUnitPlugin, "Automation AU ", request.getfixturevalue("_au_probe")
    else:
        cls, prefix, path = VST3Plugin, "Automation Probe ", request.getfixturevalue("_vst3_probe")

    def create(delay=False):
        return cls(path, plugin_name=prefix + ("Delayed " if delay else "") + "Audio MIDI")

    return fmt, create


EVENTS = [(0, True), (7, False), (31, True), (127, False), (128, True), (258, False)]


def notes(rate, events=EVENTS):
    return [
        (bytes([0x90 if on else 0x80, 60, 100 if on else 0]), frame / rate) for frame, on in events
    ]


def mask(n=259, events=EVENTS):
    result = np.zeros(n, np.float32)
    state, cursor = False, 0
    for frame, on in events:
        result[cursor:frame] = state
        cursor, state = frame, on
    result[cursor:] = state
    return result


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("buffer", [1, 31, 8192, 65536])
@pytest.mark.parametrize("delay", [False, True])
def test_audio_key_and_notes_share_exact_source_clock(host, rate, buffer, delay):
    fmt, create = host
    p = create(delay)
    main, key = signals(key_channels=1)
    actual, returned = p.process_buses(
        {0: main, 1: key},
        rate,
        output_buses={0: 2, 1: 1},
        midi_messages=notes(rate),
        return_midi=True,
        buffer_size=buffer,
    )
    gated = main * mask() * np.where(key[:1] > 0.5, np.float32(0.25), np.float32(1))
    wanted = expected(gated, np.zeros(259), np.full(259, 20000)) if fmt == "vst3" else gated
    if delay:
        wanted = np.pad(wanted, ((0, 0), (20, 0)))[:, :259]
    np.testing.assert_allclose(actual[0], wanted, atol=3e-7, rtol=3e-7)
    assert_array_equal(actual[1], main[:1] * 0.5)
    assert p.midi_capabilities["accepts_input"]
    assert p.reported_latency_samples == (20 if delay else 0)
    if fmt == "vst3":
        transposed = [(data, time) for data, time in returned if data[1] == 72]
        assert transposed == [
            (bytes([0x90 if on else 0x80, 72, 100 if on else 0]), frame / rate)
            for frame, on in EVENTS
        ]
    else:
        assert not p.midi_capabilities["produces_output"]
        assert returned == []


@pytest.mark.parametrize("pattern", ["constant", "sparse", "dense"])
@pytest.mark.parametrize("buffer", [1, 31, 8192])
@pytest.mark.parametrize("mode", ["sample_steps", "vst3_events"])
def test_joint_curves_apply_to_audio_without_losing_midi(host, pattern, buffer, mode):
    fmt, create = host
    if fmt == "au" and mode != "sample_steps":
        pytest.skip("AU has no native VST3 queues")
    p = create()
    p.automation_mode = mode
    main, key = signals(key_channels=1)
    gain = np.full(259, -6.0)
    frequency = np.full(259, 10000.0)
    if pattern == "sparse":
        gain[31:] = 3
        frequency[128:] = 2000
    elif pattern == "dense":
        gain = np.linspace(-30, 6, 259)
        frequency = np.geomspace(20, 20000, 259)
    actual = p.process_buses(
        {0: main, 1: key},
        48000,
        output_buses={0: 2, 1: 1},
        midi_messages=notes(48000),
        buffer_size=buffer,
        automation={"gain_db": gain, "frequency_hz": frequency},
    )
    gated = main * mask() * np.where(key[:1] > 0.5, np.float32(0.25), np.float32(1))
    wanted = (
        expected(gated, gain, frequency)
        if fmt == "vst3"
        else (
            gated.astype(np.float64)
            * 10 ** (gain.astype(np.float32) / 20)
            * frequency.astype(np.float32)
            / 20000
        ).astype(np.float32)
    )
    np.testing.assert_allclose(actual[0], wanted, atol=3e-7, rtol=3e-7)
    assert_array_equal(actual[1], main[:1] * 0.5)


def test_split_held_notes_and_control_changes_keep_state(host):
    _, create = host
    p = create(True)
    main, key = signals(key_channels=1)
    gain, frequency = np.linspace(-12, 3, 259), np.geomspace(200, 18000, 259)
    kwargs = {"sample_rate": 48000, "output_buses": {0: 2, 1: 1}}
    whole = p.process_buses(
        {0: main, 1: key},
        **kwargs,
        midi_messages=notes(48000),
        automation={"gain_db": gain, "frequency_hz": frequency},
    )
    chunks = []
    for start, end in [(0, 7), (7, 119), (119, 259)]:
        midi = notes(48000, [(frame - start, on) for frame, on in EVENTS if start <= frame < end])
        chunks.append(
            p.process_buses(
                {0: main[:, start:end], 1: key[:, start:end]},
                **kwargs,
                midi_messages=midi,
                automation={"gain_db": gain[start:end], "frequency_hz": frequency[start:end]},
                reset=start == 0,
                buffer_size=31,
            )
        )
    for index in [0, 1]:
        assert_array_equal(np.concatenate([part[index] for part in chunks], axis=1), whole[index])


@pytest.mark.parametrize("invalid", ["audio", "curve", "index", "rate"])
def test_pure_validation_does_not_consume_midi_or_reset(host, invalid):
    _, create = host
    p = create()
    main, key = signals(key_channels=1)
    p.process_buses({0: main, 1: key}, 48000, output_buses={0: 2}, midi_messages=notes(48000))
    before = p._get_parameter("Blocks").string_value
    consumed = []

    def midi():
        consumed.append(True)
        yield notes(48000)[0]

    args = {
        "input_buses": {0: main, 1: key},
        "sample_rate": 48000,
        "output_buses": {0: 2},
        "midi_messages": midi(),
    }
    if invalid == "audio":
        args["input_buses"] = {0: main[0]}
    elif invalid == "curve":
        args["automation"] = {"gain_db": np.full(259, 999.0)}
    elif invalid == "index":
        args["output_buses"] = {99: 2}
    else:
        args["sample_rate"] = float("nan")
    with pytest.raises((ValueError, TypeError)):
        p.process_buses(**args)
    assert consumed == []
    assert p._get_parameter("Blocks").string_value == before


def test_late_and_negative_notes_do_not_leak_into_next_call(host):
    _, create = host
    p = create()
    main, key = signals(key_channels=1)
    # A negative fractional-sample timestamp must not truncate into sample zero.
    midi = [(bytes([0x90, 60, 100]), -0.25 / 48000), *notes(48000, [(259, True)])]
    first = p.process_buses({0: main, 1: key}, 48000, output_buses={0: 2}, midi_messages=midi)
    second = p.process_buses({0: main, 1: key}, 48000, output_buses={0: 2}, reset=False)
    assert_array_equal(first[0], np.zeros_like(main))
    assert_array_equal(second[0], np.zeros_like(main))


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("buffer", [1, 31, 8192])
@pytest.mark.parametrize("delay", [False, True])
def test_combined_input_matches_independent_native_sdk_host(request, tmp_path, rate, buffer, delay):
    bundle = Path(request.getfixturevalue("_vst3_probe"))
    if platform.system() == "Darwin":
        binary, native_host = (
            bundle / "Contents/MacOS/AutomationProbe",
            bundle.parent / "NativeBusHost",
        )
    elif platform.system() == "Windows":
        binary, native_host = bundle, bundle.parent / "Release/NativeBusHost.exe"
    else:
        binary, native_host = (
            next(bundle.glob("Contents/*/AutomationProbe.so")),
            bundle.parent / "NativeBusHost",
        )
    output = tmp_path / "native.f32"
    run = subprocess.run(
        [
            str(native_host),
            str(binary),
            str(output),
            *map(str, [rate, buffer, 2, 1, 1, 259]),
            "delayed" if delay else "midi",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    native = np.fromfile(output, np.float32).reshape(3, 259)
    main = np.array(
        [
            np.float32(value) + np.arange(259, dtype=np.float32) / np.float32(65536)
            for value in [0.2, -0.1]
        ]
    )
    _, key = signals(key_channels=1)
    gated = main * mask() * np.where(key[:1] > 0.5, np.float32(0.25), np.float32(1))
    wanted = expected(gated, np.zeros(259), np.full(259, 20000))
    if delay:
        wanted = np.pad(wanted, ((0, 0), (20, 0)))[:, :259]
    assert_array_equal(native[:2], wanted)
    assert_array_equal(native[2:], main[:1] * 0.5)
    p = VST3Plugin(
        str(bundle), plugin_name="Automation Probe " + ("Delayed " if delay else "") + "Audio MIDI"
    )
    actual, midi = p.process_buses(
        {0: main, 1: key},
        rate,
        output_buses={0: 2, 1: 1},
        midi_messages=notes(rate),
        buffer_size=buffer,
        return_midi=True,
        automation={"gain_db": np.zeros(259), "frequency_hz": np.full(259, 20000.0)},
    )
    assert_array_equal(actual[0], native[:2])
    assert_array_equal(actual[1], native[2:])
    native_midi = sorted(
        [
            tuple(map(int, row.split()))
            for row in Path(str(output) + ".midi").read_text().splitlines()
        ],
        key=lambda row: row[0],
    )
    actual_midi = [
        (round(time * rate), 0 if data[0] == 0x90 else 1, data[1]) for data, time in midi
    ]
    assert actual_midi == native_midi


@pytest.mark.parametrize("fmt", ["vst3", "au"])
@pytest.mark.parametrize("rate", [44100, 48000])
def test_actual_magical_instrument_supports_new_entry_point(fmt, rate):
    folder = Path(__file__).parent / "plugins/instrument" / platform.system()
    if fmt == "au":
        if platform.system() != "Darwin":
            pytest.skip("Audio Units require macOS")
        from pedalboard import AudioUnitPlugin

        from .au_automation_plugin.register import register_bundle

        path, cls = register_bundle(folder / "Magical8bitPlug2.component"), AudioUnitPlugin
    else:
        path, cls = str(folder / "Magical8bitPlug2.vst3"), VST3Plugin
        if not Path(path).exists():
            pytest.skip("Platform-specific Magical fixture unavailable")
    p, reference = cls(path), cls(path)
    midi = notes(rate, [(0, True), (int(rate * 0.1), False)])
    n = rate // 8
    controls = {"gain": np.linspace(0.1, 0.5, n)}
    actual = p.process_buses(
        {}, rate, num_frames=n, output_buses={0: 2}, midi_messages=midi, automation=controls
    )[0]
    legacy = reference(midi, 0.125, rate, automation=controls)
    assert actual.shape == legacy.shape == (2, n)
    assert np.max(np.abs(actual)) > 0.001
    assert_array_equal(actual, legacy)


@pytest.mark.parametrize("buffer", [1, 31, 8192])
def test_midi_controller_changes_arrive_at_exact_sample(host, buffer):
    fmt, create = host
    main, key = signals(key_channels=1)
    p = create()
    midi = [*notes(48000), (bytes([0xB0, 1, 127]), 7 / 48000), (bytes([0xB0, 1, 0]), 127 / 48000)]
    actual = p.process_buses(
        {0: main, 1: key}, 48000, output_buses={0: 2}, buffer_size=buffer, midi_messages=midi
    )[0]
    ducked = main * mask() * np.where(key[:1] > 0.5, np.float32(0.25), np.float32(1))
    wanted = expected(ducked, np.zeros(259), np.full(259, 20000)) if fmt == "vst3" else ducked
    # This fixture explicitly maps CC1 to its discrete bypass control.
    wanted[:, 7:127] = main[:, 7:127]
    np.testing.assert_allclose(actual, wanted, atol=3e-7, rtol=3e-7)


@pytest.mark.parametrize("bad_time", [float("nan"), float("inf")])
def test_invalid_midi_timestamp_preserves_warm_state(host, bad_time):
    _, create = host
    p = create()
    main, key = signals(key_channels=1)
    p.process_buses({0: main, 1: key}, 48000, output_buses={0: 2}, midi_messages=notes(48000))
    before = p._get_parameter("Blocks").string_value
    with pytest.raises(ValueError, match="finite"):
        p.process_buses(
            {0: main, 1: key},
            48000,
            output_buses={0: 2},
            midi_messages=[(bytes([0x90, 60, 100]), bad_time)],
        )
    assert p._get_parameter("Blocks").string_value == before


def test_empty_request_does_not_iterate_midi(host):
    _, create = host

    def never():
        raise AssertionError("Empty render consumed MIDI")
        yield

    audio, midi = create().process_buses(
        {0: np.zeros((2, 0), np.float32)},
        48000,
        output_buses={0: 2},
        midi_messages=never(),
        return_midi=True,
    )
    assert audio[0].shape == (2, 0) and midi == []


def test_non_midi_plugin_rejects_events_before_reset(request):
    bundle = request.getfixturevalue("_vst3_probe")
    p = VST3Plugin(bundle, plugin_name="Automation Probe")
    main, _ = signals()
    p.process_buses({0: main}, 48000, output_buses={0: 2})
    before = p._get_parameter("Blocks").string_value
    assert p.midi_capabilities == {"accepts_input": False, "produces_output": False}
    with pytest.raises(ValueError, match="MIDI input"):
        p.process_buses({0: main}, 48000, output_buses={0: 2}, midi_messages=notes(48000))
    assert p._get_parameter("Blocks").string_value == before


def test_generator_can_query_plugin_without_holding_mutex(request):
    import sys

    bundle = request.getfixturevalue("_vst3_probe")
    code = """import sys, numpy as np
from pedalboard import VST3Plugin
p=VST3Plugin(sys.argv[1], plugin_name='Automation Probe Audio MIDI')
def midi():
    assert len(p.audio_buses['inputs']) == 2
    assert p.midi_capabilities['accepts_input']
    assert p.automation_mode == 'sample_steps'
    yield (bytes([0x90,60,100]), 0.0)
y=p.process_buses({0:np.ones((2,259),np.float32)},48000,output_buses={0:2},midi_messages=midi())
assert np.min(y[0]) > .99
"""
    run = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code, bundle],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("buffer", [1, 31, 8192])
@pytest.mark.parametrize("with_notes", [False, True])
def test_unmodified_juce_demo_mixes_input_and_notes_like_native_host(
    tmp_path, rate, buffer, with_notes
):
    configured = os.environ.get("PEDALBOARD_TEST_JUCE_SIDECHAIN_BUILD")
    if not configured:
        pytest.skip("Optional combined-input JUCE example build is not configured")
    build = Path(configured).resolve()
    bundles = list((build / "CombinedAcceptance_artefacts/Release/VST3").glob("*.vst3"))
    assert bundles, "Configure the optional build with PEDALBOARD_BUILD_JUCE_COMBINED=ON"
    native = build / "NativeSidechainHost_artefacts/Release/NativeSidechainHost"
    if platform.system() == "Windows":
        native = native.with_suffix(".exe")
    output = tmp_path / "mixed.f32"
    run = subprocess.run(
        [
            str(native),
            str(bundles[0]),
            str(output),
            str(rate),
            str(buffer),
            str(int(with_notes)),
            "combined",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    expected_audio = np.fromfile(output, np.float32).reshape(2, 2048)
    main = np.empty((2, 2048), np.float32)
    main[0], main[1] = 0.2, -0.1
    midi = notes(rate, [(7, True), (511, False)]) if with_notes else []
    p = VST3Plugin(str(bundles[0]))
    p._get_parameter("Gain").raw_value = 1
    p._get_parameter("Delay Feedback").raw_value = 0
    audio, returned = p.process_buses(
        {0: main},
        rate,
        output_buses={0: 2},
        midi_messages=midi,
        return_midi=True,
        buffer_size=buffer,
    )
    assert p.midi_capabilities == {"accepts_input": True, "produces_output": True}
    assert_array_equal(audio[0], expected_audio)
    assert returned == midi
    if with_notes:
        assert_array_equal(audio[0][:, :7], main[:, :7])
        assert np.max(np.abs(audio[0] - main)) > 0.01
    else:
        assert_array_equal(audio[0], main)


@pytest.mark.parametrize("rate", [44100, 48000])
def test_unmodified_demo_reset_clears_held_voices(rate):
    configured = os.environ.get("PEDALBOARD_TEST_JUCE_SIDECHAIN_BUILD")
    if not configured:
        pytest.skip("Optional combined-input JUCE example build is not configured")
    bundle = next((Path(configured) / "CombinedAcceptance_artefacts/Release/VST3").glob("*.vst3"))
    p = VST3Plugin(str(bundle.resolve()))
    p._get_parameter("Delay Feedback").raw_value = 0
    main = np.zeros((2, 2048), np.float32)
    first = p.process_buses(
        {0: main}, rate, output_buses={0: 2}, midi_messages=notes(rate, [(0, True)])
    )[0]
    continued = p.process_buses({0: main}, rate, output_buses={0: 2}, reset=False)[0]
    reset = p.process_buses({0: main}, rate, output_buses={0: 2}, reset=True)[0]
    assert np.max(np.abs(first)) > 0.01
    assert np.max(np.abs(continued)) > 0.01
    assert_array_equal(reset, main)


def test_new_and_omitted_curve_lanes_retain_native_targets(host):
    fmt, create = host
    p = create()
    main, key = signals(key_channels=1)
    p.process_buses(
        {0: main, 1: key},
        48000,
        output_buses={0: 2},
        midi_messages=notes(48000, [(0, True)]),
        automation={"gain_db": np.linspace(-12, 3, 259)},
    )
    target = p.process_buses(
        {0: main, 1: key},
        48000,
        output_buses={0: 2},
        reset=False,
        automation={"frequency_hz": np.full(259, 5000.0)},
    )[0]
    held = p.process_buses({0: main, 1: key}, 48000, output_buses={0: 2}, reset=False)[0]
    gated = main * np.where(key[:1] > 0.5, np.float32(0.25), np.float32(1))
    wanted = (
        expected(gated, np.full(259, 3.0), np.full(259, 5000.0))
        if fmt == "vst3"
        else (gated.astype(np.float64) * 10 ** (3 / 20) / 4).astype(np.float32)
    )
    np.testing.assert_allclose(target, wanted, atol=3e-7, rtol=3e-7)
    assert_array_equal(held, target)
    if fmt == "vst3":
        with pytest.raises(ValueError, match="Reset"):
            p.automation_mode = "vst3_events"
