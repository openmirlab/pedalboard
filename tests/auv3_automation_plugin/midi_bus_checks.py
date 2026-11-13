"""Actual remote AUv3 combined audio, sidechain, note and control timing."""

import json
import sys

import numpy as np
from numpy.testing import assert_array_equal

from pedalboard import AudioUnitPlugin
from tests.test_hosted_audio_buses import signals
from tests.test_hosted_audio_midi import mask, notes


def check(path, case):
    rate, buffer, delay, pattern = case.split("/")
    rate, buffer, delay = int(rate), int(buffer), int(delay)
    p = AudioUnitPlugin(
        path, plugin_name="AUv3 Automation " + ("Delayed " if delay else "") + "Audio MIDI"
    )
    main, key = signals(key_channels=1)
    gain, frequency = np.full(259, -6.0), np.full(259, 10000.0)
    if pattern == "dense":
        gain, frequency = np.linspace(-30, 6, 259), np.geomspace(20, 20000, 259)
    if pattern == "controller":
        gain, frequency = np.zeros(259), np.full(259, 20000.0)
    controls = {"gain_db": gain, "frequency_hz": frequency}
    all_midi = notes(rate)
    if pattern == "controller":
        all_midi += [(bytes([0xB0, 1, 127]), 7 / rate), (bytes([0xB0, 1, 0]), 127 / rate)]
    output, returned = p.process_buses(
        {0: main, 1: key},
        rate,
        output_buses={0: 2, 1: 1},
        buffer_size=buffer,
        midi_messages=all_midi,
        automation=controls,
        return_midi=True,
    )
    gated = main * mask() * np.where(key[:1] > 0.5, np.float32(0.25), np.float32(1))
    wanted = (
        gated.astype(np.float64)
        * 10 ** (gain.astype(np.float32).astype(np.float64) / 20)
        * frequency.astype(np.float32).astype(np.float64)
        / 20000
    ).astype(np.float32)
    if pattern == "controller":
        wanted[:, 7:127] = main[:, 7:127]
    if delay:
        wanted = np.pad(wanted, ((0, 0), (20, 0)))[:, :259]
    np.testing.assert_allclose(output[0], wanted, atol=3e-7, rtol=3e-7)
    assert_array_equal(output[1], main[:1] * 0.5)
    assert p.midi_capabilities == {"accepts_input": True, "produces_output": False}
    assert returned == []
    pieces = []
    for start, end in [(0, 7), (7, 119), (119, 259)]:
        midi = [
            (data, (round(seconds * rate) - start) / rate)
            for data, seconds in all_midi
            if start <= round(seconds * rate) < end
        ]
        pieces.append(
            p.process_buses(
                {0: main[:, start:end], 1: key[:, start:end]},
                rate,
                output_buses={0: 2, 1: 1},
                buffer_size=buffer,
                reset=start == 0,
                midi_messages=midi,
                automation={"gain_db": gain[start:end], "frequency_hz": frequency[start:end]},
            )
        )
    for index in [0, 1]:
        assert_array_equal(np.concatenate([part[index] for part in pieces], axis=1), output[index])
    return {
        "exact_partition": True,
        "rate": rate,
        "buffer": buffer,
        "latency": p.reported_latency_samples,
        "pattern": pattern,
    }


if __name__ == "__main__":
    print(json.dumps({"case": sys.argv[2], "result": check(sys.argv[1], sys.argv[2])}), flush=True)
