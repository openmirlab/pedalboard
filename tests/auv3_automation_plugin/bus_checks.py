"""Fresh-process acceptance of actual out-of-process AUv3 audio buses."""

import json
import os
import sys

import numpy as np
from numpy.testing import assert_array_equal

from pedalboard import AudioUnitPlugin


def check(path, case):
    p = AudioUnitPlugin(
        path,
        plugin_name="AUv3 Automation Bus Identity"
        if case == "identity"
        else "AUv3 Automation Buses",
    )
    if case == "identity":
        buses = p.process_buses(
            {0: np.ones((2, 259), np.float32), 1: np.ones((2, 259), np.float32)},
            48000,
            output_buses={0: 2, 1: 2},
            buffer_size=31,
            reset=False,
        )
        pid = int(buses[0][0, 0])
        assert pid > 0 and pid != os.getpid(), {
            "first": buses[0][:, :3].tolist(),
            "frames": p._get_parameter("Render Frames").string_value,
            "capacity": p._get_parameter("Render Capacity").string_value,
            "identity": p._get_parameter("Identity Probe").raw_value,
            "buses": p.audio_buses,
        }
        assert pid == int(float(p._get_parameter("Render PID").string_value))
        for index, audio in buses.items():
            assert_array_equal(
                audio,
                np.full_like(audio, pid),
                err_msg=str(
                    {
                        "bus": index,
                        "mask": p._get_parameter("Render Bus Mask").string_value,
                        "frames": p._get_parameter("Render Frames").string_value,
                        "ports": p.audio_buses,
                    }
                ),
            )
        return {"render_pid": pid, "host_pid": os.getpid(), "both_buses_exact": True}
    rate, size, channels, key_channels, aux_channels = map(int, case.split("/"))
    n = 259
    main = np.random.default_rng(73).uniform(-0.4, 0.4, (channels, n)).astype(np.float32)
    key = np.zeros((key_channels, n), np.float32)
    key[:, 31:129] = 1
    kwargs = {
        "sample_rate": rate,
        "output_buses": {0: channels, 1: aux_channels},
        "buffer_size": size,
    }
    expected = main * np.where(key[:1] > 0.5, np.float32(0.25), np.float32(1))
    output = p.process_buses({0: main, 1: key}, **kwargs)
    assert_array_equal(output[0], expected)
    assert_array_equal(output[1], main[np.arange(aux_channels) % channels] * 0.5)
    pieces = [[], []]
    for start, end in [(0, 32), (32, 259)]:
        part = p.process_buses(
            {0: main[:, start:end], 1: key[:, start:end]}, **kwargs, reset=start == 0
        )
        for index, chunks in enumerate(pieces):
            chunks.append(part[index])
    for index, chunks in enumerate(pieces):
        assert_array_equal(np.concatenate(chunks, axis=1), output[index])
    raw = p.process_buses({0: main}, rate, output_buses={0: channels}, buffer_size=size)
    assert_array_equal(raw[0], main)
    assert p.audio_buses["inputs"][1]["enabled"]
    assert int(float(p._get_parameter("Render Capacity").string_value)) >= 8192
    return {
        "exact": True,
        "rate": rate,
        "buffer": size,
        "layout": [channels, key_channels, aux_channels],
    }


if __name__ == "__main__":
    print(json.dumps({"case": sys.argv[2], "result": check(sys.argv[1], sys.argv[2])}), flush=True)
