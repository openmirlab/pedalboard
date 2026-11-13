"""Real AUv3 musical/transport blocks observed from its remote render loop."""

import json
import os
import sys

import numpy as np
from numpy.testing import assert_array_equal

from pedalboard import AudioUnitPlugin
from tests.test_hosted_transport import (
    check_joint_clock_controls_and_midi,
    check_large_origin,
    check_many_wraps,
    decode_position,
    render,
)


def check(path, case):
    kind, rate, buffer, *args = case.split("/")
    rate, buffer = int(rate), int(buffer)
    label = {
        "tempo": "Clock",
        "loop": "Clock",
        "fractional": "Position",
        "position": "Position",
        "meter": "Meter",
        "bounds": "Loop",
        "changed": "Loop",
        "joint": "Clock",
        "many": "Position",
        "origin": "Position",
    }[kind]
    p = AudioUnitPlugin(path, plugin_name="AUv3 Automation " + label)
    pid = int(float(p._get_parameter("Render PID").string_value))
    assert pid > 0 and pid != os.getpid()
    if kind == "tempo":
        playing, ppq = bool(int(args[0])), float(args[1])
        clock = {
            "bpm": 90.0,
            "ppq_position": ppq,
            "sample_position": -1234,
            "playing": playing,
            "recording": True,
            "time_signature": (7, 8),
        }
        output = render(p, clock, rate=rate, buffer=buffer)
        beat = ppq + (np.arange(259) if playing else np.zeros(259)) * 90 / (60 * rate)
        assert_array_equal(output[0][0], np.full(259, np.float32(0.09)))
        np.testing.assert_allclose(output[0][1], beat / 1000, atol=1e-9, rtol=2e-7)
        assert_array_equal(
            output[1][0],
            np.full(259, np.float32(((1 if playing else 0) + 2 + 8 + 16 + 32 + 128) / 256)),
        )
        np.testing.assert_allclose(output[1][1], np.sin(beat * 2 * np.pi), atol=3e-7, rtol=3e-7)
        default = render(p, None, rate=rate, buffer=buffer, reset=False)
        assert_array_equal(default[0][0], np.full(259, np.float32(0.12)))
        assert_array_equal(default[0][1], np.zeros(259))
    elif kind == "position":
        start = int(args[0])
        output = render(
            p, {"sample_position": start, "ppq_position": 0.0}, rate=rate, buffer=buffer
        )
        assert_array_equal(decode_position(output), start + np.arange(259))
    elif kind == "loop":
        step = 120 / (60 * rate)
        clock = {
            "bpm": 120.0,
            "ppq_position": 235 * step,
            "sample_position": 235,
            "looping": True,
            "loop_start_ppq": 0.0,
            "loop_end_ppq": 240 * step,
        }
        output = render(p, clock, rate=rate, n=1001, buffer=buffer)
        expected = ((235 + np.arange(1001)) % 240) * step / 1000
        np.testing.assert_allclose(output[0][1], expected, atol=1e-10, rtol=2e-7)
    elif kind == "fractional":
        step = 120 / (60 * rate)
        clock = {
            "bpm": 120.0,
            "sample_position": 2,
            "ppq_position": 3.5 * step,
            "looping": True,
            "loop_start_ppq": 0.0,
            "loop_end_ppq": 4.5 * step,
        }
        output = render(p, clock, rate=rate, n=103, buffer=buffer)
        expected = np.floor(2 + np.remainder(3.5 + np.arange(103), 4.5) - 3.5 + 0.5).astype(
            np.int64
        )
        expected[0] = 2
        assert_array_equal(decode_position(output), expected)
    elif kind == "meter":
        clock = {
            "bpm": 90.0,
            "ppq_position": -2.25,
            "sample_position": -99,
            "time_signature": (7, 8),
            "playing": False,
        }
        output = render(p, clock, rate=rate, buffer=buffer)
        assert_array_equal(output[0][0], np.full(259, np.float32(7 / 32)))
        assert_array_equal(output[0][1], np.full(259, np.float32(8 / 32)))
        np.testing.assert_allclose(output[1][0], -3.5 / 1000, atol=1e-9, rtol=0)
        assert_array_equal(output[1][1] * 65536, np.full(259, rate // 6))
    elif kind == "bounds":
        output = render(
            p,
            {
                "playing": False,
                "looping": True,
                "loop_start_ppq": -4.0,
                "loop_end_ppq": -1.0,
                "ppq_position": -2.0,
            },
            rate=rate,
            buffer=buffer,
        )
        assert_array_equal(output[0][0], np.full(259, np.float32(-0.004)))
        assert_array_equal(output[0][1], np.full(259, np.float32(-0.001)))
    elif kind == "origin":
        check_large_origin(p, int(args[0]), buffer)
    elif kind == "many":
        check_many_wraps(p, buffer)
    elif kind == "joint":
        check_joint_clock_controls_and_midi(p)
    elif kind == "changed":
        first = render(p, {"bpm": 120.0}, rate=rate, n=31, buffer=31)
        assert_array_equal(first[1][0], np.ones(31))
        continued = render(
            p, {"bpm": 120.0, "sample_position": 31}, rate=rate, n=31, buffer=31, reset=False
        )
        assert_array_equal(continued[1][0], np.zeros(31))
        seek = render(
            p, {"bpm": 120.0, "sample_position": 300}, rate=rate, n=31, buffer=31, reset=False
        )
        assert_array_equal(seek[1][0], np.ones(31))
    return {"native_callbacks": True, "render_pid": pid, "host_pid": os.getpid(), "case": case}


if __name__ == "__main__":
    print(json.dumps({"case": sys.argv[2], "result": check(sys.argv[1], sys.argv[2])}), flush=True)
