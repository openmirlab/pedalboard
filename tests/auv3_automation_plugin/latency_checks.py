"""Fresh-process checks of actual remote AUv3 instrument compensation."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from pedalboard import AudioUnitPlugin
from tests.test_instrument_latency import length, notes


def load(path, latency, bypass=True):
    label = {
        0: "Instrument",
        20: "Delayed Instrument",
        4097: "Long Delayed Instrument",
        16385: "Huge Delayed Instrument",
    }[latency]
    p = AudioUnitPlugin(str(path), plugin_name="AUv3 Automation " + label)
    p._get_parameter("Pass Through").raw_value = float(bypass)
    return p


def check(path, case):
    kind, latency, *args = case.split("/")
    latency = int(latency)
    p = load(path, latency, bypass=kind != "curves")
    if kind == "identity":
        # Identity is emitted from the final render loop, after its delay line.
        y = p(
            [],
            duration=0.001,
            sample_rate=48000,
            compensate_latency=True,
            automation={"identity_probe": np.ones(48, dtype=bool)},
        )
        pid = int(y[0, 0])
        assert pid > 0 and pid != os.getpid()
        assert pid == int(float(p._get_parameter("Render PID").string_value))
        assert_array_equal(y, np.full_like(y, pid))
        assert p.reported_latency_samples == latency
        return {"host_pid": os.getpid(), "render_pid": pid, "latency": latency}
    if kind == "onset":
        rate, buffer, channels = map(int, args)
        duration, n = length(257, rate)
        midi = notes(
            rate, (0, True), (3, False), (7, True), (45, False), (n - 2, True), (n + 1, False)
        )
        kwargs = {
            "duration": duration,
            "sample_rate": rate,
            "buffer_size": buffer,
            "num_channels": channels,
        }
        y = p(midi, **kwargs, compensate_latency=True)
        expected = np.zeros((channels, n), np.float32)
        expected[:, :3] = expected[:, 7:45] = expected[:, -2:] = 1
        assert_array_equal(y, expected)
        assert p.reported_latency_samples == latency
        actual_frames = int(float(p._get_parameter("Render Frames").string_value))
        capacity = int(float(p._get_parameter("Render Capacity").string_value))
        assert 0 < actual_frames <= buffer
        assert capacity >= 8192
        raw = p(midi, **kwargs)
        # Raw defaults retain the actual delay; comparison excludes late notes
        # that cannot reach this requested raw interval.
        raw_expected = np.zeros_like(expected)
        if latency < n:
            raw_expected[:, latency:] = expected[:, : n - latency]
        assert_array_equal(raw, raw_expected)
        return {
            "rate": rate,
            "buffer": buffer,
            "channels": channels,
            "latency": latency,
            "frames": n,
            "exact": True,
            "last_sample": float(y[0, -1]),
            "render_frames": actual_frames,
            "prepared_capacity": capacity,
        }
    if kind == "curves":
        buffer = int(args[0])
        rate = 48000
        duration, n = length(257, rate)
        reference = load(path, 0, bypass=False)
        controls = {"gain_db": np.linspace(-30, 6, n), "frequency_hz": np.geomspace(20, 20000, n)}
        midi = notes(rate, (0, True), (31, False), (64, True))
        kwargs = {
            "duration": duration,
            "sample_rate": rate,
            "buffer_size": buffer,
            "automation": controls,
        }
        expected = reference(midi, **kwargs)
        y = p.process(midi, **kwargs, compensate_latency=True)
        assert_array_equal(y, expected)
        for name in ["Gain", "Frequency"]:
            assert p._get_parameter(name).raw_value == reference._get_parameter(name).raw_value
        assert p._get_parameter("Low Address Decoy").raw_value == 0
        return {
            "latency": latency,
            "buffer": buffer,
            "exact": True,
            "max_abs_error": float(np.max(np.abs(y - expected))),
        }
    if kind == "late":
        duration, n = length(257, 48000)
        y = p(
            notes(48000, (n + 1, True)),
            duration=duration,
            sample_rate=48000,
            compensate_latency=True,
        )
        # A wrongly consumed late note only becomes audible after the returned
        # compensated interval; the following raw state probe exposes it.
        after = p([], duration=0.01, sample_rate=48000, reset=False)
        assert not np.any(y) and not np.any(after)
        return {"late_midi_excluded": True}
    if kind == "reset":
        midi = notes(48000, (7, True), (199, False))
        kwargs = {
            "duration": 0.01,
            "sample_rate": 48000,
            "buffer_size": 31,
            "compensate_latency": True,
        }
        a, b = p(midi, **kwargs), p(midi, **kwargs)
        assert np.any(a)
        assert_array_equal(a, b)
        empty = p([], duration=0, sample_rate=48000, compensate_latency=True)
        assert empty.shape == (2, 0)
        return {"reset_exact": True, "zero_frames": True}
    if kind == "reject":
        consumed = []

        def midi():
            consumed.append(True)
            yield (bytes([0x90, 60, 100]), 0)

        before = p._get_parameter("Instance ID").string_value
        try:
            p(midi(), duration=1, sample_rate=48000, reset=False, compensate_latency=True)
        except ValueError as error:
            assert "reset=True" in str(error)
        else:
            raise AssertionError("Compensation accepted a raw stream")
        assert not consumed
        assert p._get_parameter("Instance ID").string_value == before
        return {"unconsumed_midi": True, "unchanged_instance": True}
    if kind == "state":
        import tempfile

        from pedalboard import configuration

        names = {row["name"] for row in p._configuration_snapshot(False)["parameters"]}
        assert not names.intersection(
            {"Render Frames", "Render Capacity", "Render PID", "Instance ID"}
        )
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "instrument.json"
            configuration.save(p, target, sample_rate=48000)
            restored = configuration.load(target)
            result = configuration.verify(
                p,
                restored,
                sample_rate=48000,
                midi=notes(48000, (0, True), (199, False)),
                duration=0.01,
                exact=True,
            )
            assert result.exact
        return {"read_only_excluded": True, "restored_sound_exact": True}
    if kind == "stream":
        reference = load(path, latency)
        for midi in [notes(48000, (7, True)), [], notes(48000, (11, False))]:
            kwargs = {"duration": 0.01, "sample_rate": 48000, "reset": False, "buffer_size": 31}
            assert_array_equal(
                p(midi, **kwargs, compensate_latency=False), reference(midi, **kwargs)
            )
        return {"raw_stream_exact": True}
    raise AssertionError(case)


if __name__ == "__main__":
    result = check(Path(sys.argv[1]).resolve(), sys.argv[2])
    print(json.dumps({"case": sys.argv[2], "result": result}), flush=True)
