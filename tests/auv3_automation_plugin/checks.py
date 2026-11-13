"""Bounded subprocess checks of actual remote AUv3 effects and instruments."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from pedalboard import AudioUnitPlugin, Chain, ExternalPluginReloadType, Gain, Mix, Pedalboard
from pedalboard_native._internal import AddLatency
from tests.test_au_automation import expected


def signal(n=257):
    return np.random.default_rng(327).uniform(-0.4, 0.4, (2, n)).astype(np.float32)


def targets(n):
    return np.linspace(-30, 6, n), np.geomspace(20, 20000, n)


def load(path, name="AUv3 Automation Probe"):
    return AudioUnitPlugin(str(path), plugin_name=name)


def render(p, x, rate=48000, **kwargs):
    g, f = targets(x.shape[-1])
    y = p(x, rate, automation={"gain_db": g, "frequency_hz": f}, **kwargs)
    np.testing.assert_allclose(y, expected(x, g, f), atol=3e-7, rtol=3e-7)
    # A genuine 64-bit Frequency address shares its low32 bits with a distinct
    # boolean. Controlling Frequency must not silently truncate into that knob.
    assert p._get_parameter("Low Address Decoy").raw_value == 0
    return y


def check(path, case):
    if case.startswith("discovery"):
        supplied = str(path)
        if case.endswith("relative"):
            supplied = os.path.relpath(path)
        elif case.endswith("trailing"):
            supplied += "/"
        elif case.endswith("symlink"):
            alias = path.parents[3] / "AUv3 音訊 % Shortcut.appex"
            alias.symlink_to(path, target_is_directory=True)
            supplied = str(alias)
        names = AudioUnitPlugin.get_plugin_names_for_file(supplied)
        assert sorted(names) == sorted(
            [
                "AUv3 Automation Probe",
                "AUv3 Automation Delayed",
                "AUv3 Automation Instrument",
                "AUv3 Automation Delayed Instrument",
                "AUv3 Automation Long Delayed Instrument",
                "AUv3 Automation Huge Delayed Instrument",
                "AUv3 Automation Buses",
                "AUv3 Automation Bus Identity",
                "AUv3 Automation Audio MIDI",
                "AUv3 Automation Delayed Audio MIDI",
            ]
        )
        return {"names": names}
    if case == "unregistered":
        fake = path.parents[3] / "Unregistered Audio.appex"
        fake.mkdir()
        try:
            AudioUnitPlugin.get_plugin_names_for_file(str(fake))
        except ImportError as error:
            assert "registered" in str(error)
            assert "/Library/Audio/Plug-Ins/Components/" not in str(error)
        else:
            raise AssertionError("An unregistered extension unexpectedly loaded")
        return {"unregistered_rejected": True}
    if case == "identity":
        p = load(path)
        y = p(
            np.zeros((2, 17), np.float32),
            48000,
            automation={"identity_probe": np.ones(17, dtype=bool)},
        )
        actual_pid = int(y[0, 0])
        assert actual_pid > 0 and actual_pid != os.getpid()
        assert_array_equal(y, np.full_like(y, actual_pid))
        assert actual_pid == int(float(p._get_parameter("Render PID").string_value))
        import ctypes as ct
        import plistlib

        metadata = plistlib.loads((path / "Contents/Info.plist").read_bytes())
        item = metadata["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0]

        class Description(ct.Structure):
            _fields_ = [
                (key, ct.c_uint32) for key in ("type", "subtype", "manufacturer", "flags", "mask")
            ]

        desc = Description(
            *[
                int.from_bytes(item[key].encode(), "big")
                for key in ("type", "subtype", "manufacturer")
            ],
            0,
            0,
        )
        api = ct.CDLL("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")
        api.AudioComponentFindNext.argtypes = [ct.c_void_p, ct.POINTER(Description)]
        api.AudioComponentFindNext.restype = ct.c_void_p
        api.AudioComponentGetDescription.argtypes = [ct.c_void_p, ct.POINTER(Description)]
        component = api.AudioComponentFindNext(None, ct.byref(desc))
        assert component and api.AudioComponentGetDescription(component, ct.byref(desc)) == 0
        assert desc.flags & 4  # kAudioComponentFlag_IsV3AudioUnit
        return {"host_pid": os.getpid(), "render_pid": actual_pid, "component_flags": desc.flags}
    if case.startswith("audio/"):
        _, name, rate, buffer, layout = case.split("/")
        p = load(path, "AUv3 Automation Delayed" if name == "delayed" else "AUv3 Automation Probe")
        x = signal()
        if layout == "mono":
            x = x[0]
        if layout == "interleaved":
            g, f = targets(x.shape[1])
            y = p(
                x.T,
                int(rate),
                buffer_size=int(buffer),
                automation={"gain_db": g, "frequency_hz": f},
            )
            np.testing.assert_allclose(y, expected(x, g, f).T, atol=3e-7, rtol=3e-7)
        else:
            y = render(p, x, int(rate), buffer_size=int(buffer))
        assert p.reported_latency_samples == (20 if name == "delayed" else 0)
        return {"shape": list(y.shape)}
    if case.startswith("midi/"):
        _, rate, buffer = case.split("/")
        rate, buffer = int(rate), int(buffer)
        p = load(path, "AUv3 Automation Instrument")
        duration = 0.03125
        n = int(duration * rate)
        g, f = targets(n)
        mask = np.zeros(n)
        notes = []
        for start, end in [(7, 45), (127, n - 23)]:
            notes.extend([([144, 60, 127], start / rate), ([128, 60, 0], end / rate)])
            mask[start:end] = 1
        y = p(
            notes, duration, rate, buffer_size=buffer, automation={"gain_db": g, "frequency_hz": f}
        )
        np.testing.assert_allclose(y, expected(np.tile(mask, (2, 1)), g, f), atol=3e-7, rtol=3e-7)
        assert not p([], duration, rate, reset=False, buffer_size=buffer).any()
        return {"note_frames": int(np.count_nonzero(y[0]))}
    if case == "midi-hold":
        p = load(path, "AUv3 Automation Instrument")
        n = 1500
        g, f = targets(n)
        y = p([([144, 60, 127], 0)], 0.03125, 48000, automation={"gain_db": g, "frequency_hz": f})
        held = p([], 0.03125, 48000, reset=False, buffer_size=31)
        assert_array_equal(held, np.repeat(y[:, -1:], n, axis=1))
        with np.testing.assert_raises(ValueError):
            Pedalboard([p])
        return {"held": True}
    if case.startswith("graph/"):
        parallel = case.endswith("parallel")
        x = signal(1025)
        g, f = targets(1025)

        def run(sliced):
            p = load(path, "AUv3 Automation Delayed")
            branch = Chain([AddLatency(701), p, Gain(-3)])
            board = Pedalboard([Mix([branch, Gain(-6)])]) if parallel else Pedalboard([branch])
            board.reset()
            parts = []
            start = 0
            for end in [13, 511, 1025] if sliced else [1025]:
                parts.append(
                    board(
                        x[:, start:end],
                        48000,
                        reset=False,
                        automation={p: {"gain_db": g[start:end], "frequency_hz": f[start:end]}},
                    )
                )
                start = end
            parts.append(board(np.zeros((2, 1600), np.float32), 48000, reset=False))
            return np.concatenate(parts, axis=1)

        assert_array_equal(run(True), run(False))
        return {"partitions_exact": True}
    if case == "owned-pending":
        p = load(path)
        board = Pedalboard([AddLatency(701), p])
        x = signal(1025)
        g, f = targets(1025)
        reference_g = g.copy()
        reference_f = np.r_[f[:513], np.full(512, f[512])]
        board.reset()
        a = board(
            x[:, :513],
            48000,
            reset=False,
            automation={p: {"gain_db": g[:513], "frequency_hz": f[:513]}},
        )
        g[:513], f[:513] = 12, 20000
        b = board(x[:, 513:], 48000, reset=False, automation={p: {"gain_db": g[513:]}})
        c = board(np.zeros((2, 1600), np.float32), 48000, reset=False)
        y = np.concatenate([a, b, c], axis=1)[:, :1025]
        np.testing.assert_allclose(y, expected(x, reference_g, reference_f), atol=3e-7, rtol=3e-7)
        return {"owned": True}
    if case.startswith("reset/"):
        p = load(path)
        x = signal()
        reference = render(p, x)
        if case.endswith("reload"):
            type(p)._reload_type.__set__(p, ExternalPluginReloadType.PersistsAudioOnReset)
        before = (
            p._get_parameter("Render PID").string_value,
            p._get_parameter("Instance ID").string_value,
        )
        p.reset()
        actual = render(p, x)
        assert_array_equal(actual, reference)
        after = (
            p._get_parameter("Render PID").string_value,
            p._get_parameter("Instance ID").string_value,
        )
        if case.endswith("reload"):
            assert before != after
        return {"before": before, "after": after}
    if case == "validation":
        p = load(path)
        x = signal()
        render(p, x)
        before = p._get_parameter("Gain").raw_value
        for key, value in [
            ("gain_db", np.full(257, np.nan)),
            ("frequency_hz", np.full(257, 30000)),
            ("render_pid", np.zeros(257)),
            ("missing", np.zeros(257)),
        ]:
            with np.testing.assert_raises(ValueError):
                p(x, 48000, automation={"pass_through": np.ones(257, dtype=bool), key: value})
            assert p._get_parameter("Gain").raw_value == before
            assert p._get_parameter("Pass Through").raw_value == 0
        with np.testing.assert_raises(ValueError):
            p(x, 44100, reset=False)
        p.reset()
        render(p, x, 44100)
        return {"atomic": True}
    if case == "boolean":
        p = load(path)
        x = signal()
        mask = np.arange(257) % 2 == 0
        y = p(x, 48000, automation={"pass_through": mask})
        np.testing.assert_allclose(y, np.where(mask, x, x * 0.001), atol=3e-7, rtol=3e-7)
        assert p.automation_mode == "sample_steps"
        with np.testing.assert_raises(ValueError):
            p.automation_mode = "vst3_events"
        return {"boolean": True}
    if case == "state":
        p = load(path)
        x = signal()
        render(p, x)
        state = p.raw_state
        p.reset()
        p.raw_state = state
        assert abs(float(p._get_parameter("Gain").string_value) - 6) < 1e-4
        y = p(x, 48000)
        ref = expected(x, np.full(257, 6.0), np.full(257, 20000.0))
        np.testing.assert_allclose(y, ref, atol=3e-7, rtol=3e-7)
        return {"state_bytes": len(state)}
    raise AssertionError(case)


if __name__ == "__main__":
    extension = Path(sys.argv[1]).resolve()
    result = check(extension, sys.argv[2])
    print(json.dumps({"case": sys.argv[2], "result": result}), flush=True)
