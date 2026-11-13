"""Configuration persistence exercises settings, topology, DSP and fresh processes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pedalboard as pb
from pedalboard import configuration as c


def probe(rate=48000, channels=2):
    rng = np.random.default_rng(17)
    return rng.normal(0, 0.12, (channels, rate // 2)).astype(np.float32)


@pytest.mark.parametrize("name", sorted(c._PROPERTIES))
@pytest.mark.parametrize("rate", [44100, 48000])
def test_builtin_roundtrip(name, rate, tmp_path):
    plugin = getattr(pb, name)()
    # Exercise non-default values without assuming common units/ranges.
    for key in c._PROPERTIES[name].split():
        value = getattr(plugin, key)
        enum = c._ENUMS.get((name, key))
        if enum:
            setattr(plugin, key, list(enum.__members__.values())[1])
        elif key in ("ratio", "drive"):
            setattr(plugin, key, 3.0)
        elif isinstance(value, bool):
            setattr(plugin, key, not value)
        else:
            setattr(plugin, key, value * 0.9 if value else 0.1)
    path = tmp_path / "configuration.json"
    c.save(plugin, path, sample_rate=rate, settle_seconds=0)
    restored = c.load(path, settle_seconds=0)
    assert c._encode_properties(plugin, name) == c._encode_properties(restored, name)
    result = c.verify(plugin, restored, probe(rate), sample_rate=rate, exact=True, settle_seconds=0)
    assert result.exact and result.max_abs_error == 0


def test_shared_nested_graph_and_fresh_process(tmp_path):
    gain = pb.Gain(-3)
    graph = pb.Pedalboard(
        [
            pb.Compressor(-18, 3),
            pb.Mix([pb.Chain([gain, pb.Delay(0.02)]), gain]),
            pb.Reverb(room_size=0.3),
        ]
    )
    path = tmp_path / "graph.json"
    c.save(graph, path, sample_rate=48000, settle_seconds=0)
    loaded = c.load(path, settle_seconds=0)
    assert loaded[1][0][0] is loaded[1][1]
    # Preserve aliases in files, but the existing renderer rejects duplicate nodes.
    graph[1][1] = pb.Gain(-3)
    c.save(graph, path, sample_rate=48000, settle_seconds=0)
    loaded = c.load(path, settle_seconds=0)
    assert c.verify(graph, loaded, probe(), sample_rate=48000, exact=True, settle_seconds=0).exact
    np.save(tmp_path / "input.npy", probe())
    graph.reset()
    expected = graph(probe(), 48000)
    np.save(tmp_path / "expected.npy", expected)
    code = """
import sys
from pathlib import Path
import numpy as np
from pedalboard import configuration as c
p = Path(sys.argv[1])
graph = c.load(p / "graph.json", settle_seconds=0)
assert graph[1][0][0] is not graph[1][1]
np.testing.assert_array_equal(graph(np.load(p / "input.npy"), 48000),
                              np.load(p / "expected.npy"))
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True)


def test_save_preserves_running_delay_tail(tmp_path):
    a, b = [pb.Delay(0.01, feedback=0.6, mix=1) for _ in range(2)]
    impulse = np.zeros((2, 256), np.float32)
    impulse[:, 0] = 1
    for plugin in (a, b):
        plugin(impulse, 48000, buffer_size=256, reset=True)
    c.save(a, tmp_path / "delay.json", sample_rate=48000, settle_seconds=0)
    silence = np.zeros((2, 256), np.float32)
    aa = np.concatenate([a(silence, 48000, buffer_size=256, reset=False) for _ in range(8)], axis=1)
    bb = np.concatenate([b(silence, 48000, buffer_size=256, reset=False) for _ in range(8)], axis=1)
    assert np.max(np.abs(aa)) > 0
    np.testing.assert_array_equal(aa, bb)


def test_atomic_failure_and_unsupported(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("previous")
    with pytest.raises(c.ConfigurationError, match="Unsupported processor"):
        c.save(
            pb.Convolution(np.ones(16, np.float32), sample_rate=48000),
            path,
            sample_rate=48000,
            settle_seconds=0,
        )
    assert path.read_text() == "previous"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema=2),
        lambda d: d.update(root=True),
        lambda d: d["runtime"].update(version="different"),
        lambda d: d["nodes"][0].update(type="eval"),
        lambda d: d["nodes"][0].update(properties={"gain_db": float("nan")}),
        lambda d: d["nodes"].append(d["nodes"][0]),
    ],
)
def test_invalid_documents(tmp_path, mutation):
    path = tmp_path / "bad.json"
    c.save(pb.Gain(), path, sample_rate=48000, settle_seconds=0)
    document = json.loads(path.read_text())
    mutation(document)
    path.write_text(json.dumps(document))
    with pytest.raises(c.ConfigurationError):
        c.load(path, settle_seconds=0)


def test_cycle_and_enum_validation(tmp_path):
    path = tmp_path / "bad.json"
    c.save(pb.Pedalboard([pb.Gain()]), path, sample_rate=48000, settle_seconds=0)
    data = json.loads(path.read_text())
    data["nodes"][0]["children"] = [0]
    path.write_text(json.dumps(data))
    with pytest.raises(c.ConfigurationError, match="Cyclic"):
        c.load(path, settle_seconds=0)
    c.save(pb.LadderFilter(), path, sample_rate=48000, settle_seconds=0)
    data = json.loads(path.read_text())
    data["nodes"][0]["properties"]["mode"] = "execute"
    path.write_text(json.dumps(data))
    with pytest.raises(c.ConfigurationError, match="enum"):
        c.load(path, settle_seconds=0)


def test_verify_detects_settings_and_hidden_audio_difference():
    with pytest.raises(c.ConfigurationError, match="configuration differs"):
        c.verify(pb.Gain(-3), pb.Gain(-6), probe(), sample_rate=48000, settle_seconds=0)
    with pytest.raises(c.ConfigurationError, match="nonempty finite"):
        c.verify(pb.Gain(), pb.Gain(), np.array([np.nan]), sample_rate=48000, settle_seconds=0)


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("instrument", [False, True])
@pytest.mark.parametrize("extension", ["vst3", "component"])
def test_external_fresh_process_roundtrip(tmp_path, rate, instrument, extension):
    if sys.platform != "darwin":
        pytest.skip("This integration matrix uses macOS fixture bundles")
    category = "instrument" if instrument else "effect"
    name = "Magical8bitPlug2" if instrument else "CHOWTapeModel"
    bundle = Path(__file__).parent / "plugins" / category / "Darwin" / f"{name}.{extension}"
    if not bundle.exists():
        pytest.skip(f"Missing fixture {bundle}")
    # Each save/load/render runs in a new interpreter; AU registration stays
    # process-local and never installs a plugin into the user's system.
    code = r"""
import json, sys
from pathlib import Path
import numpy as np
import pedalboard as pb
from pedalboard import configuration as c
from tests.au_automation_plugin.register import register_bundle
folder, bundle, mode, rate, instrument = sys.argv[1:]
folder, bundle, rate = Path(folder), Path(bundle), int(rate)
instrument = instrument == "True"
if bundle.suffix == ".component":
    register_bundle(bundle)
if mode == "write":
    p = pb.load_plugin(str(bundle))
    if instrument:
        p.parameters["attack"].raw_value = 0.05
    else:
        p.parameters["input_gain"].raw_value = 0.65
        p.parameters["output_gain"].raw_value = 0.35
    c.save(p, folder / "external.json", sample_rate=rate)
else:
    p = c.load(folder / "external.json")
# Prepare *before* restore even in writer reference: same fresh render format.
if mode == "write":
    state = p._configuration_snapshot()["state"]
    p._configuration_restore(state, rate, 2)
    c._settle(0.2)
if instrument:
    audio = p([(bytes([0x90,60,100]),0.0), (bytes([0x80,60,0]),0.3)],
              duration=0.5, sample_rate=rate, reset=True)
else:
    x = np.random.default_rng(3).normal(0,0.12,(2,rate//2)).astype(np.float32)
    p(np.zeros((2,rate),np.float32),rate,reset=True)
    audio = p(x,rate,reset=False)
assert np.max(np.abs(audio)) > 0.001
if mode != "write":
    q = c.load(folder / "external.json")
    if instrument:
        report = c.verify(p, q, sample_rate=rate,
            midi=[(bytes([0x90,60,100]),0.0), (bytes([0x80,60,0]),0.3)], duration=0.5)
    else:
        report = c.verify(p, q, x, sample_rate=rate)
    # A -> B -> A must return settings and the same fresh-render sound.
    state_a = p._configuration_snapshot()
    parameter = "attack" if instrument else "input_gain"
    p.parameters[parameter].raw_value = 0.85
    c._settle(0.2)
    assert abs(p.parameters[parameter].raw_value - 0.85) < 0.01
    p._configuration_restore(state_a["state"], rate, 2)
    c._settle(0.2)
    c._parameters_match(state_a["parameters"], p._configuration_snapshot(False)["parameters"])
    if instrument:
        aba = c.verify(p, q, sample_rate=rate,
            midi=[(bytes([0x90,60,100]),0.0), (bytes([0x80,60,0]),0.3)], duration=0.5)
    else:
        aba = c.verify(p, q, x, sample_rate=rate)
if mode == "write":
    np.save(folder / "reference.npy", audio)
else:
    reference = np.load(folder / "reference.npy")
    np.testing.assert_allclose(audio, reference, atol=1e-6, rtol=1e-5)
    (folder / "measurement.json").write_text(json.dumps({
        "max_abs_error": float(np.max(np.abs(audio-reference))),
        "exact": bool(np.array_equal(audio,reference)),
        "verify_max_abs_error": report.max_abs_error,
        "aba_max_abs_error": aba.max_abs_error}))
"""
    for mode in ("write", "read", "read"):
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(tmp_path),
                str(bundle.resolve()),
                mode,
                str(rate),
                str(instrument),
            ],
            check=True,
            timeout=60,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        )


@pytest.fixture
def external_document(tmp_path):
    bundle = Path(__file__).parent / "plugins/effect/Darwin/CHOWTapeModel.vst3"
    if sys.platform != "darwin" or not bundle.exists():
        pytest.skip("Requires the CHOW macOS fixture")
    plugin = pb.VST3Plugin(str(bundle.resolve()))
    plugin.parameters["input_gain"].raw_value = 0.65
    path = tmp_path / "external.json"
    c.save(plugin, path, sample_rate=48000)
    return path


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda d: d["nodes"][0].update(state_sha256="bad"), "checksum"),
        (lambda d: d["nodes"][0].update(bundle_sha256="bad"), "bundle changed"),
        (lambda d: d["nodes"][0].update(path="/no/such/plugin.vst3"), "missing"),
        (lambda d: d["nodes"][0]["parameters"][0].update(raw=2), "normalized"),
        (lambda d: d["nodes"][0].update(state="!!invalid!!"), "encoding"),
    ],
)
def test_external_preflight_precedes_native_loading(
    external_document, monkeypatch, mutation, message
):
    document = json.loads(external_document.read_text())
    mutation(document)
    external_document.write_text(json.dumps(document))

    def forbidden(*args, **kwargs):
        pytest.fail("Native constructor called before complete preflight")

    monkeypatch.setattr(pb, "VST3Plugin", forbidden)
    with pytest.raises(c.ConfigurationError, match=message):
        c.load(external_document, settle_seconds=0)


def test_failed_opaque_restore_is_not_hidden_by_parameter_setters(external_document, monkeypatch):
    monkeypatch.setattr(pb.VST3Plugin, "_configuration_restore", lambda *args: None)
    with pytest.raises(c.ConfigurationError, match="Restored parameter differs"):
        c.load(external_document)


def test_parameter_match_quantization_requires_native_parser():
    parameter = {
        "id": "1",
        "name": "Attack",
        "label": "s",
        "text": "0.012",
        "raw": 0.05,
        "canonical": 0.048989795,
    }
    restored = {**parameter, "raw": 0.048989795}
    c._parameters_match([parameter], [restored])
    with pytest.raises(c.ConfigurationError, match="Restored parameter differs"):
        c._parameters_match([{**parameter, "canonical": None}], [restored])
    with pytest.raises(c.ConfigurationError, match="Restored parameter differs"):
        c._parameters_match([parameter], [{**restored, "text": "0.020"}])


def test_audio_mismatch_is_independent_of_parameter_checks(monkeypatch):
    # Simulate hidden DSP differences that are absent from parameter metadata.
    monkeypatch.setattr(c, "_compare_nodes", lambda *args: None)
    with pytest.raises(c.ConfigurationError, match="Sound differs: max_abs_error="):
        c.verify(pb.Gain(-3), pb.Gain(-6), probe(), sample_rate=48000, settle_seconds=0)


def test_builtin_preflight_before_any_external_constructor(external_document, monkeypatch):
    document = json.loads(external_document.read_text())
    document["nodes"].extend(
        [
            {
                "kind": "builtin",
                "type": "Compressor",
                "properties": {"threshold_db": -12, "ratio": -1, "attack_ms": 1, "release_ms": 100},
            },
            {"kind": "container", "type": "Pedalboard", "children": [0, 1]},
        ]
    )
    document["root"] = 2
    external_document.write_text(json.dumps(document))

    def forbidden(*args, **kwargs):
        pytest.fail("External constructor called before builtin property validation")

    monkeypatch.setattr(pb, "VST3Plugin", forbidden)
    with pytest.raises(c.ConfigurationError, match="Invalid builtin"):
        c.load(external_document, settle_seconds=0)


def test_builtin_workflow_does_not_require_message_thread(tmp_path):
    import concurrent.futures

    def work():
        path = tmp_path / "thread.json"
        a = pb.Delay(0.01)
        c.save(a, path, sample_rate=48000)
        b = c.load(path)
        return c.verify(a, b, probe(), sample_rate=48000, exact=True)

    with concurrent.futures.ThreadPoolExecutor(1) as executor:
        assert executor.submit(work).result(timeout=10).exact


def test_actual_auv3_configuration_fresh_processes(tmp_path):
    if sys.platform != "darwin":
        pytest.skip("AUv3 requires macOS")
    from tests.auv3_automation_plugin.runtime import registered_extension

    code = r"""
import sys
from pathlib import Path
import numpy as np
import pedalboard as pb
from pedalboard import configuration as c
bundle, folder, name, mode = sys.argv[1:]
folder = Path(folder)
if mode == "write":
    p = pb.AudioUnitPlugin(bundle, plugin_name=name)
    p._get_parameter("Gain").raw_value = 0.5
    p._get_parameter("Frequency").raw_value = 0.25
    c.save(p, folder / "au.json", sample_rate=48000)
else:
    p = c.load(folder / "au.json")
q = c.load(folder / "au.json")
if "Instrument" in name:
    c.verify(p, q, sample_rate=48000, midi=[([144,60,100],0),([128,60,0],0.01)],
             duration=0.02, exact=True)
    audio = p([([144,60,100],0),([128,60,0],0.01)],0.02,48000)
else:
    x = np.random.default_rng(83).normal(0,0.1,(2,1024)).astype(np.float32)
    c.verify(p, q, x, sample_rate=48000, exact=True)
    audio = p(x,48000)
assert np.max(np.abs(audio)) > 0.001
if mode == "write":
    np.save(folder / "output.npy",audio)
else:
    np.testing.assert_array_equal(audio,np.load(folder / "output.npy"))
"""
    with registered_extension(tmp_path / "build") as bundle:
        for index, name in enumerate(
            ["AUv3 Automation Probe", "AUv3 Automation Delayed", "AUv3 Automation Instrument"]
        ):
            folder = tmp_path / str(index)
            folder.mkdir()
            for mode in ("write", "read"):
                result = subprocess.run(
                    [sys.executable, "-c", code, str(bundle), str(folder), name, mode],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                assert result.returncode == 0, result.stdout + result.stderr


def test_verification_rechecks_settings_after_reset(monkeypatch):
    original_reset = pb.Gain.reset

    def bad_reset(plugin):
        original_reset(plugin)
        plugin.gain_db = -6

    monkeypatch.setattr(pb.Gain, "reset", bad_reset)
    with pytest.raises(c.ConfigurationError, match="configuration differs"):
        c.verify(pb.Gain(-3), pb.Gain(-3), probe(), sample_rate=48000, settle_seconds=0)


def test_external_load_failure_is_explicit(external_document, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ImportError("No compatible plug-in format exists for this plug-in")

    monkeypatch.setattr(pb, "VST3Plugin", unavailable)
    with pytest.raises(c.ConfigurationError, match="Restoring configuration failed: No compatible"):
        c.load(external_document, settle_seconds=0)
