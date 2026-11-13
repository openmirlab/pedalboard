"""Versioned configuration files, not snapshots of running DSP memory.

Only load files from trusted sources: external plugin state is passed to native
third-party code. See docs/source/configuration.rst for the persistence contract.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import pedalboard as pb
import pedalboard_native as native


class ConfigurationError(ValueError):
    """A configuration cannot be captured, restored, or verified reliably."""


# This allowlist is deliberately explicit: adding a processor requires reviewing
# every constructor dependency and writable property, not just its Python name.
_PROPERTIES = {
    "Bitcrush": "bit_depth",
    "BrickwallLimiter": "ceiling_db release_ms lookahead_ms true_peak",
    "Chorus": "rate_hz depth centre_delay_ms feedback mix",
    "Clipping": "threshold_db",
    "Compressor": "threshold_db ratio attack_ms release_ms",
    "Delay": "delay_seconds feedback mix",
    "Distortion": "drive_db",
    "GSMFullRateCompressor": "quality",
    "Gain": "gain_db",
    "HighShelfFilter": "cutoff_frequency_hz gain_db q",
    "HighpassFilter": "cutoff_frequency_hz",
    "Invert": "",
    "LadderFilter": "mode cutoff_hz resonance drive",
    "Limiter": "threshold_db release_ms",
    "LowShelfFilter": "cutoff_frequency_hz gain_db q",
    "LowpassFilter": "cutoff_frequency_hz",
    "MP3Compressor": "vbr_quality",
    "NoiseGate": "threshold_db ratio attack_ms release_ms",
    "PeakFilter": "cutoff_frequency_hz gain_db q",
    "Phaser": "rate_hz depth centre_frequency_hz feedback mix",
    "PitchShift": "semitones",
    "Resample": "target_sample_rate quality",
    "Reverb": "room_size damping wet_level dry_level width freeze_mode",
}
_ENUMS = {
    ("LadderFilter", "mode"): pb.LadderFilter.Mode,
    ("Resample", "quality"): pb.Resample.Quality,
    ("GSMFullRateCompressor", "quality"): pb.Resample.Quality,
}
_CONTAINERS = {"Pedalboard": pb.Pedalboard, "Chain": pb.Chain, "Mix": pb.Mix}
_MAX_BYTES = 64 * 1024 * 1024
_MAX_NODES = 1024
_MAX_DEPTH = 64


def _error(message: str) -> None:
    raise ConfigurationError(message)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _runtime() -> dict:
    return {
        "version": pb.__version__,
        "platform": platform.system(),
        "machine": platform.machine(),
        "native_sha256": _digest(Path(native.__file__).read_bytes()),
    }


def _fingerprint(path: str) -> str:
    root = Path(path)
    if not root.exists():
        _error(f"Plugin bundle is missing: {root}")
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
    if not files:
        _error(f"Plugin bundle is empty: {root}")
    for file in files:
        name = file.relative_to(root).as_posix() if root.is_dir() else file.name
        digest.update(name.encode() + b"\0")
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _format(sample_rate: float, num_channels: int) -> None:
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, float))
        or not math.isfinite(sample_rate)
        or not 1 <= sample_rate <= 768000
    ):
        _error("sample_rate must be finite and between 1 and 768000")
    if type(num_channels) is not int or not 1 <= num_channels <= 64:
        _error("num_channels must be an integer between 1 and 64")


def _settle(seconds: float, service: bool = True) -> None:
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 <= seconds <= 2:
        _error("settle_seconds must be in [0, 2]")
    if service:
        native._configuration_service_messages(seconds)


def _has_external(root: Any) -> bool:
    owners: dict[int, Any] = {}
    pending = [root]
    while pending:
        plugin = pending.pop()
        if id(plugin) in owners:
            continue
        owners[id(plugin)] = plugin
        if isinstance(plugin, pb.ExternalPlugin):
            return True
        if type(plugin) in _CONTAINERS.values():
            pending.extend(plugin)
    return False


def _encode_properties(plugin: Any, name: str) -> dict:
    result = {}
    for key in _PROPERTIES[name].split():
        value = getattr(plugin, key)
        enum = _ENUMS.get((name, key))
        result[key] = value.name if enum else value
    return result


def _capture(root: Any, *, include_state: bool) -> tuple[list[dict], int]:
    nodes: list[dict] = []
    owners: list[Any] = []
    seen: dict[int, int] = {}
    active: set[int] = set()

    def visit(plugin: Any, depth: int) -> int:
        identity = id(plugin)
        if identity in active:
            _error("Configuration graphs cannot contain cycles")
        if depth > _MAX_DEPTH:
            _error("Configuration graph is too deep")
        if identity in seen:
            return seen[identity]
        if len(nodes) >= _MAX_NODES:
            _error("Configuration has too many nodes")
        index = seen[identity] = len(nodes)
        node: dict = {}
        nodes.append(node)
        # Keep pybind wrappers alive: graph iteration may otherwise recycle IDs.
        owners.append(plugin)
        active.add(identity)
        name = type(plugin).__name__
        if name in _CONTAINERS and type(plugin) is _CONTAINERS[name]:
            node.update(
                kind="container", type=name, children=[visit(child, depth + 1) for child in plugin]
            )
        elif isinstance(plugin, pb.ExternalPlugin):
            snapshot = plugin._configuration_snapshot(include_state)
            state = snapshot.pop("state")
            snapshot["path"] = str(Path(snapshot["path"]).resolve())
            node.update(kind="external", type=name, **snapshot)
            if include_state:
                node.update(
                    bundle_sha256=_fingerprint(snapshot["path"]),
                    state=base64.b64encode(state).decode("ascii"),
                    state_sha256=_digest(state),
                )
        elif name in _PROPERTIES and type(plugin) is getattr(pb, name):
            node.update(kind="builtin", type=name, properties=_encode_properties(plugin, name))
        else:
            _error(f"Unsupported processor: {name}; required configuration cannot be captured")
        active.remove(identity)
        return index

    return nodes, visit(root, 0)


def _builtin(node: dict) -> Any:
    name = node["type"]
    try:
        plugin = getattr(pb, name)()
        for key, value in node["properties"].items():
            enum = _ENUMS.get((name, key))
            setattr(plugin, key, enum.__members__[value] if enum else value)
        if _encode_properties(plugin, name) != node["properties"]:
            _error(f"Builtin values cannot be represented: {name}")
        return plugin
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfigurationError(f"Invalid builtin configuration: {name}: {exc}") from exc


def _parameters_match(expected: list[dict], actual: list[dict]) -> None:
    if len({p["id"] for p in expected}) != len(expected):
        _error("Duplicate external parameter IDs")
    right = {p["id"]: p for p in actual}
    if set(right) != {p["id"] for p in expected}:
        _error("External parameter IDs changed")
    for left in expected:
        current = right[left["id"]]
        if (left["name"], left["label"]) != (current["name"], current["label"]):
            _error(f"Parameter metadata changed: {left['id']}")
        if abs(left["raw"] - current["raw"]) <= 8 * np.finfo(np.float32).eps:
            continue
        # A plugin may serialize native units with less precision than its raw
        # float. Accept only equal native displays with validated parser roundtrips.
        a, b = left["canonical"], current["canonical"]
        if (
            left["text"] == current["text"]
            and a is not None
            and b is not None
            and abs(a - b) <= 8 * np.finfo(np.float32).eps
        ):
            continue
        _error(f"Restored parameter differs: {left['name']} ({left['id']})")


def _compare_nodes(expected: list[dict], actual: list[dict]) -> None:
    if len(expected) != len(actual):
        _error("Restored graph topology differs")
    for left, right in zip(expected, actual):
        if (left["kind"], left["type"]) != (right["kind"], right["type"]):
            _error("Restored processor type differs")
        if left["kind"] == "external":
            for key in ("path", "identifier", "version", "name"):
                if left[key] != right[key]:
                    _error(f"Plugin identity differs: {key}")
            _parameters_match(left["parameters"], right["parameters"])
        elif left != right:
            _error(f"Restored configuration differs: {left['type']}")


def save(
    plugin: Any,
    path: str | os.PathLike,
    *,
    sample_rate: float,
    num_channels: int = 2,
    settle_seconds: float = 0.2,
) -> None:
    """Atomically save an idle graph without resetting/rendering it.

    External plugins require the main thread. Their files and private assets are
    referenced, not included. ``sample_rate`` describes the intended fresh render.
    """
    _format(sample_rate, num_channels)
    # Service host callbacks before obtaining opaque state, without processing DSP.
    _settle(settle_seconds, _has_external(plugin))
    nodes, root = _capture(plugin, include_state=True)
    document = {
        "schema": 1,
        "runtime": _runtime(),
        "sample_rate": sample_rate,
        "num_channels": num_channels,
        "root": root,
        "nodes": nodes,
    }
    try:
        data = json.dumps(document, allow_nan=False, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Configuration contains invalid values: {exc}") from exc
    if len(data) > _MAX_BYTES:
        _error("Configuration exceeds 64 MiB")
    _validate(document)
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _keys(value: Any, keys: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        _error("Invalid configuration fields")


def _validate(document: Any) -> None:
    """Preflight the entire document before constructing any external processor."""
    _keys(document, {"schema", "runtime", "sample_rate", "num_channels", "root", "nodes"})
    if type(document["schema"]) is not int or document["schema"] != 1:
        _error("Unsupported configuration schema")
    if document["runtime"] != _runtime():
        _error("Runtime differs; use the same Pedalboard native build and platform")
    _format(document["sample_rate"], document["num_channels"])
    nodes = document["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= _MAX_NODES:
        _error("Invalid configuration node count")
    for node in nodes:
        if not isinstance(node, dict):
            _error("Invalid configuration node")
        kind, name = node.get("kind"), node.get("type")
        if not isinstance(name, str):
            _error("Invalid processor type")
        if kind == "container":
            _keys(node, {"kind", "type", "children"})
            if (
                name not in _CONTAINERS
                or not isinstance(node["children"], list)
                or len(node["children"]) > _MAX_NODES
            ):
                _error("Invalid container")
        elif kind == "builtin":
            _keys(node, {"kind", "type", "properties"})
            if name not in _PROPERTIES:
                _error(f"Unsupported builtin: {name}")
            _keys(node["properties"], set(_PROPERTIES[name].split()))
            for key, value in node["properties"].items():
                enum = _ENUMS.get((name, key))
                if enum:
                    if not isinstance(value, str) or value not in enum.__members__:
                        _error(f"Invalid enum: {name}.{key}")
                elif (
                    (name, key) == ("BrickwallLimiter", "true_peak") and type(value) is not bool
                ) or (
                    (name, key) != ("BrickwallLimiter", "true_peak")
                    and (type(value) not in (int, float) or not math.isfinite(value))
                ):
                    _error(f"Invalid property: {name}.{key}")
            _builtin(node)
        elif kind == "external":
            _keys(
                node,
                {
                    "kind",
                    "type",
                    "path",
                    "identifier",
                    "version",
                    "name",
                    "parameters",
                    "bundle_sha256",
                    "state",
                    "state_sha256",
                },
            )
            if name not in ("VST3Plugin", "AudioUnitPlugin") or not hasattr(pb, name):
                _error(f"Unavailable external format: {name}")
            for key in (
                "path",
                "identifier",
                "version",
                "name",
                "bundle_sha256",
                "state",
                "state_sha256",
            ):
                if not isinstance(node[key], str):
                    _error(f"Invalid external field: {key}")
            if not Path(node["path"]).is_absolute():
                _error("External plugin path must be absolute")
            try:
                state = base64.b64decode(node["state"], validate=True)
            except ValueError as exc:
                raise ConfigurationError("Invalid state encoding") from exc
            if _digest(state) != node["state_sha256"]:
                _error("Plugin state checksum differs")
            if not isinstance(node["parameters"], list) or len(node["parameters"]) > 65536:
                _error("Invalid parameter list")
            for parameter in node["parameters"]:
                _keys(parameter, {"id", "name", "label", "text", "raw", "canonical"})
                for key in ("id", "name", "label", "text"):
                    if not isinstance(parameter[key], str):
                        _error("Invalid parameter metadata")
                for key in ("raw", "canonical"):
                    value = parameter[key]
                    if key == "canonical" and value is None:
                        continue
                    if (
                        type(value) not in (int, float)
                        or not math.isfinite(value)
                        or not 0 <= value <= 1
                    ):
                        _error("Invalid normalized parameter value")
            _parameters_match(node["parameters"], node["parameters"])
        else:
            _error("Invalid node kind")
    visited: set[int] = set()
    active: set[int] = set()

    def visit(index: int, depth: int) -> None:
        if type(index) is not int or not 0 <= index < len(nodes):
            _error("Invalid node reference")
        if index in active or depth > _MAX_DEPTH:
            _error("Cyclic or excessively deep graph")
        if index in visited:
            return
        active.add(index)
        for child in nodes[index].get("children", []):
            visit(child, depth + 1)
        active.remove(index)
        visited.add(index)

    visit(document["root"], 0)
    if len(visited) != len(nodes):
        _error("Unreachable configuration nodes")
    # Dependency checks follow structural preflight, still before any native load.
    for node in nodes:
        if node["kind"] == "external" and _fingerprint(node["path"]) != node["bundle_sha256"]:
            _error(f"Plugin bundle changed: {node['path']}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _error(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def load(path: str | os.PathLike, *, settle_seconds: float = 0.2) -> Any:
    """Create a fresh graph and validate restored settings; does not verify sound."""
    with Path(path).open("rb") as stream:
        data = stream.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        _error("Configuration exceeds 64 MiB")
    try:
        document = json.loads(data, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise ConfigurationError("Invalid configuration JSON") from exc
    _validate(document)
    _settle(settle_seconds, any(n["kind"] == "external" for n in document["nodes"]))
    restored: dict[int, Any] = {}

    def create(index: int) -> Any:
        if index in restored:
            return restored[index]
        node = document["nodes"][index]
        name = node["type"]
        if node["kind"] == "container":
            plugin = _CONTAINERS[name]([create(child) for child in node["children"]])
        elif node["kind"] == "builtin":
            plugin = _builtin(node)
        else:
            plugin = getattr(pb, name)(node["path"], plugin_name=node["name"])
            identity = plugin._configuration_snapshot(False)
            for key in ("identifier", "version", "name"):
                if identity[key] != node[key]:
                    _error(f"Loaded plugin identity differs: {key}")
            plugin._configuration_restore(
                base64.b64decode(node["state"]), document["sample_rate"], document["num_channels"]
            )
        restored[index] = plugin
        return plugin

    try:
        result = create(document["root"])
        _settle(settle_seconds, _has_external(result))
        actual, _ = _capture(result, include_state=False)
        _compare_nodes(document["nodes"], actual)
        return result
    except ConfigurationError:
        raise
    except (ValueError, TypeError, RuntimeError, ImportError) as exc:
        raise ConfigurationError(f"Restoring configuration failed: {exc}") from exc


@dataclass(frozen=True)
class VerificationResult:
    """Measured output errors; parameter readback has also passed."""

    exact: bool
    max_abs_error: float
    rms_error: float
    samples: int


def verify(
    original: Any,
    restored: Any,
    input_array: np.ndarray | None = None,
    *,
    sample_rate: float,
    midi: Any = None,
    duration: float | None = None,
    num_channels: int = 2,
    warmup_seconds: float = 1.0,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    exact: bool = False,
    settle_seconds: float = 0.2,
) -> VerificationResult:
    """Reset and render both graphs, checking settings and sound independently.

    Audio must be a finite channel-first array (or mono vector). For a standalone
    instrument supply ``midi``, ``duration`` and ``num_channels`` instead. This
    consumes both graphs' DSP history; use disposable instances if it matters.
    """
    _format(sample_rate, num_channels)
    for name, value in (("warmup_seconds", warmup_seconds), ("atol", atol), ("rtol", rtol)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            _error(f"{name} must be finite and nonnegative")
    if warmup_seconds > 60:
        _error("warmup_seconds must not exceed 60")
    if input_array is None:
        if midi is None or duration is None or not math.isfinite(duration) or duration <= 0:
            _error("Instrument verification requires MIDI and a positive duration")
        notes = list(midi)
        audio = None
    else:
        if midi is not None or duration is not None:
            _error("Choose audio or standalone MIDI verification")
        audio = np.asarray(input_array, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        if audio.ndim != 2 or not audio.shape[1] or not np.isfinite(audio).all():
            _error("Expected nonempty finite channel-first audio")
        num_channels = audio.shape[0]
        _format(sample_rate, num_channels)
    _settle(settle_seconds, _has_external(original) or _has_external(restored))
    left, _ = _capture(original, include_state=False)
    right, _ = _capture(restored, include_state=False)
    _compare_nodes(left, right)

    def render(plugin: Any) -> np.ndarray:
        plugin.reset()
        warmup = max(8192, int(sample_rate * warmup_seconds)) if warmup_seconds else 0
        if warmup:
            if audio is None:
                plugin(
                    [],
                    duration=warmup / sample_rate,
                    sample_rate=sample_rate,
                    num_channels=num_channels,
                    buffer_size=8192,
                    reset=False,
                )
            else:
                plugin(
                    np.zeros((num_channels, warmup), dtype=np.float32),
                    sample_rate,
                    buffer_size=8192,
                    reset=False,
                )
        if audio is None:
            return plugin(
                notes,
                duration=duration,
                sample_rate=sample_rate,
                num_channels=num_channels,
                buffer_size=8192,
                reset=False,
            )
        return plugin(audio.copy(), sample_rate, buffer_size=8192, reset=False)

    try:
        a, b = render(original), render(restored)
    except (ValueError, TypeError, RuntimeError, ImportError) as exc:
        raise ConfigurationError(f"Could not render verification probe: {exc}") from exc
    _settle(settle_seconds, _has_external(original) or _has_external(restored))
    # A reset/reinstantiation must not make both operands silently fall back to
    # the same defaults and thereby produce a misleading audio pass.
    final_left, _ = _capture(original, include_state=False)
    final_right, _ = _capture(restored, include_state=False)
    _compare_nodes(left, final_left)
    _compare_nodes(right, final_right)
    if not a.size or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        _error("Rendered output shapes differ or contain nonfinite samples")
    difference = a.astype(np.float64) - b.astype(np.float64)
    maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
    rms = float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0
    equal = np.array_equal(a, b)
    if not (equal if exact else np.allclose(a, b, atol=atol, rtol=rtol)):
        _error(f"Sound differs: max_abs_error={maximum:.9g}, rms_error={rms:.9g}")
    return VerificationResult(equal, maximum, rms, a.size)
