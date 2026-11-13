"""Actual signed AUv3 extension discovery and out-of-process DSP, bounded per case."""

import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from .auv3_automation_plugin.runtime import registered_extension

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="AUv3 extensions require macOS"
)


@pytest.fixture(scope="session")
def auv3_extension(tmp_path_factory):
    with registered_extension(tmp_path_factory.mktemp("auv3-automation-build")) as path:
        yield path


CASES = [
    "discovery/absolute",
    "discovery/relative",
    "discovery/trailing",
    "discovery/symlink",
    "identity",
    "unregistered",
    *[
        f"audio/{kind}/{rate}/{buffer}/{layout}"
        for kind in ["direct", "delayed"]
        for rate in [44100, 96000]
        for buffer in [1, 31, 8192]
        for layout in ["mono", "planar", "interleaved"]
    ],
    *[f"midi/{rate}/{buffer}" for rate in [44100, 48000, 96000] for buffer in [1, 31, 8192]],
    "midi-hold",
    "graph/serial",
    "graph/parallel",
    "owned-pending",
    "reset/clear",
    "reset/reload",
    "validation",
    "boolean",
    "state",
]


@pytest.mark.parametrize("case", CASES)
def test_actual_auv3_extension(auv3_extension, case):
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "tests.auv3_automation_plugin.checks",
            str(auv3_extension),
            case,
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output["case"] == case
