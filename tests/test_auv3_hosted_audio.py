"""Signed actual AUv3 multi-bus acceptance with bounded child processes."""

import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from .test_auv3_automation import auv3_extension  # noqa: F401

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="AUv3 requires macOS")

CASES = [
    "identity",
    *[
        f"{rate}/{buffer}/{channels}/{key}/{aux}"
        for rate in [44100, 48000]
        for buffer in [1, 31, 8192]
        for channels, key, aux in [(1, 2, 1), (2, 1, 2), (2, 2, 1)]
    ],
]


@pytest.mark.parametrize("case", CASES)
def test_actual_auv3_bus_routing(request, case, record_property):
    extension = request.getfixturevalue("auv3_extension")
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "tests.auv3_automation_plugin.bus_checks",
            str(extension),
            case,
        ],
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output["case"] == case
    record_property("measurement", json.dumps(output))
