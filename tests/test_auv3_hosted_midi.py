"""Real AUv3 combined input with bounded subprocesses and owned registration."""

import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from .test_auv3_automation import auv3_extension  # noqa: F401

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="AUv3 requires macOS")
CASES = [
    f"{rate}/{buffer}/{delay}/{pattern}"
    for rate in [44100, 48000]
    for buffer in [1, 31, 8192]
    for delay in [0, 20]
    for pattern in ["constant", "dense", "controller"]
]


@pytest.mark.parametrize("case", CASES)
def test_actual_auv3_combined_input(request, case, record_property):
    extension = request.getfixturevalue("auv3_extension")
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "tests.auv3_automation_plugin.midi_bus_checks",
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
