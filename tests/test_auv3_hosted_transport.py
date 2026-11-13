"""Bounded actual remote AUv3 transport acceptance and registration cleanup."""

import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from .test_auv3_automation import auv3_extension  # noqa: F401

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="AUv3 requires macOS")
CASES = [
    *[
        f"tempo/{rate}/{buffer}/{playing}/{ppq}"
        for rate in [44100, 48000]
        for buffer in [1, 31, 8192]
        for playing in [0, 1]
        for ppq in [-2, 2]
    ],
    *[
        f"position/48000/{buffer}/{sample}"
        for buffer in [1, 31, 8192]
        for sample in [-(2**40), 0, 2**40]
    ],
    *[
        f"{kind}/{rate}/{buffer}"
        for kind in ["loop", "fractional", "meter"]
        for rate in [44100, 48000]
        for buffer in [1, 31, 8192]
    ],
    "bounds/48000/31",
    "changed/48000/31",
    "joint/48000/31",
    *[f"origin/48000/{b}/{p}" for b in [1, 31, 8192] for p in [-(2**40), 0, 2**40]],
    *[f"many/44100/{buffer}" for buffer in [1, 31, 8192]],
]


@pytest.mark.parametrize("case", CASES)
def test_actual_auv3_transport(request, case, record_property):
    extension = request.getfixturevalue("auv3_extension")
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "tests.auv3_automation_plugin.transport_checks",
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
