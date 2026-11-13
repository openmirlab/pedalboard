"""Real signed AUv3 compensation, with an independent process bound per case."""

import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from .test_auv3_automation import auv3_extension  # noqa: F401

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="AUv3 requires macOS")

CASES = [
    "state/20",
    *[f"identity/{latency}" for latency in [20, 4097, 16385]],
    *[
        f"onset/{latency}/{rate}/{buffer}/{channels}"
        for latency in [0, 20, 4097, 16385]
        for rate in [44100, 48000]
        for buffer in [1, 31, 8192]
        for channels in [1, 2]
    ],
    *[f"curves/{latency}/{buffer}" for latency in [20, 4097, 16385] for buffer in [1, 31, 65536]],
    *[
        f"{kind}/{latency}"
        for kind in ["late", "reset", "reject", "stream"]
        for latency in [20, 16385]
    ],
]


@pytest.mark.parametrize("case", CASES)
def test_actual_auv3_instrument_latency(request, case, record_property):
    extension = request.getfixturevalue("auv3_extension")
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "tests.auv3_automation_plugin.latency_checks",
            str(extension),
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
    record_property("measurement", json.dumps(output))
