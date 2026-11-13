"""Build/register an isolated actual AUv3 extension and remove its own entries."""

import contextlib
import plistlib
import shutil
import subprocess
from pathlib import Path

LSREGISTER = "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"


def command(args, timeout=60):
    result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args!r}: {result.stdout}\n{result.stderr}")
    return result.stdout


@contextlib.contextmanager
def registered_extension(build):
    source = Path(__file__).parent.resolve()
    cmake = shutil.which("cmake")
    if not cmake:
        raise RuntimeError("AUv3 tests require CMake and Xcode's macOS toolchain")
    command([cmake, "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"], 120)
    command([cmake, "--build", str(build), "--parallel", "2"], 120)
    app = Path(build).resolve() / "AUv3 Automation Probe.app"
    extension = app / "Contents/PlugIns/AutomationAUv3.appex"
    info = plistlib.loads((extension / "Contents/Info.plist").read_bytes())
    identifier = info["CFBundleIdentifier"]
    previous = command(["pluginkit", "-m", "-A", "-D", "-i", identifier])
    if previous.strip():
        raise RuntimeError(f"Refusing to replace an existing AUv3 registration: {previous}")
    for bundle in [extension, app]:
        command(
            [
                "codesign",
                "--force",
                "--sign",
                "-",
                "--entitlements",
                str(source / "Entitlements.plist"),
                str(bundle),
            ]
        )
    try:
        command([LSREGISTER, "-f", str(app)])
        command(["pluginkit", "-a", str(extension)])
        registered = command(["pluginkit", "-m", "-A", "-D", "-i", identifier])
        if identifier not in registered:
            raise RuntimeError(f"PlugInKit did not discover {extension}")
        yield extension
    finally:
        # Both entries belong only to this generated build. Never remove or
        # override user-installed extensions, identities or signing settings.
        try:
            command(["pluginkit", "-r", str(extension)])
        finally:
            command([LSREGISTER, "-u", str(app)])
        remaining = command(["pluginkit", "-m", "-A", "-D", "-i", identifier])
        if remaining.strip():
            raise RuntimeError(f"AUv3 fixture registration was not removed: {remaining}")
