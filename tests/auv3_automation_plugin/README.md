# Actual out-of-process AUv3 fixture

This original no-UI AUAudioUnit/extension fixture is GPL-3.0-only. Generated binaries,
containing apps and signatures stay in ignored build output or pytest temporary
storage. No system plugin directories or developer certificates are used.

Run `uv run pytest tests/test_auv3_automation.py -q` with the native extension installed
on macOS. A logged-in macOS user session, Xcode toolchain, CMake, codesign, PlugInKit
and LaunchServices are required. Build/registration errors fail on macOS; the suite
is skipped on other operating systems. Only macOS arm64 has been verified locally.

`runtime.py` builds an actual `.app/Contents/PlugIns/*.appex`, signs the extension
and app ad-hoc, and registers them with LaunchServices/PlugInKit. Bundle identifiers
and component subtypes depend on the build directory. The helper refuses to replace
preexisting entries and unregisters its own entries at teardown. It uses no GUI,
network, App Store account or private signing key.

The extension principal class implements AUAudioUnitFactory. Three components share
known DSP: a direct effect, a20-sample delayed effect and a note-gated instrument.
Each test uses the public AudioUnitPlugin API in a subprocess with a30-second bound;
this captures hangs/crashes during loading, processing or destruction. It is not a
new timeout guarantee for all third-party plugins.

The identity probe emits `getpid()` from the actual render block into output audio,
which must differ from Python's PID and match the extension's reported PID. Native
component flags must include IsV3AudioUnit. This distinguishes remote rendering
from an extension process used only for auxiliary services or process-local subclass
registration. Frequency uses address0x100000002; a separate boolean uses address0x2,
checking that the legacy AU bridge does not truncate controls into the wrong node.

Analytical tests use the same physical float32 oracle as the AUv2 suite, with3e-7
absolute/relative error bounds. Partition/reset/hold comparisons require exact
output equality. Tests cover three component descriptions in one extension, relative/
absolute/trailing-slash/symlink/Unicode paths, dB/Hz/boolean curves, reported latency,
MIDI offsets/holds, graph queues, owned/omitted lanes, state and reinstantiation.
Native scheduled events, third-party AUv3 vendor compatibility and GUI behavior are
outside this fixture's evidence.
