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

The extension principal class implements AUAudioUnitFactory. Fourteen components share
known DSP: a direct effect, a20-sample delayed effect, a note-gated instrument and
three instruments delayed by20/4097/16385 reported samples.
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
output equality. Tests cover fourteen component descriptions in one extension, relative/
absolute/trailing-slash/symlink/Unicode paths, dB/Hz/boolean curves, reported latency,
MIDI offsets/holds, graph queues, owned/omitted lanes, state and reinstantiation.
Native scheduled events, third-party AUv3 vendor compatibility and GUI behavior are
outside this fixture's evidence.


`uv run pytest tests/test_auv3_instrument_latency.py -q` exercises69 fresh-process
compensation cases: both44.1/48kHz,mono/stereo,blocks1/31/8192, near-end notes,
longer-than-clip latency, source-clock curves and a65536-frame caller block.
It also checks repeated reset, empty duration, post-duration MIDI exclusion, raw
streams, early rejection without consuming MIDI, and writable-only configuration
roundtrips. Each delayed instrument must emit a render PID distinct from Python.
Read-only Render Frames/Render Capacity parameters expose actual DSP calls and
prepared capacity, so silent bypass cannot count as successful processing.

AUv3 MIDI prepares at least8192 frames while processing the requested small spans.
The regression covers the observed scalar tiny-capacity failure: without that
floor, no DSP render occurs for tested1–16-frame capacities, although curves on
an8192-frame prepared engine work. Native AU flags exclude read-only counters from
configuration comparisons even when JUCE's generic automatable flag is permissive.
This remains source-built host acceptance, not third-party vendor or GUI acceptance.

`tests/test_auv3_hosted_audio.py` exercises two additional components: **Buses**
has independent main/key inputs and main/aux outputs; **Bus Identity** always
emits the render-process PID on both output buses without relying on setters.
Their multi-output DSP requires distinct input/output memory and declares
`canProcessInPlace=NO`. Tests verify actual remote identity, asymmetric mono/stereo
layouts,44.1/48kHz,1/31/8192-frame spans, split streams and silent unconnected
inputs. The Render Bus Mask readonly meter records which native output callbacks
ran. This is a diagnostic extension, not commercial-vendor certification.

`tests/test_auv3_hosted_midi.py` adds Audio MIDI and Delayed Audio MIDI effects.
They combine input audio, independent external key/auxiliary output, note timing,
physical dB/Hz controls, CC1 discrete bypass and raw20-sample main delay. Whole/split
streams match exactly. These variants declare MIDI input only; their empty output
lists do not verify AU MIDI output. The original eight variants retain their
behavior, including in-place capability and event processing contracts.

`tests/test_auv3_hosted_transport.py` adds Clock/Meter/Loop/Position components.
The actual remote render loop calls musicalContextBlock and transportStateBlock,
then encodes received timing into four output channels. Tests assert signed 64-bit
project frames, negative PPQ/bars, recording/play/loop flags, next-beat offsets,
seek versus continuation, repeated fractional wraps and explicit split anchors.
The clock branch also consumes its scheduled parameter/MIDI events, so joint
clock/curve acceptance observes both native facilities. No vendor AUv3 clock
interpretation or GUI compatibility is implied.
