Explicit hosted audio buses
===========================

``VST3Plugin`` and ``AudioUnitPlugin`` expose ``audio_buses`` and
``process_buses``. The original array/MIDI convenience APIs retain their existing
contracts. This explicit audio path supports independent main, sidechain and
auxiliary ports without assuming equal total input/output channel counts.

Inspect the plugin's ports before connecting them::

    from pedalboard import VST3Plugin

    plugin = VST3Plugin("/path/to/compressor.vst3")
    print(plugin.audio_buses)
    result = plugin.process_buses(
        {0: main_audio, 1: detector_audio}, 48000,
        output_buses={0: 2},
    )
    processed_main = result[0]

Bus indices are zero-based in each direction, stable only for the currently
loaded plugin topology. Names are labels, not routing identifiers. Metadata is a
fresh dictionary with ``inputs`` and ``outputs`` lists; each entry contains
``index``, ``name``, ``enabled``, ``channels``, ``default_channels`` and a human
readable ``layout``. This reports current/default arrangements, not an exhaustive
list of every layout a plugin might accept. A plugin can reject a joint layout
even if its individual channel counts appear reasonable.

Array and routing contract
--------------------------

``input_buses`` maps input bus indices to two-dimensional NumPy float32/float64
arrays of shape ``(channels, frames)``. No channel-axis inference is performed.
Noncontiguous arrays are accepted; conversion/copying occurs before rendering.
All inputs must have the same number of frames. Source arrays are never written.
Each bus supports 1–64 channels, subject to the plugin's actual native layout
support. Existing speaker layouts are preferred when channel counts match;
otherwise JUCE's canonical arrangement for that count is requested. Arbitrary
speaker-label remapping is not provided.

``output_buses`` maps output indices to channel counts. At least one output must
be selected. The result is a dictionary with exactly those indices, each owning
a separate planar float32 array. For example, ``output_buses={0: 2, 1: 1}``
requests stereo main plus mono auxiliary output. Inputs and outputs may have
different total channel counts. ``num_frames`` is inferred from audio, or must be
provided explicitly if no input is connected. When supplied alongside audio it
must match the input length. An empty render returns empty arrays without
resetting or processing.

For VST3, omitted buses are requested disabled; plugins that require an omitted
bus to remain enabled reject that layout. Connect a silent array if that bus is
required. The inherited Audio Unit host cannot disable individual buses: omitted
AU input buses remain at their existing channel layout and receive silence;
omitted AU output buses are processed and discarded. Metadata shows these buses
as enabled. No unconnected signal leaks into the selected outputs through the
host's scratch buffer.

Timing and state
----------------

The explicit bus path returns **raw plugin output**, including reported latency.
It does not trim, drain latency or add a tail. All inputs are synchronized on one
source timeline. The caller owns any path compensation in its surrounding mixer.
Use the existing MIDI convenience API for
:doc:`compensated offline instrument rendering <instrument_latency>`.

``reset=True`` starts a new render, clearing processing history and preparing a
joint bus layout. ``reset=False`` continues the same stream. Changing sample rate
or routing, or switching between this explicit path and the convenience audio/MIDI
APIs, requires a reset. Caller ``buffer_size`` may vary without re-preparing a
continuous bus stream; native spans are bounded by 8192 frames and the caller's
requested size. Internal scratch is bounded by the enabled channel count and this
capacity, rather than allocating one combined full-duration bus buffer.

Array types/shapes/lengths, bus indices and numeric arguments are checked before
reset or DSP processing. Actual layout negotiation requires a stopped plugin:
with ``reset=True``, a native layout/preparation failure can already have cleared
history. Do not assume rollback of plugin memory after such a failure. Existing
GUI/main-thread/plugin-state restrictions still apply. Configuration persistence
saves settings, not an in-flight routed stream or DSP history.

Verification scope
------------------

The source-built VST3 diagnostic includes separately observable main, external
key and auxiliary output ports, with an independent SDK host loading the same
plugin binary. Tests compare actual audio, input ownership, reset boundaries,
stream partitioning and untrimmed reported latency. Actual AUv2 and signed out-of-process AUv3 diagnostics cover sidechain, multiple
outputs and asymmetric mono/stereo layouts. An always-on AUv3 identity component
emits its render-process PID into both output buses, independently of parameter
setters. These multi-bus diagnostics cannot process in-place; the extension
explicitly declares that requirement.

CHOWTapeModel VST3/AUv2 output matches the legacy convenience path exactly when
using the same native partitions and the latency reported after rate preparation.
At 44.1/48kHz this fixture reports 37/39 samples. CHOW itself is block-dependent;
changing the native partition can change its audio, including on existing APIs.
Precise delivery does not guarantee block-invariant DSP from arbitrary plugins.

An optional unmodified JUCE NoiseGate example verifies an existing musical
sidechain algorithm against a native host loading the same VST3 binary. No
commercial third-party AU sidechain or vendor-distributed AUv3 certification is
claimed. Native execution has been verified on macOS arm64; other OS/architectures
remain unverified. Diagnostic evidence does not establish compatibility with
every plugin, platform or speaker layout.
