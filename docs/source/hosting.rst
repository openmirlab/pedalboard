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

MIDI and parameter curves on the same clock
------------------------------------------

``process_buses`` accepts ``midi_messages`` and ``automation`` together with audio::

    import numpy as np

    outputs, midi_out = plugin.process_buses(
        {0: main_audio, 1: detector_audio}, 48000,
        output_buses={0: 2},
        midi_messages=[(bytes([0x90, 60, 100]), 31 / 48000)],
        automation={"gain_db": np.linspace(-12, 0, main_audio.shape[1])},
        return_midi=True,
    )

The default ``return_midi=False`` preserves the original output-bus dictionary.
With ``True``, the result is ``(output_bus_dict, midi_messages)``. MIDI output
messages are ``(bytes, seconds)`` pairs relative to the beginning of this call.
No hidden global cursor is added to returned timestamps. Output messages are
captured only when requested and are never fed into later input spans.

Input MIDI uses the existing byte/timestamp or mido.Message representation;
timestamps are absolute seconds within the call, not delta times. Events are
placed relative to each actual native processing span, including subdivisions
created by curves. Negative timestamps and events at/after ``num_frames`` are
excluded. No note-off, tail or latency drain is synthesized. To hold a note across
calls, send its note-on once, continue with ``reset=False``, then send its note-off
at the appropriate offset in a later call. Input-free instruments use
``input_buses={}`` and an explicit ``num_frames``. These are audio-producing
instruments; the existing loader still rejects MIDI-only processors with no audio
output ports.

``midi_capabilities`` reports the native ``accepts_input`` and ``produces_output``
declarations. A nonempty MIDI buffer on a plugin declaring no input is rejected
before reset. A plugin declaring no output returns an empty output list. Declarations
alone do not prove that a plugin responds to every message type or controller.
VST3 controller messages follow the plugin's native MIDI-to-parameter mapping;
unmapped controls need not affect its sound. MIDI port routing currently uses the
host's existing main event port; this does not expose separate MIDI bus selection.

Automation retains display units (dB, Hz or the parameter's native numeric units),
with one NumPy value per source frame. ``sample_steps`` is the precise default;
constant runs coalesce in C++ and differing targets split native spans. VST3 also
accepts the existing opt-in ``vst3_events`` mode; AU supports sample steps only.
Physical conversion is shared with the existing convenience APIs and is checked
again after format preparation. If load-time ranges exclude desired targets,
first prepare the desired format (for a compatible main layout, use
``prepare_automation``), then submit bus curves with the prepared model. Preparation
and native failures can already have changed history. Plugin smoothers and
block-dependent DSP remain part of its behavior. MIDI controller mappings and
curves addressing the same knob follow native plugin/queue ordering, without a
new host-wide priority rule.

Each raw bus call consumes all its submitted targets. Omitted lanes retain the
plugin's current setting. Curves may be added to an existing bus stream; switching
native automation delivery modes after activation requires a reset. The graph
convenience API's pending-control queues and automatic PDC are separate contracts.
Pure array/index/curve failures precede MIDI iteration and reset; MIDI timestamps
are validated before resetting. Python generators run without the plugin mutex,
and routing/schema are revalidated afterward. Empty renders do not consume MIDI.

Timing and state
----------------

The explicit bus path returns **raw plugin output**, including reported latency.
It does not trim, drain latency or add a tail. All inputs are synchronized on one
source timeline. The caller owns any path compensation in its surrounding mixer.
Use the existing MIDI convenience API for
:doc:`compensated offline instrument rendering <instrument_latency>`.

``reset=True`` starts a new render using the existing native reset/reload policy
and prepares a joint bus layout. After this explicit path has delivered MIDI,
reset conservatively reinstantiates the plugin: audio-only reset detection does
not establish that an effect clears its held voices. This costs more than a
simple native reset and requires the main thread, like the existing instrument
reload path. It leaves continuous ``reset=False`` processing unchanged. ``reset=False`` continues the same stream. Changing sample rate
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

Combined-input verification includes source-built VST3/AUv2 and actual remote
AUv3 note-gated effects with separate sidechain/auxiliary ports, gain/Hz curves,
CC1 mapped to a discrete bypass, and raw20-sample delay. VST3 returned note events
are compared to an independent SDK host loading the same binary, with output/input
separation checks. The AU fixtures declare no MIDI output, so their empty results
do not certify AU output delivery. Magical8bitPlug2 VST3/AUv2 audio matches the
existing MIDI convenience renderer exactly through the new input-free entry point.
An optional unchanged JUCE AudioPluginDemo mixes external audio and a musical
synthesizer; its VST3 audio matches an independent native host exactly, with/without
notes and at several partitions. These source examples and selected binaries do
not certify arbitrary commercial MIDI effects, AU MIDI output or other platforms.
