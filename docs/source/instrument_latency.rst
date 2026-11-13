Offline instrument latency compensation
=======================================

MIDI instrument processing accepts keyword-only ``compensate_latency=False`` on
``process`` and ``__call__``. The default preserves raw plugin output timing and
existing streaming behavior. Set it to ``True`` for one standalone offline clip:

.. code-block:: python

    import pedalboard

    instrument = pedalboard.load_plugin("instrument.vst3")
    notes = [(bytes([0x90, 60, 100]), 1000 / 48000),
             (bytes([0x80, 60, 0]), 0.5)]
    audio = instrument(notes, duration=1.0, sample_rate=48000,
                       reset=True, compensate_latency=True)

If the instrument introduces and reports 20 samples of latency, compensation
removes those 20 leading frames and renders 20 additional frames to recover the
end. A note scheduled at frame 1000 then begins at frame 1000, provided the plugin
otherwise follows MIDI event timing. The result has the same requested length as
the raw render: the existing native conversion is ``int(float32(duration) *
float32(sample_rate))`` frames per channel. Buffering and alignment run in C++.
Scratch storage is bounded by the processing block, rather than latency length.

Compensation requires ``reset=True``. Using it with ``reset=False`` raises before
consuming MIDI or changing plugin state. Do not compensate each chunk of a stream:
that would drain and advance its future state repeatedly. For streaming use raw
output, retain one instance, and align the complete stream once at the caller's
chosen boundary. A compensated clip advances the instance through its drain;
reset before starting another independent clip.

MIDI timestamps and automation curves use the instrument input/source timeline.
Curves still need one value per requested source frame, not extra latency frames.
Drain processing holds the last curve values and does not deliver MIDI at or after
the requested end. Consequently a late note is not lost just because its raw output
would have arrived after the requested duration. Returned MIDI is not re-fed.

``duration`` must include any musical release or tail the caller wants. The extra
latency drain only aligns that requested interval; it does not append an automatic
note-off, release or reverb tail. A zero-frame request returns an empty array.

The latency is read after reset/format preparation and must be fixed for the clip.
If the host observes the reported latency changing during processing, compensation
raises instead of returning a guessed alignment. Plugins can misreport latency or
update it asynchronously; this check is not a guarantee that every internal delay
is visible. It cannot compensate a deliberately slow attack envelope or an unknown
internal delay. Other hosting work (buses, sidechain and transport) is separate.

AUv3 MIDI prepares a capacity of at least 8192 frames, including scalar renders,
while still processing the requested small spans. This avoids the observed silent
failure with tiny prepared capacities; it does not quantize MIDI event positions
to that capacity. VST3 and AUv2 retain their existing preparation behavior.

The implementation is shared by VST3 and Audio Unit MIDI processing. Validation
uses deterministic source-built delayed VST3/AUv2 instruments and an actual signed,
out-of-process AUv3 extension, plus the original TactusBass VST3 integration.
AUv3 coverage includes 44.1/48 kHz, mono/stereo, small blocks, longer-than-clip
latency, source-clock curves, raw streams and configuration roundtrips. Other
plugins, architectures and GUI workflows require separate acceptance.
