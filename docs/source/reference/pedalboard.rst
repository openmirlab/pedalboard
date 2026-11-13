The ``pedalboard`` API
======================

This module provides classes and functions for adding effects to audio.
Most classes in this module are subclasses of :class:`Plugin`, each of
which allows applying effects to an audio buffer or stream.

.. note::
   For audio I/O functionality (i.e.: reading and writing audio files), see
   the :py:mod:`pedalboard.io` module.

The :py:mod:`pedalboard` module is named after
`the concept of a guitar pedalboard <https://en.wikipedia.org/wiki/Guitar_pedalboard>`_,
in which musicians will chain various effects pedals together to give them
complete control over their sound. The :py:mod:`pedalboard` module implements
this concept with its main :class:`Pedalboard` class::

   from pedalboard import Pedalboard, Chorus, Distortion, Reverb

   # Create an empty Pedalboard object:
   my_pedalboard = Pedalboard()

   # Treat this object like a Python list:
   my_pedalboard.append(Chorus())
   my_pedalboard.append(Distortion())
   my_pedalboard.append(Reverb())

   # Pass audio through this pedalboard:
   output_audio = my_pedalboard(input_audio, input_audio_samplerate)


:class:`Pedalboard` objects are lists of zero or more :class:`Plugin` objects,
and :class:`Pedalboard` objects themselves are subclasses of :class:`Plugin` -
which allows for nesting and composition.

The following documentation lists all of the available types of :class:`Plugin`.


.. automodule:: pedalboard
   :members:
   :imported-members:
   :inherited-members:
   :special-members: __call__

Native parameter curves
-----------------------

This fork adds an optional keyword-only ``automation`` argument to ``process``
and ``__call__`` on these four effects:

================== ==================
Effect             Parameter
================== ==================
``Gain``           ``gain_db``
``Clipping``       ``threshold_db``
``Distortion``     ``drive_db``
``Bitcrush``       ``bit_depth``
================== ==================

For mono or planar audio (channels, frames), an example gain ramp is::

    import numpy as np
    from pedalboard import Gain

    curve = np.linspace(-18, 0, audio.shape[-1], dtype=np.float32)
    output = Gain().process(
        audio, sample_rate, automation={"gain_db": curve}
    )

Each curve must be a one-dimensional real NumPy array containing one target
value per input audio frame, shared by all channels. Float64 and integer curves
are converted to the effects' float32 parameter precision. Noncontiguous and
read-only arrays are accepted. Curves are validated and copied into native
storage before processing; input audio and control arrays are not modified.

Targets use the existing parameter units and are applied before their input
frame is processed. No additional interpolation or smoothing is introduced.
The final consumed value remains on the effect for subsequent scalar calls.
Empty audio with empty curves leaves the parameter unchanged. ``bit_depth``
accepts fractional values in [0, 32]. The dB controls accept negative infinity
as silence; NaN, positive infinity and values that overflow native gains are
rejected before DSP parameters change.

For successive audio chunks, pass matching curve slices and ``reset=False``.
The usual audio-layout detection rules still apply: establish the channel layout
with an unambiguous block before tiny or square multichannel fragments.
Omit ``automation`` (or pass ``None`` or an empty dict) to use the existing
scalar processing path. ``Pedalboard``/``Chain``/``Mix`` also accept effect-instance
mappings, as described below. VST3 physical-unit automation is described in the
VST3 section below; Audio Unit automation is not included. Additional dynamics controls are listed below.

Dynamics controls
~~~~~~~~~~~~~~~~~

The same ``automation`` dict is also available on ``Compressor``, ``NoiseGate``
and ``Limiter``. Multiple curves may be supplied together::

    output = compressor.process(
        audio,
        sample_rate,
        reset=False,
        automation={
            "threshold_db": threshold_curve,
            "ratio": ratio_curve,
            "attack_ms": attack_curve,
            "release_ms": release_curve,
        },
    )

``Compressor`` and ``NoiseGate`` support all four controls above. ``Limiter``
supports ``threshold_db`` and ``release_ms``. Omitted controls keep their scalar
values. All curves validate before DSP mutation; all changes at a frame are
applied together before its audio is processed, independent of dict order.
The first curve values are applied before reset/preparation so the existing
Limiter makeup ramp starts from the requested threshold. ``reset=False`` retains
its ramp progress and the dynamics detectors' envelopes. Increasing only the
processing block size preserves these three processors' state; a sample-rate or
channel-count change still prepares the DSP as before.

Dynamics thresholds must be finite float32 values for which the JUCE threshold
and reciprocal remain finite and positive. In particular, the -200 dB floor and
lower values are rejected. Limiter also requires a finite makeup gain. Ratios
must be finite and at least 1; attack/release times must be finite and nonnegative.
Zero milliseconds selects JUCE's immediate detector response. These rules apply
to the new curve path; existing scalar setter acceptance is unchanged.

No new parameter smoothing is added. Compressor/NoiseGate attack and release
control envelope response, not smoothing of parameter changes. Limiter keeps its
existing JUCE makeup-gain smoothing and loudness-maximizing behavior; it does not
become a configurable-ceiling or true-peak limiter.


First-order filter controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``LowpassFilter`` and ``HighpassFilter`` accept
``automation={"cutoff_frequency_hz": curve}`` using the same dense curve contract.
These remain the original first-order, 6 dB/octave JUCE filters. Cutoff changes
refresh coefficients without resetting the filter memory; block-size growth also
preserves memory. Sample-rate or channel-count changes still prepare the DSP.
The scalar setter still takes effect at the next processing call.

Curve cutoffs must remain finite and positive after float32 conversion and strictly
below half the current sample rate (Nyquist). Values whose float32 coefficient
calculation puts the filter pole on or outside the unit circle are rejected before
processing; this also excludes extreme near-zero and near-Nyquist rounding cases.
These validation rules apply to curves only; scalar setter acceptance is unchanged.
No clamping, smoothing, interpolation or reduction of control resolution is added.
Rapid sweeps or abrupt cutoff changes can still produce audible transients inherent
to the original time-varying filter. Fixed settings should use the scalar path to
avoid curve-validation and scheduling overhead.


Peak and shelf EQ controls
~~~~~~~~~~~~~~~~~~~~~~~~~~

``PeakFilter``, ``LowShelfFilter`` and ``HighShelfFilter`` support synchronized
``cutoff_frequency_hz``, ``gain_db`` and ``q`` curves. All supplied targets at a
frame are stored before one coefficient refresh. JUCE's existing formulas and
scalar setter behavior are preserved; no smoothing is added. Filter memory is
retained on curve updates and block-size growth. Rate/channel changes still
prepare the DSP.

These EQs retain their original frequency handling: positive requested cutoff
is clamped during coefficient design to [0.01, sample_rate / 2 - 100] Hz; JUCE
also floors EQ design frequency at 2 Hz. This differs from the first-order
filters' curve domain. Frequency and Q targets must be finite positive float32
values. Gain targets must produce finite positive native linear gain; the -100 dB
floor and lower values are excluded from curves. Rates too low to define the
original clamp are rejected.

Before reset, preparation or target mutation, the complete synchronized parameter
combinations are validated under the native processing lock, including scalar
values for omitted curves. Each must produce finite normalized float32
coefficients with a strictly stable denominator. Unsafe combinations are rejected
without clearing filter history or applying an earlier part of the curves.
Empty audio with empty curves retains the existing no-op behavior. Concurrent
native processing calls share this validation/processing lock.

Static coefficient stability does not guarantee that arbitrary fast modulation
is free of transients, level growth or other audible artifacts. Q and boosts can
increase peaks substantially. Fixed controls should use the existing scalar path;
curves intentionally add validation and scheduling work.


Ladder filter controls
~~~~~~~~~~~~~~~~~~~~~~

``LadderFilter`` accepts ``cutoff_hz``, ``resonance`` and ``drive`` curves with
all six fixed modes (LPF12/LPF24, HPF12/HPF24, BPF12/BPF24). Mode remains a scalar
architecture choice: changing it keeps the original JUCE state-reset behavior,
and a ``mode`` curve is rejected.

Targets arrive before their input frame. JUCE's existing 50 ms linear ramps on
the transformed cutoff and scaled resonance remain active; delivering a new Hz
or resonance target is not a promise of an instantaneous audible change. Drive
is a unitless multiplier, not dB, and retains the existing immediate saturation
and gain compensation. No additional smoothing or new filter algorithm is added.

Curve cutoff must be finite positive float32, resonance must be finite in [0, 1],
and drive finite and at least 1. The original cutoff mapping has no Nyquist clamp;
these are distinct validation rules from the first-order filters. The curve path
also requires a positive finite float32 sample rate whose 50 ms smoother length
fits a native int. Omitted scalar controls are validated under the processing
lock before reset, preparation or parameter changes. Scalar setter acceptance
is unchanged.

``reset=False`` preserves both nonlinear filter history and smoother progress;
block-size growth alone also preserves them. Rate/channel changes still prepare
the DSP. High resonance can self-oscillate, and higher drive intentionally adds
saturation; numerical parity does not imply every extreme setting is musically
appropriate. Fixed controls should use the existing scalar path.


Reverb controls
~~~~~~~~~~~~~~~

``Reverb`` accepts synchronized ``room_size``, ``damping``, ``wet_level``,
``dry_level`` and ``width`` curves. All must be finite in [0, 1]. ``freeze_mode``
remains a scalar setting and cannot be supplied as a curve. The algorithm and
its 10 ms linear damping, feedback and gain ramps remain unchanged. Room size
sets feedback/decay rather than physically resizing the delay lines. Damping
controls high-frequency decay. Wet and dry levels retain JUCE's existing gain
scaling; width changes the stereo wet mix. In the pinned JUCE mono path, only the
first wet gain is used, so width also changes mono wet level.

All targets at a frame are applied before its audio, in the original constructor
setter order: room size, damping, wet level, dry level, width. The order is
independent of dict insertion order. In particular, wet level and width share
smoothed gain targets; the original intermediate setter updates are preserved
rather than batched into one JUCE parameter update. Target timing is not a promise
of instantaneous audible response.

``reset=False`` retains delay-line memory and running ramps. Block-size growth
alone also retains them, including on scalar calls. ``reset=True`` clears the
reverb buffers but retains the original running smoothers; it does not snap them
to their targets. Rate/channel changes still prepare the DSP, clear buffers and
reset smoothers as before. Supply zero-valued input with ``reset=False`` to render
a tail; the automation call returns one output frame per input frame and does
not append an unbounded tail automatically.

The curve path requires mono or stereo audio and a sample rate in
[196, 1309442) Hz, the safe range for the pinned JUCE integer delay-buffer
calculation. Before reset, preparation or target changes, format and all effective
controls (including omitted scalar values and fixed freeze mode) validate under
the native processing lock. Existing scalar setter acceptance is unchanged.
No clamping, additional smoothing or reverb algorithm is introduced. Fixed
settings should use the scalar path to avoid curve validation/scheduling overhead.


Delay controls
~~~~~~~~~~~~~~

``Delay`` accepts synchronized ``delay_seconds``, ``feedback`` and ``mix`` curves.
Delay time must be finite in [0, 30] seconds; feedback and mix must be finite in
[0, 1]. Targets retain float32 precision. All same-frame setters apply before
processing, in constructor order (delay, feedback, mix), independently of dict order.
Omitted scalar controls and rate safety validate under the processing lock before
reset, preparation or target mutation.

This remains the original integer digital delay: seconds are converted to float32,
multiplied by the double-precision sample rate, then truncated to integer samples.
For example, 0.01 seconds at 48 kHz maps to 479 samples because of the existing
float32 rounding and truncation. No fractional interpolation, read-head crossfade
or parameter smoothing is added. Rapid changes can skip/repeat stored samples and
produce clicks or other time-varying artifacts; sample-position target delivery
does not imply artifact-free delay-time modulation.

Exactly zero seconds retains the original bypass: audio is returned unchanged,
regardless of mix/feedback, and delay memory does not advance or record that input.
Resuming a nonzero delay continues the paused memory. A nonzero sub-sample delay
that truncates to zero is different: the original pop-before-push code reads an
old slot, producing a full-ring delay, rather than the zero-second bypass. The ring
length is max(4, int(30 * sample_rate) + 1). This behavior is preserved, not clamped.

Mix is linear: dry gain is 1 - mix, wet gain is mix. Feedback sends delayed output
back into the stored input. Feedback 1 remains allowed and can sustain repeats;
input accumulation and arbitrary controls can still increase peaks. No limiter is
added. Channels retain independent delay memory, including multichannel audio.

``reset=False`` and block-size growth retain echo history. ``reset=True`` clears
it; rate/channel changes still allocate and reset as before. Supply zero input to
render the tail: no extra output frames are appended automatically. The curve path
requires a positive finite rate with 30 * sample_rate <= (INT_MAX - 1) / 2 to keep
the pinned JUCE int buffer/index arithmetic safe. This does not impose a memory
budget; the original implementation allocates up to 30 seconds per channel even
for short requested delays. Scalar setter acceptance is unchanged. Use the scalar
path for fixed settings to avoid curve validation/scheduling overhead.


Chorus controls
~~~~~~~~~~~~~~~

``Chorus`` accepts synchronized ``rate_hz``, ``depth``, ``centre_delay_ms``,
``feedback`` and ``mix`` curves. Rate targets must be finite in [0, 100] Hz,
depth/mix finite in [0, 1], feedback finite in [-1, 1]. Centre delay accepts any
finite float32 target, preserving the original internal [1, 100] ms clamp. Its
Python getter retains the requested value, even when the DSP clamps it. Omitted
controls keep their scalar values and validate under the native processing lock
before reset/preparation/target changes. Existing scalar setter acceptance is
unchanged; the new curve path rejects out-of-domain depth/feedback targets rather
than relying on JUCE debug assertions.

All five same-frame targets are applied before that frame in constructor order,
independently of dict insertion order. The original oscillator frequency, depth,
feedback and linear dry/wet gains keep their 50 ms ramps. Centre delay changes
immediately; large/rapid changes can produce original read-head artifacts. No
additional interpolation, crossfade, smoother or algorithm is introduced. The
original linear fractional-delay interpolation and shared LFO across channels
remain unchanged; this does not add a stereo phase offset or new widening mode.

``reset=False`` preserves LFO phase, delay memory, previous feedback output and
smoother progress. ``reset=True`` retains the original phase/delay reset and
smoother target initialization. Sample-rate/channel-count changes still prepare
and reset. To preserve memory on block-size growth, the DSP is prepared with a
fixed 1024-frame scratch capacity and larger calls are split internally. Caller
``buffer_size`` still controls outer processing; internal spans never exceed the
prepared buffers. This adds no output latency and does not change frame counts.

The curve path requires a finite sample rate >= 100 Hz to bound the pinned
oscillator's per-sample phase wrapping over its full 0–100 Hz range. The float32
rate and 50 ms ramp length must be representable, and the maximum 110 ms delay
allocation/index sums must fit native ints. These are format/arithmetic limits,
not a memory budget. Fixed settings should use the scalar path to avoid curve
validation/scheduling overhead. Numerical parity does not guarantee that extreme
feedback or modulation settings are musically appropriate.


Phaser controls
~~~~~~~~~~~~~~~

``Phaser`` accepts synchronized ``rate_hz``, ``depth``, ``centre_frequency_hz``,
``feedback`` and ``mix`` curves. Rates must be finite in [0, 100] Hz, depth/mix
finite in [0, 1], feedback finite in [-1, 1]. Centre targets must remain finite,
positive float32 and strictly below the current Nyquist frequency. The native
modulated frequency retains its logarithmic mapping/clamp to
[20, min(20000, 0.49 * sample_rate)] Hz. The getter retains the requested centre.
All supplied same-frame targets apply before audio in constructor order, independent
of dict order. Omitted scalars and format limits validate under the processing lock
before reset, preparation or parameter changes. Scalar setter acceptance is unchanged.

The six original all-pass stages and shared LFO are retained. The LFO and filter
frequency update every four audio frames; frequency/depth smoothers advance on
that clock. Feedback and linear dry/wet gains advance at audio rate. All retain
JUCE's original 50 ms ramps. A centre target between frequency ticks can wait up
to three frames; a short pulse that returns before the next tick can have no audible
frequency effect. Native target delivery does not bypass this cadence or promise
instantaneous audible response. No new smoothing or phase-widening mode is added.

``reset=False`` and block-size growth preserve LFO phase, the four-frame update
counter, filter memory, feedback output and smoother progress. Phaser and Chorus
share fixed 1024-frame DSP scratch preparation and internally bounded processing.
Rate/channel changes and ``reset=True`` retain the original reset/smoother behavior.
The curve path requires a finite sample rate >= 400 Hz to bound the downsampled
oscillator's phase wrapping; its float32 clock and 50 ms native int step counts
must be representable. Fixed settings should use the scalar path to avoid curve
validation/scheduling overhead.

There is one explicit initialization correction in the dense path. Original JUCE
normalizes centre frequency using the DSP rate at setter time, then changes rate
in prepare without recomputing that mapping. On cold 22.05/32 kHz renders this can
leave a constructor-rate mapping until centre is assigned again. The curve path
reapplies the effective centre after preparation, including omitted scalar centre,
so requested Hz and continuation do not depend on chunking. Original comparisons
at these rates prepare the format and reapply centre explicitly before processing.
No-automation calls (including None/empty automation) retain the legacy mapping;
prepare the scalar effect at the new rate then reassign centre when needed. This
correction makes cold dense and legacy scalar output intentionally differ at low
rates; at 44.1/48/96 kHz the original normalization range is already the same.


Convolution controls
~~~~~~~~~~~~~~~~~~~~

``Convolution`` accepts ``automation={"mix": curve}``, finite targets in [0, 1]
at float32 precision. The constructor, IR filename/array properties, normalization
and IR sample-rate conversion retain their interfaces. IR content is fixed: no
impulse-response curve/replacement is introduced. The final mix target persists.

Mix uses the original linear DryWetMixer with 50 ms gain ramps. The input is copied
into that mixer, wet convolution runs over an available block, then the mix targets
are applied to corresponding output-frame subspans. Mix does not affect the wet
engine's input or history. This avoids recalculating partial FFTs at each target
change; convolution still runs at mix zero so later wet output retains its tail.
No new mixing rule, smoothing or backend is added.

Preparation occurs on sample-rate/channel-count changes. The first prepared capacity
is bounded to [128, 8192] frames; normal default8192 processing keeps its original
partition, while tiny/large first blocks are bounded. Later larger calls are split
at that capacity without rebuilding the engine or clearing the tail/mixer progress.
``reset=False`` preserves histories; ``reset=True`` and format changes retain the
original clearing/initialization behavior. To render a tail, supply zero input and
matching curves with ``reset=False``; no output frames are appended automatically.

The curve path supports mono/stereo and the existing zero-latency engine. Leading
silence or phase delay in an IR is part of the IR, not added engine latency. Rates
must be finite and >= 100 Hz, with representable native 50 ms ramps and projected
resampled-IR size. These format/size checks run under the native lock before reset
or target changes and are not a memory budget. Scalar setter acceptance stays unchanged.

FFT rounding depends on partition and partial-call sizes. Fine scalar calls and
bulk convolution followed by mix automation can therefore differ by small float32
residuals even with the same control timeline. Validation includes long tails,
independent original comparisons and direct FIR timing; do not interpret byte
inequality alone as a changed algorithm. Fixed settings should use the scalar path
to avoid dense curve copies/validation. Small caller audio chunks still require
small zero-latency FFT processing; the curve path does not buffer extra audio to
make those calls larger or add output latency.


BrickwallLimiter controls
~~~~~~~~~~~~~~~~~~~~~~~~~

``BrickwallLimiter`` accepts ``ceiling_db`` and ``release_ms`` curves. Ceiling
values must produce a positive finite float32 linear gain; release values must
be positive finite float32 values whose sample counts fit the native signed int.
``lookahead_ms`` and ``true_peak`` remain scalar configuration.

Unlike scalar setters, curve targets align with compensated output time: the
complete ceiling/release configuration travels through a queue with the delayed
audio. The original peak detector, gain envelope, hold and safety clip are retained.
This does not add smoothing or a new true-peak guarantee. Scalar-only processing
retains the original input-clock behavior; original scalar replay of curves must
delay its parameter updates by the actual lookahead latency to match this path.

Each curve still has one target per source input frame. With ``reset=False``,
initial calls may return fewer frames or no frames; concatenate returned audio
and supply zero input to drain outstanding latency. Pending targets survive
subsequent scalar calls. Getters report the last consumed source targets, which
may still be queued. ``reset=True`` automatically drains latency with silence
and holds the last targets, returning the input frame count. Explicit reset,
sample-rate/channel changes and structural configuration changes start a new
history. Growing block sizes preserve the detector, delay and control queues by
using bounded internal scratch. Scalar history is retained for a warm transition
into automation. Curves omitted from a call hold their requested scalar values;
all effective values are checked before DSP history changes.


PitchShift controls
~~~~~~~~~~~~~~~~~~~

``PitchShift`` accepts ``automation={"semitones": curve}`` with finite targets in
[-72,72] at float32 precision. Curves have one value per source input frame and
are shared across channels. Pitch targets travel through the existing one-second
priming delay alongside source audio, then are applied immediately before those
frames enter Rubber Band. Its original real-time/high-quality backend, spectral
windows and variable output availability are retained. Scheduling a target is not
a promise of an instantaneous audible pitch transition at an exact output sample.
The latency hint is an allocation hint, not a control-to-output mapping.

``reset=False`` preserves pending audio, control targets and pitch-engine state;
concatenate returned frames and feed zero input to drain outstanding audio. Scalar
continuations hold the requested target while pending earlier targets drain.
``reset=True`` drains automatically with held final targets and returns the input
frame count. Getters retain the last consumed source target. Scalar-only pitch
changes keep their original engine-clock semantics. The fork fixes repeated
preparation clearing the priming delay, which previously discarded short stream
inputs; a warm transition into curves retains recent scalar target history.

Varying curves use canonical one-frame steps inside the native engine, including
priming and drain, so outer call boundaries do not move target updates. Uniform
constant controls and histories retain bulk processing; fixed scalar values remain
the least expensive route. No per-frame Python callbacks are used.

The first prepared capacity is bounded to[1,8192]; larger subsequent contexts reuse
bounded subspans without rebuilding the engine. Valid output is packed according
to the host's right-aligned frame-count contract. Explicit reset and format changes
clear the timeline. Explicit reset also reconstructs the nested Rubber Band engine
on next preparation: its upstream reset retains configuration selected by previous
pitch changes and is insufficient to reproduce a fresh dynamic render. This adds
setup cost to reset renders; warm streaming retains the existing engine.
Fixed pitch should use the scalar API. ``time_stretch`` remains
a separate interface with its own processing, parameter resolution and duration
contract; this feature does not change it.


Resample controls
~~~~~~~~~~~~~~~~~

``Resample`` accepts ``automation={"target_sample_rate": curve}`` in physical Hz,
with one float32 target per source input frame, shared across channels. Targets
must be finite and in [sample_rate/64, sample_rate], including after conversion
to float32. The sample rate passed to ``process`` is the fixed host rate, not a
curve. All eleven existing ``quality`` choices remain available as fixed scalar
configuration, with their original interpolation/filter arithmetic. No smoothing,
new interpolation backend or quality automation is added.

The curve path integrates targets on the source timeline and timestamps the virtual
sampling events. Downsampling and reconstruction use that shared clock; changing
the rate does not discard fractional phase or pending audio. Timestamped native
and target rings are bounded for the validated ratio range. Blocks can grow or
shrink without re-preparing filters. The original fractional latency-rounding
phase convention is retained at initialization. Very small float32 differences
from the scalar resampling partition can remain; this is not a new sound-quality
or ideal reconstruction guarantee. Existing aliasing/filter behavior depends on
``quality``, and abrupt rate changes can still produce the intended audible effect.

Clocked processing has a fixed startup deficit ``L = 65 * B + 2`` frames, where
``B`` is the interpolation mode's base latency:

.. list-table:: Base latency used by clocked resampling
   :header-rows: 1

   * - Quality
     - B
   * - ZeroOrderHold
     - 0
   * - Linear
     - 1
   * - CatmullRom, Lagrange
     - 2
   * - WindowedSinc
     - 100
   * - WindowedSinc8/16/32/64/128/256
     - 16/32/64/128/256/512

For default WindowedSinc32, L=4162 (about86.7ms at48kHz). ``reset=True`` feeds final
silence, holds the last target and compensates this delay, returning exactly the
input frame count. ``reset=False`` initially returns fewer or zero frames; slice
input and curves together, concatenate outputs and supply L zero frames to drain
outstanding audio. Scalar ``reset=False`` calls after curve activation hold the
requested rate using the same clock and pending history. Getters report the last
consumed source target, which may still precede delayed output.

Entering automation after already processing scalar audio requires ``reset=True``:
the original scalar path has a different latency/state timeline. This is rejected
before parameters or DSP history change when ``reset=False``. Starting a fresh
object with curves and ``reset=False`` is supported. Explicit reset, format changes
and quality changes clear the timeline; quality is not a seamless switching event.
Requested scalar holds on an active clock must also stay within the curve range.
A scalar restart retains the upstream wider rate range and processing behavior.
``None``/``{}`` and fixed scalar processing retain that existing path when no
clocked stream is active. No GSM controls or GSM automation are introduced.


Effect-chain automation
-----------------------

``Pedalboard``, ``Chain`` and ``Mix`` accept
``automation={effect_instance: {parameter_name: numpy_curve}}`` on ``process`` and
``__call__``. Targets must be unique reachable supported builtin, VST3 or Audio Unit
instances, not containers, indices, strings or foreign effects. External plugin
curves use displayed parameter units. Uncontrolled effects and ``None``
branches retain scalar behavior. Containers and their public child identities are
not replaced. Duplicate references and cycles are rejected before processing.

All curves have one target per root input frame. Serial nodes consume targets
with valid upstream audio, and parallel branches retain pending audio and targets
across calls. This source-frame ownership does not remove a processor's documented
smoothing or windowed response. ``reset=True`` resets and compensates startup
latency, holding final targets during synthetic silence; it does not export full
effect tails. Send explicit zero audio to render those tails.

For streaming, explicitly call the root's ``reset()`` first, then process every
slice with ``reset=False``. Curves and audio are sliced on the same root-input
boundaries; returned audio may initially contain fewer frames. Omitted active
lanes hold their last scheduled target, including when automation is ``None`` or
an empty dict. Empty audio does not advance or flush a session. Curve arrays are
validated and copied before rendering; pending copies survive caller mutation.
All controlled effects receive joint preflight before setters, reset or DSP
preparation, and invalid maps preserve pending queues and processing memory.

Keep node/parameter lanes, topology, sample rate and channel count fixed within a
session. Change these with ``reset=True`` or explicitly reset the root before a
new stream. The first prepared native capacity is retained (at most 8192 frames);
larger calls reuse it in sub-blocks, so buffer-size growth does not reset history.
Configure scalar defaults before starting; active lanes use curves.
Independently processing/resetting a participating child or nested container during
a root session is unsupported and requires restarting the root. VST3 parameter
curves are supported as described below. AU events, buses/transport and arbitrary
state cloning remain separate capabilities. Serialize calls that continue the same streaming session. Scalar calls
on graphs without an active session retain the original processing path.


VST3 display-unit curves
-----------------------

``VST3Plugin.process`` and ``__call__`` accept ``automation={parameter_name: curve}``
for both audio effects and MIDI instruments. Existing Python parameter names and
displayed numeric values are used: e.g. dB, Hz and percentages. These are not
normalized0–1 curves. ``Pedalboard``/``Chain``/``Mix`` use the same instance-keyed
mapping as for builtins, and can mix VST3 and builtin targets.

Each real NumPy curve has one value per input frame, shared across channels. For
MIDI, the count equals the renderer's duration-derived frame count and values use
the instrument's input clock alongside MIDI messages. Literal infinite display
endpoints are accepted only when they identify a reported endpoint. Boolean arrays
and0/1 values are accepted for recognized boolean parameters; other curves must
contain real numeric values. Text-only choices, program selectors and non-automatable
parameters are rejected. Ambiguous names require a plugin with unambiguous names;
no parameter is silently selected. Existing scalar configuration remains available.

Conversion does not invoke the legacy Python range-guessing wrapper or its
parameter-mutating fallback. It checks a controller's numeric display model and
uses the SDK plain conversion only if it agrees with display units; otherwise it
uses the controller's text parser. Every requested value is checked for range and
round-trip agreement at display precision. Unreliable conversions raise errors.
Reported unit labels take precedence; unlabeled Hz/kHz displays use Hz, consistent
with existing scalar naming. The plugin's own quantization is retained, and final
normalized targets have the existing host's float32 precision. There is no
interpolated lookup-table approximation or promise of more precision than a plugin
provides. Curves validate against the current controller model before parameter
setters, reset or DSP preparation. After preparation, the model is checked again;
format-dependent changes trigger reconversion before audio processing. If a target
is valid only in the new format's range, call
``vst.prepare_automation(sample_rate, num_channels=2)`` first and render with
``reset=False``. This method resets processing state. A conversion failure after
preparation does not restore prior DSP state; correct the curves and restart.
Unit mappings must remain stable during an active graph session; a model change
requires restarting the root.
Stable VST3 ParamIDs, not raw parameter pointers, survive instance reloads.

``automation_mode`` is host configuration, set before a session:

* ``"sample_steps"`` (default) coalesces equal synchronized targets and processes
  those spans in native code. It supports plugins reading only a block's final
  parameter value. Dense curves can require one VST processing call per frame.
* ``"vst3_events"`` sends all targets through ordered VST3 parameter queues with
  their sample offsets, retaining larger blocks. Use this only when the plugin
  consumes the points correctly; some plugins read only the final point. The host
  cannot infer that support from the standard automatable flag.

Neither mode invokes Python per audio frame. Both retain plugin smoothing,
windowing and block-dependent DSP; exact host delivery is not a guarantee of an
instantaneous audible response. VST3 queue semantics are documented by
`Steinberg <https://steinbergmedia.github.io/vst3_doc/vstinterfaces/classSteinberg_1_1Vst_1_1IParameterChanges.html>`_.

Automated streams retain an8192-frame prepared capacity and subdivide larger calls
to avoid reprepare on block growth. Use ``reset=False`` on matching slices after
explicitly resetting the plugin/root. A warm scalar-to-curve transition, mode or
audio-format change requires a reset. Scalar calls following an automated stream
hold final targets on that prepared engine. Reset preserves requested parameter
values. Presets/state and participating children must not be independently changed
during a root session. MIDI timestamps retain double precision and sample-aligned
values are snapped only within floating-point rounding error before integer-frame
conversion. Genuine fractional-frame positions retain the existing truncation rule.

Standalone VST3 reset and audio/MIDI rendering share the instance lock. This does
not remove a third-party plugin's own main-thread requirements. Scalar and automated
calls validate native MIDI sizes and retain sample-aligned timestamp precision.
Standalone MIDI clips can opt into fixed reported-latency
compensation with ``compensate_latency=True`` and ``reset=True``; raw MIDI timing
remains the default. See :doc:`../instrument_latency`. Native AU events, transport
and bus hosting remain separate.


Audio Unit display-unit curves
-----------------------------

On macOS, ``AudioUnitPlugin.process`` and ``__call__`` accept the same physical-unit
``automation`` keyword on audio and MIDI entry points. Graphs use
``automation={au_instance: {parameter_name: curve}}``. NumPy ownership, source-frame
clock, joint validation, streaming omissions, stable lanes and root reset rules
match VST3 and builtin graphs. Standalone MIDI rendering also accepts the shared
``compensate_latency=True`` option with ``reset=True``; see
:doc:`../instrument_latency` for the offline and raw-streaming boundaries.

AU uses ``automation_mode="sample_steps"`` only; other modes raise an error.
Changes at a frame are sent together through native AU setters before that span is
rendered. Equal targets share a span. No Python calls occur per frame. Native
scheduled/ramped AU events are not exposed. Exact host delivery does not remove
plugin smoothing, internal update intervals or block-dependent DSP.

Only stable global-scope UInt32 parameter IDs are used. Native parameter metadata
supplies range, units, writable and non-real-time flags. Non-real-time/read-only
parameters, ambiguous names and unreliable/non-numeric display models are rejected.
When metadata values agree with display units, metadata provides the conversion;
otherwise native AU string conversion is used. When an AU only names special
endpoints and omits a parser, direct metadata values are tried and must still
round-trip through the actual display. Range guessing and parameter mutation
for unit discovery are not used. Text-only indexed choices are unsupported.
Boolean curves accept bool or0/1; literal infinite displays are supported only at
unambiguous endpoints. Native AU physical values and normalized targets are float32.
For nonlinear display conversion, round-trip checks account for one physical float
ULP by measuring the adjacent value through the plugin's own display function.
This is native quantization, not a user-selected approximation tolerance.

``prepare_automation(sample_rate, num_channels=2)`` and post-prepare range checking
apply equally to AU. Restart with reset=False after explicit preparation. A failed
post-preparation conversion does not roll back the prior DSP state. During a graph
session, presets/state and participating children must not change independently.

Actual AUv2 diagnostic effects/instruments, CHOWTapeModel.component and Magical8bit
Plug2.component are tested on macOS arm64. AUv2 fixtures register only in the test
process, without copying bundles to user/system plugin directories.

Actual AUv3 extension verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original no-UI AUv3 diagnostic extension is packaged in a containing app, signed
ad-hoc and discovered through the public ``AudioUnitPlugin`` scanner. It is loaded
out-of-process; a dedicated diagnostic mode returns the actual render PID in the
audio buffer, which must differ from the Python host PID. This does not rely on
process-local AUAudioUnit subclass registration. Typed dB/Hz and boolean curves,
reported delay, MIDI note offsets/holds, graph queues, independent scalar
continuation, parameter state, reset/reinstantiation and destruction are verified.

The AU bridge's global UInt32 IDs can represent a plugin's original 64-bit
AUParameterAddress values. Tests exercise a frequency at0x100000002 alongside a
separate boolean at0x2, ensuring host controls do not truncate one into the other.
This is tested bridge behavior; no direct 64-bit-address public API is introduced.

The containing app must be registered with macOS for the extension to be discovered.
The scanner compares decoded canonical filesystem paths, accepting relative paths,
spaces/Unicode and symlinks, and returns all matching components in one ``.appex``.
Unregistered extensions report an AUv3-specific error instead of suggesting an
AUv2 Components directory. Async scans synchronize completion across threads and
propagate worker exceptions. Tests isolate each check in a subprocess with a
30-second bound; this is not a new universal plugin-load timeout guarantee.

Automation retains native parameter ranges after format preparation. AUv3 parameter
updates already flow through its framework bridge, so the host avoids an additional
legacy global AUParameterListenerNotify broadcast, which can echo into its own
listener and stall each sub-block. AUv2 retains the existing listener notification.
Native scheduled events remain unavailable; AUv3 still uses ``sample_steps``.

The AUv3 fixture requires a macOS user session with Xcode tools, CMake, codesign,
PlugInKit and LaunchServices. It creates per-build identifiers, temporarily
registers only generated app/extension bundles, refuses preexisting registrations
and removes its entries on teardown. No developer signing certificate, App Store
installation or copying to system plugin directories is required. Missing build or
registration support fails the macOS tests, rather than silently skipping coverage.
Other operating systems skip AU tests.

Manual acceptance on macOS arm64 additionally covers an unchanged source build
of `Morass v1.0.3 <https://github.com/bogware/morass/tree/v1.0.3>`_: nine constant
dB/Hz checks across 44.1/48/96 kHz and buffer sizes 1/31/8192, three dense-curve
comparisons against per-sample scalar setters, sparse streaming, boolean bypass
and parameter-state/reset checks. All audio comparisons match exactly. The
published installer contains AUv2/VST3 rather than AUv3; source-build acceptance
is distinct from testing a vendor-distributed AUv3 binary.

Morass AUv3 and CHOWTapeModel AUv2 native editors were visually checked for
parameter-to-UI updates, automation with the editor open, close/reopen and reopening
after automation preparation/reset. A human-operated Morass AUv3 check additionally
verifies GUI control -> native host readback -> rendered audio: editing Input Gain
to -12.0 dB produces normalized readback 0.25 and quieter output, followed by a
-12 to -3 dB automation curve with no intervening reset. The GUI and native getter
both show -3.0 dB afterward. Immediate output RMS is not an exact gain oracle
because the plugin retains tails and envelope state.

Physical UI automation could not reliably deliver input in this desktop session;
the successful GUI action was performed by the human operator. Direct GUI-to-host/
audio roundtrip for CHOW AUv2, other third-party AUv3 vendors and non-arm64
architectures remain unverified. Plugin-specific main-thread requirements,
internal smoothing and block behavior still apply.
