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
scalar processing path. Whole ``Pedalboard``/``Chain``/``Mix`` curve routing and external-plugin
automation are not supported. Additional dynamics controls are listed below.

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
