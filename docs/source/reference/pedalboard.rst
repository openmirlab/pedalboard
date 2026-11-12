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
