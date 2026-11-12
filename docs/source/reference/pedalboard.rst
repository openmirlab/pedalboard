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
scalar processing path. Supplying curves to a whole ``Pedalboard``/``Chain``/
``Mix``, other builtin effects or external plugins is outside this first batch.
