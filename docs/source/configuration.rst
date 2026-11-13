Configuration persistence
=========================

``pedalboard.configuration`` saves static settings and reconstructs a fresh graph.
It does not guarantee preservation of running DSP memory: delay lines, reverb
tails, active notes or transport position. Host automation queues are excluded. This is a configuration
file, not an exact continuation of a running session.

Basic workflow
--------------

.. code-block:: python

    import numpy as np
    from pedalboard import Pedalboard, Gain, Reverb, configuration

    original = Pedalboard([Gain(gain_db=-3), Reverb(room_size=0.4)])
    configuration.save(original, "chain.json", sample_rate=48000, num_channels=2)
    restored = configuration.load("chain.json")

    # Explicitly resets and renders BOTH objects. Use disposable instances when
    # their running history matters. Use representative audio for your project.
    probe = np.random.default_rng(1).normal(0, 0.1, (2, 48000)).astype("float32")
    result = configuration.verify(original, restored, probe, sample_rate=48000)
    print(result.exact, result.max_abs_error, result.rms_error)

``save`` does not reset or render the source. The destination file is replaced
atomically after successful capture and validation. ``load`` validates the whole
file and its dependencies before loading external plugins, then checks parameter
readback. It does not claim that sound has been verified. Neither operation
changes existing ``raw_state`` or ``preset_data`` behavior.

Supported data
--------------

* Pedalboard, Chain and Mix topology, including repeated references to one node.
  Cycles, excessively deep graphs and unknown processor classes are rejected.
  Aliases are preserved in the file; the existing renderer's rejection of duplicate
  instances within a processing graph still applies.
* The public scalar properties of Bitcrush, BrickwallLimiter, Chorus, Clipping,
  Compressor, Delay, Distortion, GSMFullRateCompressor, Gain, HighShelfFilter,
  HighpassFilter, Invert, LadderFilter, Limiter, LowShelfFilter, LowpassFilter,
  MP3Compressor, NoiseGate, PeakFilter, Phaser, PitchShift, Resample and Reverb.
* VST3 and platform-supported AudioUnit identity, opaque state and readable
  automatable parameter values. Plugin state remains owned by the plugin.

Convolution is rejected because its original impulse response is not exposed by
the current binding. Generic IIRFilter and user-defined subclasses are also
unsupported. Unsupported processors produce an error rather than partial files.
Automation curves and host scheduling modes are not stored.

External plugins
----------------

Operate on an idle graph; do not mutate it concurrently. External plugin capture
and restore require the main thread and service pending JUCE messages for a bounded
``settle_seconds`` interval (default 0.2, allowed range 0 through 2). This allows
plugins that defer changes to the message loop to synchronize. It is not a promise
that arbitrary asynchronous plugins have finished; readback and sound verification
remain necessary. These operations are not suitable for an audio callback.

Fresh destinations are prepared at the saved sample rate and channel count before
state is applied. Parameter checks use stable native IDs. Float32 rounding is
allowed; larger raw-value differences require matching native display text and
validated native text-parser roundtrips. Desired values are never written again
to conceal a failed opaque-state restore. Read-only meters are not compared.

Files record the exact Pedalboard native binary, package version, platform,
plugin identity/version, absolute bundle path and a SHA-256 fingerprint of its
files. Loading requires the same runtime and unchanged plugin bundle. Moving or
updating a plugin requires creating a new configuration. The format is versioned;
there is no migration or cross-platform compatibility promise in this first version.
The initial limits are 64 MiB per document, 1024 nodes, graph depth 64,
1 through 64 channels and sample rates from 1 through 768000 Hz.

Plugin binaries, samples, impulse responses and other private assets are not
embedded or discovered. A plugin can depend on files not described in its state;
parameter readback alone cannot establish that these assets are available.
Only load configurations from trusted sources: checksums detect corruption, not
malicious input, and opaque state is passed to third-party native code.

Sound verification
------------------

``verify`` checks settings before and after rendering, resets both operands and renders identical input after
one second of silence by default. A positive warm-up is at least 8192 frames
to avoid changing block capacity between warm-up and the probe. Audio is a mono vector or channel-first array.
For a standalone instrument, omit the array and supply ``midi``, ``duration`` and
``num_channels``. MIDI accepts Pedalboard's regular message/time pairs.

The default acceptance is NumPy's elementwise comparison with ``atol=1e-6`` and
``rtol=1e-5``. Set ``exact=True`` to require bit-identical output. Success returns
an immutable ``VerificationResult`` containing exactness, maximum absolute error,
RMS error and sample count; any settings or output failure raises
``ConfigurationError``. A successful probe establishes reproducibility for that
input and format, not every possible future input. Randomized or analog-modeling
plugins may require a project-specific tolerance and additional listening checks.


Verification coverage and known limits
--------------------------------------

The configuration tests exercise all 23 supported scalar builtins at 44.1/48 kHz,
nested graphs in a fresh interpreter, and CHOWTapeModel/Magical8bitPlug2 as VST3
and AUv2 at both rates with two fresh readers and A/B/A state restoration. A real
source-built AUv3 extension exercises effect, delayed effect and instrument
persistence in fresh processes at 48 kHz. This is not third-party AUv3 or GUI
persistence acceptance, and does not establish universal plugin compatibility.

One AUv2 reset/reinstantiation discovery failure occurred during concurrent plugin
test activity. An isolated full suite, five targeted repeats, and a subsequent
24-run stress check (three concurrent test processes) passed without reproducing
it; its cause remains unclassified. There is no automatic retry or fallback: load/render failures
are reported explicitly. Deterministic builtin outputs are checked exactly;
external comparisons report actual residuals against the declared tolerance.
