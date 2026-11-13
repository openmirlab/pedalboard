# Unmodified JUCE NoiseGate sidechain acceptance

This wrapper includes `JUCE/examples/Plugins/NoiseGatePluginDemo.h` unchanged from
the pinned submodule. Its original ISC notice remains in that file. The wrapper
and native host are GPL-3.0-only. Unlike the diagnostic fixtures, this is an
independent existing musical-effect implementation, with a smoothed detector and
a one-second hold. It is a JUCE example, not commercial-vendor certification.

The native host loads the same built VST3 as Python, enables main/sidechain ports,
and sets alpha=0 and threshold=0.5. Both hosts must produce exactly the expected
gate onset and one-second hold at 44.1/48kHz, with and without an external key,
across caller sizes 1/31/8192/65536. Source and detector remain distinct signals.
No editor opens and nothing is installed in system plugin folders.

Build with a C++ toolchain and CMake from the repository root:

```sh
cmake -S tests/juce_sidechain_plugin -B /tmp/pedalboard-juce-sidechain -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/pedalboard-juce-sidechain --config Release --parallel 2
PEDALBOARD_TEST_JUCE_SIDECHAIN_BUILD=/tmp/pedalboard-juce-sidechain \
  uv run --no-project pytest tests/test_hosted_audio_buses.py -k unmodified_juce -q
```

The optional tests explicitly skip if this build path is absent. Core source-built
SDK VST3/AUv2 and signed AUv3 tests do not require this optional build.

The pinned JUCE version predates current macOS SDKs. On macOS, configure its nested
helper target first to retain an older deployment target:

```sh
cmake -S JUCE -B /tmp/pedalboard-juce-sidechain/JUCE/tools \
  -DJUCE_BUILD_HELPER_TOOLS=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
cmake -S tests/juce_sidechain_plugin -B /tmp/pedalboard-juce-sidechain \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
```

VST3 and the native host have been exercised on macOS arm64. The optional AU build
is disabled by default: current SDKs lack the legacy `AudioUnit.r` Rez resource
required by this JUCE wrapper. Enable only with an appropriate toolchain using
`-DPEDALBOARD_BUILD_JUCE_SIDECHAIN_AU=ON`. AU bus coverage uses the separate native
AUv2 and AUv3 fixtures; no JUCE NoiseGate AU acceptance is claimed. Other OS builds
remain unverified.

## Combined audio and MIDI example

Configure with `-DPEDALBOARD_BUILD_JUCE_COMBINED=ON` to build a VST3 wrapper for the
unchanged pinned `AudioPluginDemo.h`. It mixes incoming audio with its musical
sine-wave synthesizer and retains the original MIDI output behavior. The native
host accepts an extra `combined` CLI argument and routes the same input notes and
audio. `test_hosted_audio_midi.py -k unmodified_juce` requires exact audio and
returned note timing for with/without MIDI,44.1/48kHz and1/31/8192 caller spans.
The same PEDALBOARD_TEST_JUCE_SIDECHAIN_BUILD variable selects this optional build.
This is verified on macOS arm64; no editor, DAW or system installation is used.

The example's own reset clears its delay memory but does not stop held synth
voices. A targeted regression requires the new hosted MIDI path to cold-reload
on an explicit reset after delivering events, while reset=False retains a held
voice. This uses the existing reload mechanism and its main-thread restriction;
it does not alter the audio-only convenience reset classification.
