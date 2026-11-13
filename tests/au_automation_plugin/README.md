# Actual AUv2 automation fixture

Original diagnostic DSP is GPL-3.0-only. It is built against the vendored Apple
CoreAudioUtilityClasses (their original notices/licenses remain). Only fixture
source/build instructions are tracked. No binaries or system installation are needed.

Run `uv run pytest tests/test_au_automation.py -q` with the native extension installed
on macOS. CMake and a C++17 compiler are required; missing tools fail rather than
silently skip. Tests build a temporary `.component`, load its factory with
CoreFoundation and register it with AudioComponentRegister for this process only.
`register.py` also registers the repository's real CHOWTapeModel AU fixture.

Sixteen component descriptions share an original Audio Unit implementation:

- Automation AU: direct physical dB/Hz/boolean controls with nonconsecutive IDs.
- Automation AU Delayed: twenty reported samples of delay.
- Automation AU Instrument: constant input1 gated by timestamped MIDI notes.
- Automation AU Rate Dependent: frequency maximum follows the prepared Nyquist rate.
- Automation AU Infinite Gain: linear native gain0..2 displayed in dB, including-inf;
  native string parsing exercises float32 physical quantization and nonlinear units.

- Automation AU Partial Display: only the minimum is named-inf; remaining values
  use metadata dB, with no string parser. This is permitted by Apple's AU contract.
- Automation AU Delayed Instrument, Long Delayed Instrument and Huge Delayed
  Instrument: note-gated sources before 20, 4097 and 16385 reported samples of
  delay, used by `tests/test_instrument_latency.py`. They test offline alignment,
  raw streaming, post-duration MIDI exclusion and held drain controls.

Read-only Blocks counts native renders; Setup is marked non-real-time and Broken
has an unsupported display parser. The audio oracle is
`float(input * pow(10, actual_gain_db/20) * actual_frequency_hz/20000)`;
normalized targets and native physical values each have float32 precision.
Analytical audio comparisons allow3e-7 absolute/relative error. Same-binary stream/
reset comparisons require exact equality. AUv3 and native scheduled events are not
covered by this AUv2 fixture. Only macOS arm64 has been exercised locally.

The bus and combined-input variants cover main/key inputs and main/aux outputs.
Clock, Meter, Loop and Position diagnostics additionally encode actual native AU
host callback values. They call the extended transport callback when supplied,
including recording and transport-change state, and the musical callback for
meter/bar/next-beat timing. tests/test_hosted_transport.py verifies these values
against known clocks, loop phases and split anchors; no vendor AU certification
or GUI behavior is claimed.
