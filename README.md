# Pedalboard — OpenMIRLab fork

This fork adds native, per-frame NumPy parameter automation to selected built-in
Pedalboard effects, VST3 plugins and macOS Audio Units. It is maintained by [OpenMIRLab](https://github.com/openmirlab)
and builds on [Spotify's Pedalboard](https://github.com/spotify/pedalboard).
The existing DSP algorithms and scalar processing API are retained.

Native NumPy curves are available for the 21 selected built-in effects listed below.

## Automation in this fork

| Effects | Supported curve parameters |
| --- | --- |
| `Gain` | `gain_db` |
| `Clipping` | `threshold_db` |
| `Distortion` | `drive_db` |
| `Bitcrush` | `bit_depth` |
| `Compressor`, `NoiseGate` | `threshold_db`, `ratio`, `attack_ms`, `release_ms` |
| `Limiter` | `threshold_db`, `release_ms` |
| `LowpassFilter`, `HighpassFilter` | `cutoff_frequency_hz` |
| `PeakFilter`, `LowShelfFilter`, `HighShelfFilter` | `cutoff_frequency_hz`, `gain_db`, `q` |
| `LadderFilter` | `cutoff_hz`, `resonance`, `drive` |
| `Reverb` | `room_size`, `damping`, `wet_level`, `dry_level`, `width` |
| `Delay` | `delay_seconds`, `feedback`, `mix` |
| `Chorus` | `rate_hz`, `depth`, `centre_delay_ms`, `feedback`, `mix` |
| `Phaser` | `rate_hz`, `depth`, `centre_frequency_hz`, `feedback`, `mix` |
| `Convolution` | `mix` (fixed impulse response) |
| `BrickwallLimiter` | `ceiling_db`, `release_ms` (fixed lookahead and detector) |
| `PitchShift` | `semitones` (source time at the Rubber Band input) |
| `Resample` | `target_sample_rate` (source-time clock; fixed `quality`) |

Pass an `automation` dict to an effect's `process` or `__call__`. Each curve is a
one-dimensional real NumPy array with one target per input audio frame, shared
across channels. Targets use the existing physical units at float32 precision.
The native scheduler applies all targets at a frame before processing that frame,
without per-frame Python callbacks or additional smoothing.

```python
import numpy as np
from pedalboard import PeakFilter

sample_rate = 48000
seconds = np.arange(sample_rate, dtype=np.float32) / sample_rate
audio = (0.2 * np.sin(2 * np.pi * 220 * seconds)).astype(np.float32)

output = PeakFilter().process(
    audio,
    sample_rate,
    automation={
        "cutoff_frequency_hz": np.geomspace(200, 8000, audio.size).astype(np.float32),
        "gain_db": np.linspace(0, 6, audio.size, dtype=np.float32),
        "q": np.full(audio.size, 0.707, dtype=np.float32),
    },
)
```

For streaming, slice audio and curves together and pass `reset=False`. Supported
filters and dynamics retain processing memory when only the block size grows.
EQ curves also receive joint coefficient validation before parameters or DSP
history change. LadderFilter retains JUCE's original 50 ms cutoff/resonance
smoothing and immediate drive control; its six modes remain fixed scalar choices.
Reverb retains its original 10 ms smoothers and delay tails on control changes
and block-size growth; `freeze_mode` remains scalar.
Delay retains the original integer read-head updates and linear mix, without
new smoothing; fast delay-time changes can produce read-head artifacts.
Chorus keeps its original LFO phase and native smoothing; bounded scratch
processing preserves history when blocks grow.
Phaser retains its original four-frame frequency-update clock and 50 ms ramps;
the dense path also refreshes centre-frequency mapping after format preparation.
Convolution computes the wet signal in blocks, then applies the mix curve through
its original 50 ms dry/wet ramps. Block growth preserves its tail; differing FFT
partitions can produce small float32 differences.
BrickwallLimiter queues ceiling/release targets with its lookahead so they align
with compensated output frames. Streaming calls can initially return fewer frames;
explicit zero input drains pending audio, while `reset=True` drains automatically.
Scalar-only calls retain their original input-clock behavior.
PitchShift queues targets through its one-second priming delay, then applies them
when corresponding source frames enter Rubber Band. Windowed pitch response is
not an instantaneous sample-exact output transition. Short streaming calls retain
the priming audio; block growth keeps the first prepared capacity and histories.
Resample curves share a clock between downsampling and reconstruction, retaining
all eleven interpolation choices. Curve rates are in `[sample_rate/64, sample_rate]`;
the host rate stays fixed. The clocked path uses fixed compensated latency (4162
frames for default WindowedSinc32); a warm scalar-to-curve transition requires
`reset=True`. Once active, `reset=False` scalar calls hold the rate on that clock.
Fixed parameters should use the existing scalar path.

For example, with the mono audio and sample rate above:

```python
from pedalboard import Resample

rate_curve = np.linspace(sample_rate / 2, sample_rate / 12, audio.size, dtype=np.float32)
output = Resample()(audio, sample_rate, automation={"target_sample_rate": rate_curve})
```

See the [automation reference](docs/source/reference/pedalboard.rst) for accepted
values, layouts, reset behavior and effect-specific limits. This interface targets
supported built-ins, VST3/Audio Unit plugins and their `Pedalboard`/`Chain`/`Mix` graphs.
VST3 parameter curves use display units as described below.

### Automate an effect chain

Use effect instances as keys, so curves follow their effects when the graph is
reordered. Nested `Chain` and `Mix` nodes use the same mapping:

```python
from pedalboard import Pedalboard, Gain, LowpassFilter

gain, lowpass = Gain(), LowpassFilter()
board = Pedalboard([gain, lowpass])
output = board(audio, sample_rate, automation={
    gain: {"gain_db": np.linspace(-12, 0, audio.size, dtype=np.float32)},
    lowpass: {"cutoff_frequency_hz": np.geomspace(300, 8000, audio.size).astype(np.float32)},
})
```

Every curve uses the root input's frame count. Downstream nodes wait for valid
upstream audio; parallel branches retain their own pending audio and controls.
For offline rendering, the default `reset=True` compensates startup latency and
holds final targets while draining it. For streaming, call `board.reset()` first,
then pass `reset=False` on every audio/curve slice. Omitted active lanes hold their
last scheduled targets. Send explicit zero audio to drain delayed output; a zero-
length call does not flush it. Existing reverb/delay tails still require enough
explicit silence if you want to export the full tail.

Within a streaming session, keep the same node/parameter lanes, topology, sample
rate and channel count. To change them, restart with `reset=True`, or explicitly
reset the root and start a new `reset=False` stream. Set scalar defaults before
starting the session; use curves to change active lanes. Do not independently
render/reset participating child nodes or nested containers during a root session.
Other effects can remain in the graph with scalar settings. The 21 listed
built-ins and supported VST3/Audio Unit parameters can receive curves. Omit automation
on a graph without an active session for its original scalar processing path.

### VST3 automation in display units

VST3 curves use the same displayed values as scalar parameter attributes: dB,
Hz, percentages, numeric choices or booleans, rather than normalized 0–1 values.
Names come from `plugin.parameters`; not every plugin appends its units to a name.
For example, CHOWTapeModel names its gain `input_gain` and its Hz cutoff
`input_low_cut`:

```python
from pedalboard import VST3Plugin

vst = VST3Plugin("CHOWTapeModel.vst3")  # Supply your installed plugin's path.
output = vst(audio, sample_rate, automation={
    "input_gain": np.linspace(-12, 6, audio.size, dtype=np.float32),
    "input_low_cut": np.geomspace(100, 1000, audio.size).astype(np.float32),
})
```

The default `vst.automation_mode = "sample_steps"` processes synchronized constant
spans in C++. It also works with plugins that only read the last parameter value
in a block. Dense changing curves may require one native process call per frame;
Python is not called per frame. For plugins known to consume intra-block offsets,
set `vst.automation_mode = "vst3_events"` **before** starting the stream to send
sample-positioned points in larger blocks. This can be faster, but plugins that
ignore offsets will respond differently. A mode change requires a root/plugin reset.
Both modes retain the plugin's own smoothing and block-dependent behavior.

Use `automation={vst: {...}}` in a graph, including alongside builtin curves.
MIDI instruments accept the same keyword on their existing render entry point:
`vst(midi_messages, duration=seconds, sample_rate=rate, automation={...})`.
Instrument curves have one value per returned frame, on the same input clock as
MIDI. For standalone offline instruments, opt into fixed reported-latency
compensation with `compensate_latency=True` and `reset=True`; raw output remains
the default. See [instrument latency](docs/source/instrument_latency.rst).

Display-unit conversion validates and copies the entire curve against the current
controller model before DSP changes. It checks that model again after preparation
and reconverts targets when the plugin changes its range with the audio format.
It uses the plugin's SDK conversion model when it agrees with display units,
otherwise its display parser, with round-trip checks at display precision. There
is no range-guessing or approximate lookup-table fallback. Numeric displays and
recognized two-state booleans are supported; read-only/program parameters,
ambiguous names/units, text-only choices and unreliable conversions raise errors.
Hz/kHz displays are converted to Hz when no unit label is reported. Literal infinite
endpoints, such as a displayed `-inf` gain, map to their normalized endpoint.
Normalized processing precision remains float32, matching the existing scalar host.
If a target lies outside the load-time range but inside the desired format's range,
call `vst.prepare_automation(sample_rate, num_channels=2)` first, then render with
`reset=False`. Preparation resets processing state. Post-preparation conversion
errors stop rendering but do not undo preparation; correct the curves and restart.
An active graph stream requires stable unit mappings until the root is reset.

For streaming, explicitly reset the root/plugin first and use `reset=False` on
matching slices. Starting curves after warm scalar processing also requires a
reset. Do not change plugin state/presets or process participating children
independently during an active graph session. See below for AU automation and
explicit audio buses and caller-controlled transport. Warm-state cloning remains
a separate capability.

### Audio Unit automation on macOS

`AudioUnitPlugin` accepts the same display-unit `automation` mapping for audio and
MIDI instruments, including instance-keyed curves in `Pedalboard`/`Chain`/`Mix`:

```python
from pedalboard import AudioUnitPlugin

au = AudioUnitPlugin("/Library/Audio/Plug-Ins/Components/CHOWTapeModel.component")
output = au(audio, sample_rate, automation={
    "input_gain": np.linspace(-12, 6, audio.size, dtype=np.float32),
    "input_low_cut": np.geomspace(100, 1000, audio.size),
})
```

Audio Units currently use `automation_mode = "sample_steps"` only. Equal targets
are coalesced and changing spans are dispatched in C++; no per-frame Python calls.
AU native scheduled events are not exposed. Plugin smoothing and block-dependent
behavior remain. Global writable, real-time parameters with reliable numeric or
boolean conversions are supported; read-only/non-real-time parameters and text-only
choices are rejected. Conversions use native AU metadata or the native display
parser with round-trip checks; AU's physical float32 quantization is retained.
Parameters naming only special endpoints (such as `-inf`) also use their documented
metadata values when the actual display confirms that conversion.
This avoids the legacy Python scalar wrapper's inferred/rounded parameter ranges.
The audio/MIDI streaming, reset, stable-unit-model and `prepare_automation` rules
above also apply to AU. Actual AUv2 plugins and an original, ad-hoc-signed AUv3
extension are tested on macOS arm64. The AUv3 checks load the real `.appex`
out-of-process and cover dB/Hz/boolean curves, MIDI, delay, graph streams and
reset/reinstantiation. A 64-bit AU parameter address is exercised alongside a
separate parameter with the same low32 bits through Apple's AU bridge.

An AUv3's containing application must be registered with macOS before its `.appex`
can be discovered. Relative paths, encoded spaces/Unicode, symlinks and multiple
components in one extension are supported by the scanner. The AUv3 test fixture
temporarily registers only its generated build and removes those entries afterward.
Manual macOS arm64 acceptance also covers an unchanged source build of
[Morass v1.0.3](https://github.com/bogware/morass/tree/v1.0.3): dB/Hz/boolean
curves, scalar-reference audio, streaming and parameter-state checks. Its published
macOS installer contains AUv2/VST3, so this does not verify a vendor-distributed
AUv3 binary. Morass AUv3 and CHOWTapeModel AUv2 editors were visually checked for
parameter-to-UI updates, automation while open and reopening after reset.
A human-operated Morass AUv3 check also verifies that editing Input Gain in its GUI
updates the native host readback and affects rendered audio, followed by another
automation curve without reset. Direct GUI-to-host/audio roundtrip for CHOW AUv2,
other AUv3 vendors and other architectures is not covered; this is not universal
plugin compatibility.

## Explicit main, sidechain and auxiliary buses

```python
print(vst.audio_buses)  # Inspect each direction's indices, names and layouts.
outputs = vst.process_buses(
    {0: main_audio, 1: detector_audio}, sample_rate,
    output_buses={0: 2},
)
processed_main = outputs[0]

# Optional MIDI and physical-unit curves share the source-audio timeline.
outputs, midi_out = vst.process_buses(
    {0: main_audio, 1: detector_audio}, sample_rate,
    output_buses={0: 2},
    midi_messages=[(bytes([0x90, 60, 100]), 31 / sample_rate)],
    automation={"gain_db": np.linspace(-12, 0, main_audio.shape[1])},
    return_midi=True,
)
```

Each input is an explicit planar `(channels, frames)` float32/float64 array.
Select independent output buses by index and channel count; each returned array
owns its memory. This API returns raw plugin latency and preserves continuous
processing with `reset=False`. Reset before changing native layout, sample rate
or switching processing APIs. VST3 omitted buses are requested disabled; the
inherited AU host keeps omitted inputs active with silence and discards omitted
outputs. Native layout negotiation may reject unsupported combinations after
reset. See the [hosting contract](docs/source/hosting.rst) for state boundaries,
supported arrangements, timing and verification scope.

### Caller-controlled plugin transport

`process_buses(..., transport={"bpm": 90.0, "sample_position": 0,
"ppq_position": 0.0, "playing": True, "time_signature": (4, 4)})` supplies
an explicit project clock to VST3 and Audio Unit plugins alongside buses, MIDI and
parameter curves. Each call supplies its own start anchor; `transport=None`
preserves the original default clock. Optional recording and quarter-note loop
bounds are supported. Looping wraps the plugin clock, while the caller supplies
source audio and notes. See [hosted routing and transport](docs/source/hosting.rst)
for stopped-clock, fractional-loop and verification boundaries.

## Save and restore configurations

```python
from pedalboard import configuration

configuration.save(board, "chain.json", sample_rate=48000, num_channels=2)
restored = configuration.load("chain.json")  # Validates settings and plugin state.
report = configuration.verify(board, restored, audio, sample_rate=48000)
```

This saves supported builtin settings, Chain/Mix topology and opaque VST3/AU state.
Saving leaves the source DSP history alone; `verify` explicitly resets and renders
both objects to compare sound. Running tails and active notes are not guaranteed to
survive configuration loading. Convolution is currently rejected because its IR
cannot be captured through the existing binding. External plugin binaries/assets
are not bundled; loading requires the same native runtime and unchanged plugins.
See the [configuration contract](docs/source/configuration.rst) for format limits,
main-thread requirements, parameter quantization and verification tolerances.

## Install the OpenMIRLab source

The automation extensions are a development snapshot, not a separate PyPI release.
Installing `pedalboard` from PyPI installs the upstream distribution.

A C++ toolchain is required. With [uv](https://docs.astral.sh/uv/) installed:

```shell
git clone --branch main --recurse-submodules https://github.com/openmirlab/pedalboard.git
cd pedalboard
uv venv --python 3.12
uv pip install .
```

The Python package and import name remain `pedalboard`. The automation extensions
have been built and tested on macOS arm64 with Python 3.12; their Windows/Linux
behavior has not yet been independently validated. The upstream platform support
and distribution information below describes Spotify's project.

## Contributing and license

Use [CONTRIBUTING.md](CONTRIBUTING.md) for this fork's development workflow.
The fork retains the [GNU GPL v3 license](LICENSE) and upstream attribution.

---

## Upstream Pedalboard

The original Spotify README follows for the upstream project and distribution.

![Pedalboard Logo](https://user-images.githubusercontent.com/213293/131147303-4805181a-c7d5-4afe-afb2-f591a4b8e586.png)


[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/spotify/pedalboard/blob/master/LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-on%20github.io-brightgreen)](https://spotify.github.io/pedalboard)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pedalboard)](https://pypi.org/project/pedalboard)
[![Supported Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Windows%20%7C%20Linux-green)](https://pypi.org/project/pedalboard)
[![Apple Silicon support for macOS and Linux (Docker)](https://img.shields.io/badge/Apple%20Silicon-macOS%20and%20Linux-brightgreen)](https://pypi.org/project/pedalboard)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/pedalboard)](https://pypi.org/project/pedalboard)
[![Test Badge](https://github.com/spotify/pedalboard/actions/workflows/all.yml/badge.svg)](https://github.com/spotify/pedalboard/actions/workflows/all.yml)
[![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/psobot/8736467e9952991ef44a67915ee7c762/raw/coverage.json)](https://gist.githubusercontent.com/psobot/8736467e9952991ef44a67915ee7c762/raw/coverage.json)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pedalboard)](https://pypistats.org/packages/pedalboard)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7817838.svg)](https://doi.org/10.5281/zenodo.7817838)
[![GitHub Repo stars](https://img.shields.io/github/stars/spotify/pedalboard?style=social)](https://github.com/spotify/pedalboard/stargazers)


`pedalboard` is a Python library for working with audio: reading, writing, rendering, adding effects, and more. It supports most popular audio file formats and a number of common audio effects out of the box, and also allows the use of [VST3®](https://www.steinberg.net/en/company/technologies/vst3.html) and [Audio Unit](https://en.wikipedia.org/wiki/Audio_Units) formats for loading third-party software instruments and effects.

`pedalboard` was built by [Spotify's Audio Intelligence Lab](https://research.atspotify.com/audio-visual-intelligence/) to enable using studio-quality audio effects from within Python and TensorFlow. Internally at Spotify, `pedalboard` is used for [data augmentation](https://en.wikipedia.org/wiki/Data_augmentation) to improve machine learning models and to help power features like [Spotify's AI DJ](https://newsroom.spotify.com/2023-02-22/spotify-debuts-a-new-ai-dj-right-in-your-pocket/) and [AI Voice Translation](https://newsroom.spotify.com/2023-09-25/ai-voice-translation-pilot-lex-fridman-dax-shepard-steven-bartlett/). `pedalboard` also helps in the process of content creation, making it possible to add effects to audio without using a Digital Audio Workstation.

[![Documentation](https://img.shields.io/badge/Documentation-on%20github.io-brightgreen)](https://spotify.github.io/pedalboard)

## Features 

 - Built-in audio I/O utilities ([pedalboard.io](https://spotify.github.io/pedalboard/reference/pedalboard.io.html))
   - Support for reading and writing AIFF, FLAC, MP3, OGG, and WAV files on all platforms with no dependencies
   - Additional support for reading AAC, AC3, WMA, and other formats depending on platform
   - Support for on-the-fly resampling of audio files and streams with `O(1)` memory usage
   - Live audio effects via <a href="https://spotify.github.io/pedalboard/reference/pedalboard.io.html#pedalboard.io.AudioStream"><code class="docutils literal"><span class="pre">AudioStream</span></code></a>
 - Built-in support for a number of basic audio transformations, including:
   - Guitar-style effects: `Chorus`, `Distortion`, `Phaser`, `Clipping`
   - Loudness and dynamic range effects: `Compressor`, `Gain`, `Limiter`
   - Equalizers and filters: `HighpassFilter`, `LadderFilter`, `LowpassFilter`
   - Spatial effects: `Convolution`, `Delay`, `Reverb`
   - Pitch effects: `PitchShift`
   - Lossy compression: `GSMFullRateCompressor`, `MP3Compressor`
   - Quality reduction: `Resample`, `Bitcrush`
 - Supports VST3® instrument and effect plugins on macOS, Windows, and Linux (<a href="https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.load_plugin"><code class="docutils literal"><span class="pre">pedalboard.load_plugin</span></code></a>)
 - Supports instrument and effect Audio Units on macOS
 - Strong thread-safety, memory usage, and speed guarantees
   - Releases Python's Global Interpreter Lock (GIL) to allow use of multiple CPU cores
     - No need to use `multiprocessing`!
   - Even when only using one thread:
     - Processes audio up to **300x** faster than [pySoX](https://github.com/rabitt/pysox) for single transforms, and 2-5x faster than [SoxBindings](https://github.com/pseeth/soxbindings) (via [iCorv](https://github.com/iCorv/pedalboard_with_tfdata))
     - Reads audio files up to **4x** faster than [librosa.load](https://librosa.org/doc/latest/api/generated/librosa.load.html) (in many cases)
 - Tested compatibility with TensorFlow - can be used in `tf.data` pipelines!

## Installation

`pedalboard` is available via PyPI (via [Platform Wheels](https://packaging.python.org/guides/distributing-packages-using-setuptools/#platform-wheels)):
```
pip install pedalboard
```

If you are new to Python, follow [INSTALLATION.md](https://github.com/spotify/pedalboard/blob/master/INSTALLATION.md) for a robust guide.

### Compatibility

`pedalboard` is thoroughly tested with Python 3.10, 3.11, 3.12, 3.13, 3.14, and 3.15.

- Linux
  - Tested heavily in production use cases at Spotify
  - Tested automatically on GitHub with VSTs
  - Platform `manylinux` and `musllinux` wheels built for `x86_64` (Intel/AMD) and `aarch64` (ARM/Apple Silicon)
  - `x86_64` wheels require AVX support
  - Most Linux VSTs require a relatively modern Linux installation (with glibc > 2.27)
- macOS
  - Tested manually with VSTs and Audio Units
  - Tested automatically on GitHub with VSTs
  - Platform wheels available for both Intel and Apple Silicon
  - Compatible with a wide range of VSTs and Audio Units
- Windows
  - Tested automatically on GitHub with VSTs
  - Platform wheels available for `amd64` (x86-64, Intel/AMD)

## Examples

> **Note**: If you'd rather watch a video instead of reading examples or documentation, <strong>watch <a href="https://www.youtube.com/watch?v=NYhkqXpFAlg" target="_blank"><em>Working with Audio in Python (feat. Pedalboard)</em> on YouTube</a></strong>.

### Quick start

```python
from pedalboard import Pedalboard, Chorus, Reverb
from pedalboard.io import AudioFile

# Make a Pedalboard object, containing multiple audio plugins:
board = Pedalboard([Chorus(), Reverb(room_size=0.25)])

# Open an audio file for reading, just like a regular file:
with AudioFile('some-file.wav') as f:
  
  # Open an audio file to write to:
  with AudioFile('output.wav', 'w', f.samplerate, f.num_channels) as o:
  
    # Read one second of audio at a time, until the file is empty:
    while f.tell() < f.frames:
      chunk = f.read(f.samplerate)
      
      # Run the audio through our pedalboard:
      effected = board(chunk, f.samplerate, reset=False)
      
      # Write the output to our output file:
      o.write(effected)
```

> **Note**: For more information about how to process audio through
> Pedalboard plugins, including how the `reset` parameter works,
> see <a href="https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.Plugin.process">
> the documentation for <code class="docutils literal"><span class="pre">pedalboard.Plugin.process</span></code></a>.

### Making a guitar-style pedalboard

```python
# Don't do import *! (It just makes this example smaller)
from pedalboard import *
from pedalboard.io import AudioFile

# Read in a whole file, resampling to our desired sample rate:
samplerate = 44100.0
with AudioFile('guitar-input.wav').resampled_to(samplerate) as f:
  audio = f.read(f.frames)

# Make a pretty interesting sounding guitar pedalboard:
board = Pedalboard([
    Compressor(threshold_db=-50, ratio=25),
    Gain(gain_db=30),
    Chorus(),
    LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=900),
    Phaser(),
    Convolution("./guitar_amp.wav", 1.0),
    Reverb(room_size=0.25),
])

# Pedalboard objects behave like lists, so you can add plugins:
board.append(Compressor(threshold_db=-25, ratio=10))
board.append(Gain(gain_db=10))
board.append(Limiter())

# ... or change parameters easily:
board[0].threshold_db = -40

# Run the audio through this pedalboard!
effected = board(audio, samplerate)

# Write the audio back as a wav file:
with AudioFile('processed-output.wav', 'w', samplerate, effected.shape[0]) as f:
  f.write(effected)
```

### Using VST3® or Audio Unit instrument and effect plugins

```python
from pedalboard import Pedalboard, Reverb, load_plugin
from pedalboard.io import AudioFile
from mido import Message # not part of Pedalboard, but convenient!

# Load a VST3 or Audio Unit plugin from a known path on disk:
instrument = load_plugin("./VSTs/Magical8BitPlug2.vst3")
effect = load_plugin("./VSTs/RoughRider3.vst3")

print(effect.parameters.keys())
# dict_keys([
#   'sc_hpf_hz', 'input_lvl_db', 'sensitivity_db',
#   'ratio', 'attack_ms', 'release_ms', 'makeup_db',
#   'mix', 'output_lvl_db', 'sc_active',
#   'full_bandwidth', 'bypass', 'program',
# ])

# Set the "ratio" parameter to 15
effect.ratio = 15

# Render some audio by passing MIDI to an instrument:
sample_rate = 44100
audio = instrument(
  [Message("note_on", note=60), Message("note_off", note=60, time=5)],
  duration=5, # seconds
  sample_rate=sample_rate,
)

# Apply effects to this audio:
effected = effect(audio, sample_rate)

# ...or put the effect into a chain with other plugins:
board = Pedalboard([effect, Reverb()])
# ...and run that pedalboard with the same VST instance!
effected = board(audio, sample_rate)
```

### Creating parallel effects chains

This example creates a delayed pitch-shift effect by running
multiple Pedalboards in parallel on the same audio. `Pedalboard`
objects are themselves `Plugin` objects, so you can nest them
as much as you like:

```python
from pedalboard import Pedalboard, Compressor, Delay, Distortion, Gain, PitchShift, Reverb, Mix

passthrough = Gain(gain_db=0)

delay_and_pitch_shift = Pedalboard([
  Delay(delay_seconds=0.25, mix=1.0),
  PitchShift(semitones=7),
  Gain(gain_db=-3),
])

delay_longer_and_more_pitch_shift = Pedalboard([
  Delay(delay_seconds=0.5, mix=1.0),
  PitchShift(semitones=12),
  Gain(gain_db=-6),
])

board = Pedalboard([
  # Put a compressor at the front of the chain:
  Compressor(),
  # Run all of these pedalboards simultaneously with the Mix plugin:
  Mix([
    passthrough,
    delay_and_pitch_shift,
    delay_longer_and_more_pitch_shift,
  ]),
  # Add a reverb on the final mix:
  Reverb()
])
```

### Running Pedalboard on Live Audio

`pedalboard` supports streaming live audio through
<a href="https://spotify.github.io/pedalboard/reference/pedalboard.io.html#pedalboard.io.AudioStream">
an <code class="docutils literal"><span class="pre">AudioStream</span></code> object</a>,
allowing for real-time manipulation of audio by adding effects in Python.

```python
from pedalboard import Pedalboard, Chorus, Convolution, Compressor, Delay, Gain, Reverb, Phaser
from pedalboard.io import AudioStream

# Open up an audio stream:
with AudioStream(
  input_device_name="Apogee Jam+",  # Guitar interface
  output_device_name="MacBook Pro Speakers"
) as stream:
  # Audio is now streaming through this pedalboard and out of your speakers!
  stream.plugins = Pedalboard([
      Compressor(threshold_db=-50, ratio=25),
      Gain(gain_db=30),
      Chorus(),
      Phaser(),
      Convolution("./guitar_amp.wav", 1.0),
      Reverb(room_size=0.25),
  ])
  input("Press enter to stop streaming...")

# The live AudioStream is now closed, and audio has stopped.
```

### Using Pedalboard in `tf.data` Pipelines

```python
import tensorflow as tf 

sr = 48000 

# Put whatever plugins you like in here:
plugins = pedalboard.Pedalboard([pedalboard.Gain(), pedalboard.Reverb()]) 

# Make a dataset containing random noise:
# NOTE: for real training, here's where you'd want to load your audio somehow:
ds = tf.data.Dataset.from_tensor_slices([np.random.rand(sr)])

# Apply our Pedalboard instance to the tf.data Pipeline:
ds = ds.map(lambda audio: tf.numpy_function(plugins.process, [audio, sr], tf.float32)) 

# Create and train a (dummy) ML model on this audio:
model = tf.keras.models.Sequential([tf.keras.layers.InputLayer(input_shape=(sr,)), tf.keras.layers.Dense(1)])
model.compile(loss="mse") 
model.fit(ds.map(lambda effected: (effected, 1)).batch(1), epochs=10)
```

For more examples, see:
 - [the "examples" folder of this repository](https://github.com/spotify/pedalboard/tree/master/examples)
 - [the "Pedalboard Demo" Colab notebook](https://colab.research.google.com/drive/1bHjhJj1aCoOlXKl_lOfG99Xs3qWVrhch)
 - [_Working with Audio in Python (feat. Pedalboard)_ by Peter Sobot at EuroPython 2022](https://www.youtube.com/watch?v=NYhkqXpFAlg)
 - [an interactive web demo on Hugging Face Spaces and Gradio](https://huggingface.co/spaces/akhaliq/pedalboard) (via [@AK391](https://github.com/AK391)) 

## Contributing

Contributions to `pedalboard` are welcomed! See [CONTRIBUTING.md](https://github.com/spotify/pedalboard/blob/master/CONTRIBUTING.md) for details.

## Citing

To cite `pedalboard` in academic work, use [its entry on Zenodo](https://doi.org/10.5281/zenodo.7817838): [![DOI 7817838](https://zenodo.org/badge/DOI/10.5281/zenodo.7817838.svg)](https://doi.org/10.5281/zenodo.7817838)

To cite via BibTeX:

```tex
@software{sobot_peter_2023_7817838,
  author       = {Sobot, Peter},
  title        = {Pedalboard},
  month        = jul,
  year         = 2021,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.7817838},
  url          = {https://doi.org/10.5281/zenodo.7817838}
}
```

## License
`pedalboard` is Copyright 2021-2026 Spotify AB.

`pedalboard` is licensed under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html). `pedalboard` includes a number of libraries that are statically compiled, and which carry the following licenses:

 - The core audio processing code is pulled from [JUCE 6](https://juce.com/), which is [dual-licensed under a commercial license and the GPLv3](https://juce.com/juce-6-licence).
 - The [VST3 SDK](https://github.com/steinbergmedia/vst3sdk), bundled with JUCE, is owned by [Steinberg® Media Technologies GmbH](https://www.steinberg.net/en/home.html) and licensed under the GPLv3.
 - The `PitchShift` plugin and `time_stretch` functions use [the Rubber Band Library](https://github.com/breakfastquay/rubberband), which is [dual-licensed under a commercial license](https://breakfastquay.com/technology/license.html) and the GPLv2 (or newer). [FFTW](https://www.fftw.org/) is also included to speed up Rubber Band, and [is licensed under the GPLv2 (or newer)](https://www.fftw.org/doc/License-and-Copyright.html).
 - The `MP3Compressor` plugin uses [libmp3lame from the LAME project](https://lame.sourceforge.io/), which is [licensed under the LGPLv2](https://github.com/lameproject/lame/blob/master/README) and [upgraded to the GPLv3 for inclusion in this project (as permitted by the LGPLv2)](https://www.gnu.org/licenses/gpl-faq.html#AllCompatibility).
 - The `GSMFullRateCompressor` plugin uses [libgsm](http://quut.com/gsm/), which is [licensed under the ISC license](https://github.com/timothytylee/libgsm/blob/master/COPYRIGHT) and [compatible with the GPLv3](https://www.gnu.org/licenses/license-list.en.html#ISC).
 - WAV files with formats not natively supported by JUCE (ADPCM, A-law, µ-law, 64-bit float) are decoded using [dr_wav](https://github.com/mackron/dr_libs) by David Reid, which is released into the [public domain](https://unlicense.org/).

_VST is a registered trademark of Steinberg Media Technologies GmbH._
