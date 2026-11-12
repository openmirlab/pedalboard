"""Supported effects expose native dense automation in their stubs."""

import numpy as np

from pedalboard import (
    Bitcrush,
    BrickwallLimiter,
    Chain,
    Chorus,
    Clipping,
    Compressor,
    Convolution,
    Delay,
    Distortion,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    LadderFilter,
    Limiter,
    LowpassFilter,
    LowShelfFilter,
    Mix,
    NoiseGate,
    PeakFilter,
    Pedalboard,
    Phaser,
    PitchShift,
    Resample,
    Reverb,
)

audio = np.ones((2, 128), dtype=np.float32)
curve = np.zeros(128, dtype=np.float64)
Gain().process(audio, 48000, automation={"gain_db": curve})
Clipping()(audio, 48000, automation={"threshold_db": curve})
Distortion().process(audio, 48000, reset=False, automation={"drive_db": curve})
Bitcrush()(audio, 48000, automation={"bit_depth": np.full(128, 8, dtype=np.int32)})
Gain()(audio, 48000)

Compressor().process(audio, 48000, automation={"threshold_db": curve, "ratio": np.ones(128)})
NoiseGate()(audio, 48000, reset=False, automation={"attack_ms": np.zeros(128)})
Limiter()(audio, 48000, automation={"threshold_db": curve, "release_ms": np.full(128, 100)})

LowpassFilter().process(audio, 48000, automation={"cutoff_frequency_hz": np.full(128, 800)})
HighpassFilter()(audio, 48000, automation={"cutoff_frequency_hz": np.full(128, 200.0)})

PeakFilter()(
    audio,
    48000,
    automation={"cutoff_frequency_hz": np.full(128, 1000), "gain_db": curve, "q": np.ones(128)},
)
LowShelfFilter().process(audio, 48000, reset=False, automation={"gain_db": curve})
HighShelfFilter()(audio, 48000, automation={"q": np.ones(128)})

LadderFilter(mode=LadderFilter.LPF24).process(
    audio,
    48000,
    automation={
        "cutoff_hz": np.full(128, 1000),
        "resonance": np.full(128, 0.5),
        "drive": np.ones(128),
    },
)

Reverb().process(audio, 48000, automation={"room_size": np.full(128, 0.8), "width": curve})
Reverb()(
    audio, 48000, reset=False, automation={"damping": curve, "wet_level": curve, "dry_level": curve}
)

Delay().process(audio, 48000, automation={"delay_seconds": np.full(128, 0.125), "feedback": curve})
Delay()(audio, 48000, reset=False, automation={"mix": curve})

Chorus().process(
    audio,
    48000,
    automation={"rate_hz": np.ones(128), "depth": curve, "centre_delay_ms": np.full(128, 8)},
)
Chorus()(audio, 48000, reset=False, automation={"feedback": curve, "mix": curve})

Phaser().process(
    audio,
    48000,
    automation={"rate_hz": np.ones(128), "depth": curve, "centre_frequency_hz": np.full(128, 1300)},
)
Phaser()(audio, 48000, reset=False, automation={"feedback": curve, "mix": curve})

Convolution(np.ones(16, dtype=np.float32), sample_rate=48000).process(
    audio, 48000, automation={"mix": curve}
)
Convolution(np.ones(16, dtype=np.float32), sample_rate=48000)(
    audio, 48000, reset=False, automation={"mix": curve}
)

BrickwallLimiter().process(audio, 48000, automation={"ceiling_db": curve})
BrickwallLimiter()(audio, 48000, reset=False, automation={"release_ms": np.ones(128)})

PitchShift().process(audio, 48000, automation={"semitones": curve})
PitchShift()(audio, 48000, reset=False, automation={"semitones": curve})

Resample().process(audio, 48000, automation={"target_sample_rate": np.full(128, 12000)})
Resample()(audio, 48000, reset=False, automation={"target_sample_rate": np.full(128, 4000)})


gain, lowpass = Gain(), LowpassFilter()
board = Pedalboard([gain, Mix([Chain([lowpass]), Gain()])])
board(
    audio,
    48000,
    automation={gain: {"gain_db": curve}, lowpass: {"cutoff_frequency_hz": np.full(128, 800)}},
)
Chain([gain]).process(audio, 48000, automation={gain: {"gain_db": curve}})
Mix([gain])(audio, 48000, automation={gain: {"gain_db": curve}})
