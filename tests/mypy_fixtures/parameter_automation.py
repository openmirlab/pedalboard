"""Supported effects expose native dense automation in their stubs."""

import numpy as np

from pedalboard import (
    Bitcrush,
    Clipping,
    Compressor,
    Distortion,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    LowpassFilter,
    LowShelfFilter,
    NoiseGate,
    PeakFilter,
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
