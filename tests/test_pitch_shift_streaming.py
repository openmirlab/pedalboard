"""Repeated preparation must not erase the one-second priming delay."""

import numpy as np
import pytest

from pedalboard import PitchShift


@pytest.mark.parametrize("size", [128, 513, 8192])
def test_unshifted_short_stream_keeps_audio(size):
    rate = 22050
    x = (0.2 * np.sin(2 * np.pi * 440 * np.arange(2 * rate) / rate)).astype(np.float32)
    effect = PitchShift(0)
    parts = []
    for start in range(0, len(x), size):
        parts.append(effect(x[start : start + size], rate, buffer_size=size, reset=False))
    # Drain the priming delay and Rubber Band's startup deficit.
    for _ in range((2 * rate) // size + 2):
        parts.append(effect(np.zeros(size, np.float32), rate, buffer_size=size, reset=False))
    actual = np.concatenate(parts)[: len(x)]
    np.testing.assert_allclose(actual, x, atol=1e-6)
