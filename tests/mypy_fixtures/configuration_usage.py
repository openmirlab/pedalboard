from pathlib import Path

import numpy as np

from pedalboard import Gain, configuration

plugin = Gain(gain_db=-3)
configuration.save(plugin, Path("configuration.json"), sample_rate=48000)
restored = configuration.load("configuration.json")
result: configuration.VerificationResult = configuration.verify(
    plugin, restored, np.zeros((2, 1000), dtype=np.float32), sample_rate=48000
)
maximum: float = result.max_abs_error
