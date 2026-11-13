"""Static contract for explicit hosted audio routing."""

import numpy as np

from pedalboard import VST3Plugin

plugin = VST3Plugin("compressor.vst3")
ports = plugin.audio_buses
main = np.zeros((2, 1000), dtype=np.float32)
detector = np.ones((1, 1000), dtype=np.float64)
result = plugin.process_buses({0: main, 1: detector}, 48000, output_buses={0: 2})
plugin.process_buses({}, 48000, output_buses={0: 2}, num_frames=1000)
