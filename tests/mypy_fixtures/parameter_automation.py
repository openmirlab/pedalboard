"""The four supported effects expose native dense automation in their stubs."""
import numpy as np

from pedalboard import Bitcrush, Clipping, Distortion, Gain

audio = np.ones((2, 128), dtype=np.float32)
curve = np.zeros(128, dtype=np.float64)
Gain().process(audio, 48000, automation={"gain_db": curve})
Clipping()(audio, 48000, automation={"threshold_db": curve})
Distortion().process(audio, 48000, reset=False, automation={"drive_db": curve})
Bitcrush()(audio, 48000, automation={"bit_depth": np.full(128, 8, dtype=np.int32)})
Gain()(audio, 48000)
