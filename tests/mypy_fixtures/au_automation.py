"""AU display-unit audio, graph and MIDI automation signatures."""

import numpy as np

from pedalboard import AudioUnitPlugin, Gain, Pedalboard

au = AudioUnitPlugin("effect.component")
au.automation_mode = "sample_steps"
au.prepare_automation(48000, num_channels=2)
audio = np.ones((2, 128), np.float32)
curve = np.zeros(128, np.float32)
au(audio, 48000, automation={"gain_db": curve})
au.process(audio, 48000, automation={"gain_db": curve})
au([], duration=0.1, sample_rate=48000, automation={"gain_db": np.zeros(4800)})
au.process(
    [([144, 60, 127], 0.0)], duration=0.1, sample_rate=48000, automation={"gain_db": np.zeros(4800)}
)
gain = Gain()
Pedalboard([gain, au])(audio, 48000, automation={au: {"gain_db": curve}, gain: {"gain_db": curve}})
