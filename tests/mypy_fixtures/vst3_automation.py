"""VST3 display-unit audio, graph and MIDI automation are typed."""

import numpy as np

from pedalboard import Gain, Pedalboard, VST3Plugin

vst = VST3Plugin("filter.vst3")
vst.automation_mode = "sample_steps"
vst.prepare_automation(48000, num_channels=2)
curve = np.zeros(128, np.float32)
audio = np.ones((2, 128), np.float32)
vst.process(audio, 48000, automation={"gain_db": curve})
vst(audio, 48000, automation={"gain_db": curve})
vst.reset()
vst.automation_mode = "vst3_events"
vst.process(
    [([144, 60, 127], 0.0)],
    duration=0.1,
    sample_rate=48000,
    automation={"cutoff_hz": np.ones(4800)},
)
vst([], duration=0.1, sample_rate=48000, automation={"cutoff_hz": np.ones(4800)})
gain = Gain()
Pedalboard([vst, gain])(
    audio, 48000, automation={vst: {"gain_db": curve}, gain: {"gain_db": curve}}
)
