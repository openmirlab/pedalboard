"""Type-check the explicit offline instrument option for both native formats."""

from pedalboard import AudioUnitPlugin, VST3Plugin
from pedalboard_native import MIDIMessageLike

midi: list[MIDIMessageLike] = [(bytes([0x90, 60, 100]), 0.0)]
vst = VST3Plugin("instrument.vst3")
vst(midi, duration=1, sample_rate=48000, compensate_latency=True)
vst.process(midi, duration=1, sample_rate=48000, reset=False, compensate_latency=False)
au = AudioUnitPlugin("instrument.component")
au(midi, duration=1, sample_rate=48000, compensate_latency=True)
