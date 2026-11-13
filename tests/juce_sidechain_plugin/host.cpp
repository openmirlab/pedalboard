/* Native JUCE host for the unmodified JUCE NoiseGate example. GPL-3.0-only. */
#include <fstream>
#include <iostream>
#include <juce_audio_utils/juce_audio_utils.h>

int main(int argc, char **argv) {
  juce::ScopedJuceInitialiser_GUI init;
  if (argc != 6 && argc != 7)
    return 2;
  const bool combined = argc == 7;
  juce::AudioPluginFormatManager manager;
  manager.addDefaultFormats();
  const juce::String path(argv[1]);
  juce::OwnedArray<juce::PluginDescription> descriptions;
  for (int i = 0; i < manager.getNumFormats(); ++i) {
    auto *format = manager.getFormat(i);
    if (format->fileMightContainThisPluginType(path))
      format->findAllTypesForFile(descriptions, path);
  }
  if (descriptions.isEmpty())
    return 3;
  const double rate = std::stod(argv[3]);
  const int size = std::stoi(argv[4]),
            frames = combined ? 2048 : (int)rate + 160;
  const bool keyEnabled = std::stoi(argv[5]);
  juce::String error;
  auto plugin =
      manager.createPluginInstance(*descriptions[0], rate, 8192, error);
  if (!plugin) {
    std::cerr << error << '\n';
    return 4;
  }
  plugin->enableAllBuses();
  auto layout = plugin->getBusesLayout();
  if (layout.inputBuses.size() != (combined ? 1 : 2) ||
      layout.outputBuses.size() != 1)
    return 5;
  layout.inputBuses.set(0, juce::AudioChannelSet::stereo());
  if (!combined)
    layout.inputBuses.set(1, juce::AudioChannelSet::stereo());
  layout.outputBuses.set(0, juce::AudioChannelSet::stereo());
  if (!plugin->setBusesLayout(layout))
    return 6;
  // Immediate detector permits exact onset and the example's one-second hold.
  plugin->getParameters()[0]->setValueNotifyingHost(combined ? 1.f : .5f);
  plugin->getParameters()[1]->setValueNotifyingHost(0);
  plugin->setNonRealtime(true);
  plugin->setRateAndBufferSizeDetails(rate, 8192);
  plugin->prepareToPlay(rate, 8192);
  std::vector<float> output(2 * frames);
  juce::AudioBuffer<float> scratch(4, std::min(size, 8192));
  for (int start = 0; start < frames;) {
    const int count = std::min({size, 8192, frames - start});
    scratch.clear();
    juce::AudioBuffer<float> block(scratch.getArrayOfWritePointers(), 4, count);
    for (int i = 0; i < count; ++i) {
      block.setSample(0, i, .2f);
      block.setSample(1, i, -.1f);
      if (!combined && keyEnabled && start + i == 31) {
        block.setSample(2, i, 1.f);
        block.setSample(3, i, 1.f);
      }
    }
    juce::MidiBuffer midi;
    if (combined && keyEnabled) {
      if (7 >= start && 7 < start + count)
        midi.addEvent(juce::MidiMessage::noteOn(1, 60, (juce::uint8)100),
                      7 - start);
      if (511 >= start && 511 < start + count)
        midi.addEvent(juce::MidiMessage::noteOff(1, 60), 511 - start);
    }
    plugin->processBlock(block, midi);
    for (int c = 0; c < 2; ++c)
      std::copy_n(block.getReadPointer(c), count,
                  output.data() + c * frames + start);
    start += count;
  }
  std::ofstream file(argv[2], std::ios::binary);
  file.write(reinterpret_cast<const char *>(output.data()),
             output.size() * sizeof(float));
  plugin->releaseResources();
  return file.good() ? 0 : 7;
}
