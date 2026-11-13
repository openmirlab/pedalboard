/* Build wrapper only; unmodified pinned JUCE AudioPluginDemo (ISC source).
 * GPL-3.0-only wrapper. */
#include <juce_audio_utils/juce_audio_utils.h>
using namespace juce;
#include "../../JUCE/examples/Plugins/AudioPluginDemo.h"

juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter() {
  return new JuceDemoPluginAudioProcessor;
}
