/* Build wrapper only; the JUCE NoiseGate algorithm is included unchanged from
 * the pinned submodule. Its source retains the ISC license. GPL-3.0-only. */
#include <juce_audio_utils/juce_audio_utils.h>
using namespace juce;
#include "../../JUCE/examples/Plugins/NoiseGatePluginDemo.h"

juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter() {
  return new NoiseGate;
}
