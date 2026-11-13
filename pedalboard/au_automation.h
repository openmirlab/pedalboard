/* Audio Unit backend for shared display-unit automation. GPL-3.0-only. */
#pragma once
#include "au_automation_bridge.h"
#include "external_plugin_automation.h"
namespace Pedalboard {
inline void init_au_automation(py::module &module) {
#if JUCE_MAC && JUCE_PLUGINHOST_AU
  initHostedAutomation<juce::AudioUnitPluginFormat, AudioUnitAutomationBackend>(
      module, "AudioUnitPlugin");
#endif
}
} // namespace Pedalboard
