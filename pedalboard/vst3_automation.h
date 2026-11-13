/* VST3 backend for shared display-unit automation. GPL-3.0-only. */
#pragma once
#include "external_plugin_automation.h"

namespace Pedalboard {
struct VST3AutomationBackend {
  using Parameter = juce::PatchedVST3AutomationParameter;
  static double precision(double) { return 0; }
  static double neighbor(juce::AudioPluginInstance &, const Parameter &,
                         double raw, int) {
    return raw;
  }
  static Parameter describe(juce::AudioPluginInstance &p, int index) {
    return juce::describePatchedVST3Parameter(p, index);
  }
  static Parameter describeById(juce::AudioPluginInstance &p, uint32_t id) {
    return juce::describePatchedVST3ParameterById(p, id);
  }
  static double plain(juce::AudioPluginInstance &p, const Parameter &info,
                      double v) {
    return juce::getPatchedVST3ParameterPlain(p, info.id, v);
  }
  static double normalized(juce::AudioPluginInstance &p, const Parameter &info,
                           double v) {
    return juce::getPatchedVST3ParameterNormalized(p, info.id, v);
  }
  static juce::String text(juce::AudioPluginInstance &p, const Parameter &info,
                           double v) {
    return juce::getPatchedVST3ParameterText(p, info.id, v);
  }
  static double fromText(juce::AudioPluginInstance &p, const Parameter &info,
                         const juce::String &v) {
    return juce::getPatchedVST3ParameterFromText(p, info.id, v);
  }
  static void set(juce::AudioPluginInstance &p, uint32_t id, float v) {
    juce::setPatchedVST3Parameter(p, id, v);
  }
};
inline void init_vst3_automation(py::module &module) {
  initHostedAutomation<juce::PatchedVST3PluginFormat, VST3AutomationBackend>(
      module, "VST3Plugin");
}
} // namespace Pedalboard
