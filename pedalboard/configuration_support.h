/* Private native support for configuration persistence. GPL-3.0-only. */
#pragma once
#include "ExternalPlugin.h"

namespace Pedalboard {
struct ConfigurationParameter {
  std::string id, name, label, text;
  float raw;
  std::optional<float> canonical;
};

template <typename Format>
void bindConfigurationSupport(py::module &m, const char *name) {
  using Effect = ExternalPlugin<Format>;
  auto binding = py::reinterpret_borrow<
      py::class_<Effect, AbstractExternalPlugin, std::shared_ptr<Effect>>>(
      m.attr(name));
  binding.def(
      "_configuration_snapshot",
      [](std::shared_ptr<Effect> self, bool includeState) {
        if (!juce::MessageManager::getInstance()->isThisTheMessageThread())
          throw std::runtime_error(
              "Plugin configuration requires the main thread");
        std::vector<ConfigurationParameter> parameters;
        juce::MemoryBlock state;
        juce::String path, identifier, version, pluginName;
        {
          py::gil_scoped_release release;
          std::scoped_lock lock(self->mutex);
          auto &instance = self->automationInstance();
          path = self->configurationPath();
          identifier = self->foundPluginDescription.createIdentifierString();
          version = self->foundPluginDescription.version;
          pluginName = instance.getName();
          if (includeState)
            self->getState(state);
          for (int i = 0; i < instance.getParameters().size(); ++i) {
            auto *parameter = instance.getHostedParameter(i);
            if (!parameter || !parameter->isAutomatable())
              continue;
#if JUCE_MAC && JUCE_PLUGINHOST_AU
            // JUCE's legacy AU flag is not sufficient for read-only AUv3
            // meters. Capture only writable, real-time native parameters.
            if constexpr (std::is_same_v<Format, juce::AudioUnitPluginFormat>)
              if (!AudioUnitAutomationBackend::isAutomatable(instance, i))
                continue;
#endif
            const auto raw = parameter->getValue();
            const auto text = parameter->getText(raw, 512);
            const auto canonical = parameter->getValueForText(text);
            std::optional<float> validated;
            if (std::isfinite(canonical) && canonical >= 0 && canonical <= 1 &&
                parameter->getText(canonical, 512) == text)
              validated = canonical;
            parameters.push_back({parameter->getParameterID().toStdString(),
                                  parameter->getName(512).toStdString(),
                                  parameter->getLabel().toStdString(),
                                  text.toStdString(), raw, validated});
          }
        }
        py::dict result;
        result["path"] = path.toStdString();
        result["identifier"] = identifier.toStdString();
        result["version"] = version.toStdString();
        result["name"] = pluginName.toStdString();
        result["state"] =
            py::bytes((const char *)state.getData(), state.getSize());
        py::list rows;
        for (const auto &p : parameters) {
          py::dict row;
          row["id"] = p.id;
          row["name"] = p.name;
          row["label"] = p.label;
          row["text"] = p.text;
          row["raw"] = p.raw;
          row["canonical"] = p.canonical ? py::cast(*p.canonical) : py::none();
          rows.append(row);
        }
        result["parameters"] = rows;
        return result;
      },
      py::arg("include_state") = true);
  binding.def("_configuration_restore", [](std::shared_ptr<Effect> self,
                                           py::bytes data, double rate,
                                           unsigned int channels) {
    if (!juce::MessageManager::getInstance()->isThisTheMessageThread())
      throw std::runtime_error("Plugin configuration requires the main thread");
    const std::string bytes = data;
    py::gil_scoped_release release;
    std::scoped_lock lock(self->mutex);
    self->prepareAutomation(rate, channels);
    self->setState(bytes.data(), bytes.size());
  });
}

inline void init_configuration_support(py::module &m) {
  m.def("_configuration_service_messages", [](double seconds) {
    if (!std::isfinite(seconds) || seconds < 0 || seconds > 2)
      throw std::invalid_argument("settle_seconds must be in [0, 2]");
    if (!juce::MessageManager::getInstance()->isThisTheMessageThread())
      throw std::runtime_error("Plugin configuration requires the main thread");
    py::gil_scoped_release release;
    if (seconds > 0)
      juce::MessageManager::getInstance()->runDispatchLoopUntil(
          (int)std::ceil(seconds * 1000));
  });
  bindConfigurationSupport<juce::PatchedVST3PluginFormat>(m, "VST3Plugin");
#if JUCE_MAC && JUCE_PLUGINHOST_AU
  bindConfigurationSupport<juce::AudioUnitPluginFormat>(m, "AudioUnitPlugin");
#endif
}
} // namespace Pedalboard
