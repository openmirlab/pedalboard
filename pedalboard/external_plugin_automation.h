/* Shared display-unit plugin curves and native audio/MIDI bindings.
 * Format backends supply pure conversion and stable parameter IDs.
 * GPL-3.0-only. */
#pragma once
#include "ExternalPlugin.h"
#include "parameter_automation.h"
#include <cstdlib>
#include <set>

namespace Pedalboard {

struct HostedDisplayNumber {
  double value, tolerance;
  juce::String canonicalUnits;
};

inline HostedDisplayNumber parseHostedDisplayNumber(juce::String text,
                                                    const juce::String &units) {
  text = text.trim()
             .replace(juce::String::charToString(0x2212), "-")
             .replace(juce::String::charToString(0x221e), "inf");
  const auto bytes = text.toStdString();
  char *end = nullptr;
  const double value = std::strtod(bytes.c_str(), &end);
  if (end == bytes.c_str() && units.isEmpty()) {
    const auto word = text.toLowerCase();
    if (word == "off" || word == "false" || word == "no" || word == "disabled")
      return {0, 0, {}};
    if (word == "on" || word == "true" || word == "yes" || word == "enabled")
      return {1, 0, {}};
  }
  if (end == bytes.c_str() || std::isnan(value))
    throw std::invalid_argument(
        "External plugin automation requires a numeric display value");
  const auto suffix = juce::String::fromUTF8(end).trim();
  const auto lower = suffix.toLowerCase();
  double scale = 1;
  juce::String canonical = units;
  if (units.isEmpty()) {
    if (lower == "hz" || lower == "khz" || lower == "mhz") {
      canonical = "Hz";
      scale = lower == "khz" ? 1e3 : lower == "mhz" ? 1e6 : 1;
    } else if (lower == "db" || lower == "dbtp" || lower == "ms" ||
               lower == "s" || lower == "sec" || lower == "seconds" ||
               lower == "%" || lower == "x" || lower == "*") {
      canonical = (lower == "sec" || lower == "seconds") ? "s" : lower;
    } else if (suffix.isNotEmpty())
      throw std::invalid_argument("Ambiguous plugin display units");
  } else if (suffix.isNotEmpty() && !suffix.equalsIgnoreCase(units)) {
    if (!suffix.endsWithIgnoreCase(units))
      throw std::invalid_argument("Ambiguous plugin display units");
    const auto prefix = suffix.dropLastCharacters(units.length()).trim();
    if (prefix == "k")
      scale = 1e3;
    else if (prefix == "M")
      scale = 1e6;
    else if (prefix == "m")
      scale = 1e-3;
    else if (prefix == "u" || prefix == juce::String::charToString(0x00b5))
      scale = 1e-6;
    else if (prefix.isNotEmpty())
      throw std::invalid_argument("Unsupported plugin display prefix");
  }
  if (std::isinf(value))
    return {value * scale, 0, canonical};
  const std::string numeric(bytes.c_str(), (size_t)(end - bytes.c_str()));
  const auto exponentAt = numeric.find_first_of("eE");
  const int exponent = exponentAt == std::string::npos
                           ? 0
                           : std::atoi(numeric.c_str() + exponentAt + 1);
  const auto decimalAt = numeric.find('.');
  const size_t digitsEnd =
      exponentAt == std::string::npos ? numeric.size() : exponentAt;
  const int decimals =
      decimalAt == std::string::npos ? 0 : (int)(digitsEnd - decimalAt - 1);
  return {value * scale,
          std::abs(scale) * .51 * std::pow(10.0, exponent - decimals) +
              std::numeric_limits<double>::epsilon() * 32 *
                  std::max(1.0, std::abs(value * scale)),
          canonical};
}

// The SDK plain-value model is used only when it agrees with displayed units.
// Otherwise use the controller's display parser. Every requested value must
// round-trip at the plugin's display precision; no inferred/linear LUT
// fallback.
template <typename Backend> class HostedDisplayConverter {
public:
  HostedDisplayConverter(juce::AudioPluginInstance &instance,
                         typename Backend::Parameter parameter)
      : instance(instance), parameter(std::move(parameter)) {
    if (!this->parameter.automatable)
      throw std::invalid_argument("External plugin parameter is read-only, a "
                                  "program selector or not automatable");
    modelSignature = this->parameter.name.toStdString() + "|" +
                     this->parameter.units.toStdString() + "|" +
                     std::to_string(this->parameter.steps);
    bool increasing = true, decreasing = true;
    double previous = 0;
    for (int i = 0; i <= 16; ++i) {
      const double raw = i / 16.0;
      const auto display = parsed(raw);
      const double plain = Backend::plain(instance, this->parameter, raw);
      const double back = Backend::normalized(instance, this->parameter, plain);
      modelSignature += "|" + juce::String(display.value, 17).toStdString() +
                        ":" +
                        juce::String(display.tolerance, 17).toStdString() +
                        ":" + juce::String(plain, 17).toStdString();
      usePlain = usePlain && std::isfinite(plain) && std::isfinite(back) &&
                 std::abs(plain - display.value) <= display.tolerance &&
                 std::abs(back - raw) <= 1e-6;
      if (i) {
        increasing = increasing && display.value >= previous;
        decreasing = decreasing && display.value <= previous;
      }
      previous = display.value;
      if (i == 0)
        low = display.value;
      if (i == 16)
        high = display.value;
    }
    if ((!increasing && !decreasing) || low == high)
      throw std::invalid_argument(
          "External plugin parameter has no unambiguous monotonic numeric "
          "display range");
    displayEndpoints = {low, high};
    if (this->parameter.steps == 1 && this->parameter.units.isEmpty() &&
        ((low == 0 && high == 1) || (low == 1 && high == 0))) {
      booleanDisplay = true;
      booleanTrueRaw = high == 1 ? 1 : 0;
    }
    if (usePlain) {
      low = Backend::plain(instance, this->parameter, 0);
      high = Backend::plain(instance, this->parameter, 1);
    }
    if (low > high)
      std::swap(low, high);
  }
  const std::string &signature() const { return modelSignature; }
  float convert(double value) {
    if (std::isinf(value)) {
      if (value == displayEndpoints[0])
        return 0;
      if (value == displayEndpoints[1])
        return 1;
    }
    if (booleanDisplay) {
      if (value != 0 && value != 1)
        throw std::invalid_argument(
            "Boolean plugin curves require only false/true or 0/1");
      return value == 1 ? booleanTrueRaw : 1 - booleanTrueRaw;
    }
    if (!std::isfinite(value) || value < low || value > high)
      throw std::invalid_argument("External plugin automation value is outside "
                                  "the displayed parameter range");
    const double raw =
        usePlain
            ? Backend::normalized(instance, parameter, value)
            : Backend::fromText(instance, parameter, juce::String(value, 17));
    if (!std::isfinite(raw) || raw < 0 || raw > 1)
      throw std::invalid_argument(
          "External plugin controller returned an invalid normalized value");
    const auto back = parsed(raw);
    double tolerance = back.tolerance;
    if (std::abs(back.value - value) > tolerance) {
      // AU values themselves are float32 in physical units. A nonlinear
      // display can amplify a single physical ULP (e.g. linear gain -> dB).
      // Measure that native quantization through the display, not an assumed
      // epsilon in display units. VST3 returns the same raw value here.
      for (int direction : {-1, 1}) {
        const double adjacent =
            Backend::neighbor(instance, parameter, raw, direction);
        if (adjacent != raw) {
          const auto display = parsed(adjacent);
          if (std::isfinite(display.value) && std::isfinite(back.value))
            tolerance =
                std::max(tolerance,
                         back.tolerance + std::abs(display.value - back.value));
        }
      }
    }
    if (std::abs(back.value - value) > tolerance)
      throw std::invalid_argument(
          "External plugin display-unit conversion does not "
          "round-trip; refusing an approximate curve");
    return (float)raw;
  }

private:
  HostedDisplayNumber parsed(double raw) {
    auto result = parseHostedDisplayNumber(
        Backend::text(instance, parameter, raw), parameter.units);
    result.tolerance =
        std::max(result.tolerance, Backend::precision(result.value));
    if (result.canonicalUnits.isNotEmpty()) {
      if (displayUnits.isEmpty())
        displayUnits = result.canonicalUnits;
      else if (displayUnits != result.canonicalUnits)
        throw std::invalid_argument("External plugin parameter changes "
                                    "display-unit dimensions across its range");
    }
    return result;
  }
  juce::AudioPluginInstance &instance;
  typename Backend::Parameter parameter;
  bool usePlain = true, booleanDisplay = false;
  float booleanTrueRaw = 1;
  double low = 0, high = 0;
  std::array<double, 2> displayEndpoints{};
  juce::String displayUnits;
  std::string modelSignature;
};

struct HostedCurveModel {
  std::string signature;
  std::vector<double> physical;
};

template <typename Format, typename Backend>
inline void initHostedAutomation(py::module &module, const char *className) {
  using Effect = ExternalPlugin<Format>;
  DenseParameterFactory factory = [](std::shared_ptr<Plugin> plugin,
                                     py::dict mapping, size_t frames,
                                     double sampleRate, unsigned int channels,
                                     ChannelLayout layout, bool reset) {
    auto self = std::dynamic_pointer_cast<Effect>(plugin);
    std::scoped_lock lock(self->mutex);
    auto &instance = self->automationInstance();
    // Naming metadata only: do not instantiate the Python range-guessing
    // wrapper, which has a parameter-mutating fallback for some third-party
    // plugins.
    auto normalize = py::module_::import("pedalboard._pedalboard")
                         .attr("to_python_parameter_name");
    std::map<std::string, std::vector<int>> aliases;
    for (auto *parameter : self->getParameters()) {
      auto alias =
          normalize(py::cast(parameter, py::return_value_policy::reference));
      if (!alias.is_none())
        aliases[alias.template cast<std::string>()].push_back(
            parameter->getParameterIndex());
    }
    std::vector<DenseParameterControl> controls;
    std::vector<uint32_t> ids;
    for (auto entry : mapping) {
      if (!py::isinstance<py::str>(entry.first))
        throw py::type_error(
            "External plugin automation keys must be parameter names");
      const auto name = py::cast<std::string>(entry.first);
      auto found = aliases.find(name);
      if (found == aliases.end() || found->second.size() != 1)
        throw py::value_error(
            "Unknown or ambiguous plugin automation parameter: " + name);
      const auto description =
          Backend::describe(instance, found->second.front());
      if (!py::isinstance<py::array>(entry.second))
        throw py::type_error("Each automation curve must be a NumPy array");
      auto curve = py::reinterpret_borrow<py::array>(entry.second);
      const std::string kind = py::str(curve.dtype().attr("kind"));
      if (kind != "f" && kind != "i" && kind != "u" &&
          !(kind == "b" && description.steps == 1 &&
            description.units.isEmpty()))
        throw py::type_error(
            "External plugin display curves must contain real numbers");
      if (curve.ndim() != 1 || (size_t)curve.shape(0) != frames)
        throw py::value_error(
            "Curves require one display-unit value per input frame");
      auto values =
          py::array_t<double,
                      py::array::c_style | py::array::forcecast>::ensure(curve);
      auto data = values.unchecked<1>();
      std::vector<float> normalized;
      normalized.reserve(frames);
      auto model = std::make_shared<HostedCurveModel>();
      model->physical.reserve(frames);
      if (frames) {
        try {
          HostedDisplayConverter<Backend> converter(instance, description);
          model->signature = converter.signature();
          double previous = std::numeric_limits<double>::quiet_NaN();
          float converted = 0;
          for (size_t i = 0; i < frames; ++i) {
            if (data(i) != previous) {
              converted = converter.convert(data(i));
              previous = data(i);
            }
            normalized.push_back(converted);
            model->physical.push_back(data(i));
          }
        } catch (const std::exception &error) {
          throw py::value_error("Invalid plugin automation for '" + name +
                                "': " + error.what());
        }
      }
      const auto id = description.id;
      controls.push_back({std::move(normalized),
                          [self, id](float value) {
                            Backend::set(self->automationInstance(), id, value);
                          },
                          {},
                          name,
                          id});
      controls.back().validateSchema = [self, id, model] {
        auto &current = self->automationInstance();
        HostedDisplayConverter<Backend> converter(
            current, Backend::describeById(current, id));
        if (converter.signature() != model->signature)
          throw std::invalid_argument(
              "External plugin conversion model changed while "
              "targets are pending; reset the root");
      };
      controls.back().resolvePrepared = [self, id, model](auto &control,
                                                          bool continuation) {
        auto &current = self->automationInstance();
        HostedDisplayConverter<Backend> converter(
            current, Backend::describeById(current, id));
        const bool changed = converter.signature() != model->signature;
        if (changed) {
          if (continuation || model->physical.size() != control.values.size())
            throw std::invalid_argument(
                "External plugin conversion model changed in a "
                "stream; reset the plugin/root");
          double previous = std::numeric_limits<double>::quiet_NaN();
          float value = 0;
          for (size_t i = 0; i < model->physical.size(); ++i) {
            if (model->physical[i] != previous) {
              value = converter.convert(model->physical[i]);
              previous = model->physical[i];
            }
            control.values[i] = value;
          }
          model->signature = converter.signature();
        }
        std::vector<double>().swap(model->physical);
        return changed;
      };
      ids.push_back(id);
    }
    std::sort(controls.begin(), controls.end(),
              [](const auto &a, const auto &b) {
                return *a.externalId < *b.externalId;
              });
    auto adapter = std::make_shared<DenseParameterProcessor>(
        self, std::move(controls), layout, channels,
        [self, reset, sampleRate, channels](const auto &values) {
          self->validateAutomationStart(reset, sampleRate, channels);
          for (const auto &control : values) {
            const auto info = Backend::describeById(self->automationInstance(),
                                                    *control.externalId);
            if (!info.automatable)
              throw std::invalid_argument(
                  "External plugin automation parameter changed; restart the "
                  "render");
          }
        },
        [self, ids] { self->beginAutomation(ids); },
        [self](const auto &context, const auto &values, size_t cursor) {
          return self->processAutomated(context, values, cursor);
        },
        DenseParameterLatency::Buffered, reset,
        [self](const auto &) { self->requestAutomation(); });
    return adapter;
  };
  auto binding = py::reinterpret_borrow<
      py::class_<Effect, AbstractExternalPlugin, std::shared_ptr<Effect>>>(
      module.attr(className));
  binding.def_property_readonly("audio_buses", &Effect::getAudioBuses,
                                "Fresh input/output audio bus metadata; "
                                "indices are routing identifiers.");
  binding.def(
      "process_buses", &Effect::processBuses, py::arg("input_buses"),
      py::arg("sample_rate"), py::kw_only(), py::arg("output_buses"),
      py::arg("num_frames") = py::none(),
      py::arg("buffer_size") = DEFAULT_BUFFER_SIZE, py::arg("reset") = true,
      "Route planar arrays through explicit audio buses. Return a dict from "
      "output bus index to planar float32 audio, with raw plugin latency. "
      "Omitted VST3 buses are disabled; omitted AU inputs receive silence. "
      "Layouts must be jointly supported. "
      "Reset before changing routing, rate or processing API.");
  binding.def(
      "prepare_automation",
      [](std::shared_ptr<Effect> self, double rate, unsigned int channels) {
        py::gil_scoped_release release;
        std::scoped_lock lock(self->mutex);
        self->prepareAutomation(rate, channels);
      },
      py::arg("sample_rate"), py::arg("num_channels") = 2,
      "Reset and configure the plugin format before querying display-unit "
      "curves; continue with reset=False.");
  binding.def(
      "reset",
      [](std::shared_ptr<Effect> self) {
        py::gil_scoped_release release;
        std::scoped_lock lock(self->mutex);
        self->reset();
        self->resetLastChannelLayout();
      },
      "Clear plugin processing and automation memory while preserving "
      "parameter "
      "values.");
  binding.def_property(
      "automation_mode", &Effect::getAutomationMode, &Effect::setAutomationMode,
      "Native automation delivery mode; configure before starting a stream.");
  auto midiCallback = [factory](std::shared_ptr<Effect> self, py::object midi,
                                float duration, float rate,
                                unsigned int channels, unsigned long bufferSize,
                                bool reset, py::object automation,
                                bool compensateLatency) {
    if (compensateLatency && !reset)
      throw py::value_error("Instrument latency compensation requires "
                            "reset=True for an offline clip");
    if (automation.is_none())
      return self->renderMIDIMessages(midi, duration, rate, channels,
                                      bufferSize, reset, compensateLatency);
    if (!py::isinstance<py::dict>(automation))
      throw py::type_error(
          "automation must be a dict of display-unit NumPy curves");
    auto mapping = automation.cast<py::dict>();
    if (mapping.empty())
      return self->renderMIDIMessages(midi, duration, rate, channels,
                                      bufferSize, reset, compensateLatency);
    const double count = (double)(duration * rate);
    if (!std::isfinite(duration) || duration < 0 || !std::isfinite(rate) ||
        rate <= 0 || !std::isfinite(count) || count > INT_MAX || !channels ||
        channels > INT_MAX || !bufferSize || bufferSize > INT_MAX)
      throw py::value_error(
          "Unsafe MIDI duration, rate, channels or buffer size");
    auto adapter = factory(self, mapping, (size_t)count, rate, channels,
                           ChannelLayout::NotInterleaved, reset);
    return self->renderMIDIMessagesAutomated(midi, duration, rate, channels,
                                             bufferSize, reset, adapter,
                                             compensateLatency);
  };
  for (const char *method : {"process", "__call__"})
    binding.def(
        method, midiCallback, py::arg("midi_messages"), py::arg("duration"),
        py::arg("sample_rate"), py::arg("num_channels") = 2,
        py::arg("buffer_size") = DEFAULT_BUFFER_SIZE, py::arg("reset") = true,
        py::kw_only(), py::arg("automation") = py::none(),
        py::arg("compensate_latency") = false,
        "Render MIDI with synchronized display-unit parameter curves on the "
        "instrument input clock. With compensate_latency=True, require "
        "reset=True, "
        "drain the fixed reported latency and align exactly the requested "
        "duration. "
        "Default False preserves raw streaming output.");
  bindDenseParameterFactory(binding, std::move(factory));
}
} // namespace Pedalboard
