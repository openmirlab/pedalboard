/*
 * Native dense parameter scheduling for explicitly supported builtin effects.
 * Owns validated curves and synchronized updates; reuses process.h conversions
 * and tree locking. Effect-specific bindings own domains and DSP preparation.
 * Licensed under the GNU General Public License, Version 3.0.
 */
#pragma once

#include "process.h"
#include <cmath>
#include <functional>
#include <limits>

namespace Pedalboard {

enum class DenseParameterDomain {
  Decibels,
  BitDepth,
  Threshold,
  LimiterThreshold,
  Ratio,
  TimeMilliseconds,
  CutoffFrequency,
  Positive,
  EQGain,
  UnitInterval,
  DelaySeconds,
  Drive
};

inline bool validDenseParameter(double original, float value,
                                DenseParameterDomain domain,
                                double sampleRate) {
  if (domain == DenseParameterDomain::Decibels)
    return !std::isnan(original) &&
           original != std::numeric_limits<double>::infinity() &&
           (!std::isfinite(original) || std::isfinite(value)) &&
           std::isfinite(juce::Decibels::decibelsToGain(value));
  if (!std::isfinite(original) || !std::isfinite(value))
    return false;
  switch (domain) {
  case DenseParameterDomain::CutoffFrequency: {
    if (!(original > 0 && original < sampleRate * 0.5 && value > 0 &&
          value < sampleRate * 0.5) ||
        !std::isfinite((float)sampleRate))
      return false;
    // At extreme cutoffs, float32 tan/normalization can put the pole on or
    // outside the unit circle even when the requested Hz is below Nyquist.
    const auto coefficients =
        juce::dsp::IIR::ArrayCoefficients<float>::makeFirstOrderLowPass(
            sampleRate, value);
    const float inverse = 1.0f / coefficients[2];
    return coefficients[2] > 0 && std::isfinite(inverse) &&
           std::isfinite(coefficients[0] * inverse) &&
           std::abs(coefficients[3] * inverse) < 1.0f;
  }
  case DenseParameterDomain::Positive:
    return original > 0 && value > 0;
  case DenseParameterDomain::EQGain: {
    const float gain = juce::Decibels::decibelsToGain(value);
    return gain > 0 && std::isfinite(gain);
  }
  case DenseParameterDomain::BitDepth:
    return original >= 0 && original <= 32;
  case DenseParameterDomain::DelaySeconds:
    return original >= 0 && original <= 30;
  case DenseParameterDomain::UnitInterval:
    return original >= 0 && original <= 1;
  case DenseParameterDomain::Ratio:
  case DenseParameterDomain::Drive:
    return original >= 1;
  case DenseParameterDomain::TimeMilliseconds:
    return original >= 0;
  case DenseParameterDomain::Threshold:
  case DenseParameterDomain::LimiterThreshold: {
    const float threshold = juce::Decibels::decibelsToGain(value, -200.0f);
    if (!(threshold > 0) || !std::isfinite(threshold) ||
        !std::isfinite(1.0f / threshold))
      return false;
    if (domain == DenseParameterDomain::LimiterThreshold) {
      const float makeup = (float)std::pow(10.0, 10.0 * (1.0 - 0.25) / 40.0) *
                           juce::Decibels::decibelsToGain(-value, -100.0f);
      return std::isfinite(makeup);
    }
    return true;
  }
  default:
    return false;
  }
}

struct DenseParameterControl {
  std::vector<float> values;
  std::function<void(float)> setter;
  std::optional<float> lastValue;
  std::string name;
};

using DenseControlPreflight =
    std::function<void(const std::vector<DenseParameterControl> &)>;

// This private container makes the existing processing lock cover the child.
// Only zero-latency effects are admitted by explicit bindings.
class DenseParameterProcessor : public PluginContainer {
public:
  DenseParameterProcessor(std::shared_ptr<Plugin> effect,
                          std::vector<DenseParameterControl> controls,
                          ChannelLayout layout, unsigned int channels,
                          DenseControlPreflight preflight = {})
      : PluginContainer({effect}), controls(std::move(controls)),
        preflight(std::move(preflight)) {
    lastChannelLayout = layout;
    lastSpec.numChannels = channels;
  }

  void prepare(const juce::dsp::ProcessSpec &spec) override {
    validateOnce();
    applyControls(0);
    plugins.front()->prepare(spec);
    lastSpec = spec;
  }

  void reset() override {
    // In particular, Limiter must initialize its makeup ramp from the first
    // supplied threshold, not the previous/default scalar threshold.
    validateOnce();
    applyControls(0);
    plugins.front()->reset();
  }

  int process(
      const juce::dsp::ProcessContextReplacing<float> &context) override {
    auto block = context.getOutputBlock();
    const size_t count = block.getNumSamples();
    if (cursor + count > controls.front().values.size())
      throw std::runtime_error(
          "Automation consumed more frames than supplied.");

    size_t start = 0;
    while (start < count) {
      applyControls(cursor + start);
      size_t end = start + 1;
      for (; end < count; ++end) {
        bool changed = false;
        for (const auto &control : controls) {
          if (control.values[cursor + end] != control.values[cursor + start]) {
            changed = true;
            break;
          }
        }
        if (changed)
          break;
      }
      auto span = block.getSubBlock(start, end - start);
      juce::dsp::ProcessContextReplacing<float> subContext(span);
      if (plugins.front()->process(subContext) != (int)(end - start))
        throw std::runtime_error(
            "Dense automation requires a zero-latency effect.");
      start = end;
    }
    cursor += count;
    return (int)count;
  }

private:
  void validateOnce() {
    // process.h holds the effect's tree lock here. Omitted scalar controls
    // cannot change between joint validation and native processing calls.
    if (!validated) {
      if (preflight)
        preflight(controls);
      validated = true;
    }
  }

  void applyControls(size_t frame) {
    for (auto &control : controls) {
      const float value = control.values[frame];
      if (!control.lastValue || *control.lastValue != value) {
        control.setter(value);
        control.lastValue = value;
      }
    }
  }

  std::vector<DenseParameterControl> controls;
  size_t cursor = 0;
  DenseControlPreflight preflight;
  bool validated = false;
};

template <typename Effect> struct DenseParameterSpec {
  const char *name;
  std::function<void(Effect &, float)> update;
  DenseParameterDomain domain;
};

template <typename Effect>
using DenseParameterPreflight =
    std::function<void(const Effect &,
                       const std::vector<DenseParameterControl> &, double,
                       unsigned int)>;

// The specification order defines simultaneous updates, independently of the
// Python dict's insertion order. All controls validate before DSP mutation.
template <typename Effect, typename... Options>
void bindDenseParameters(py::class_<Effect, Options...> &binding,
                         std::vector<DenseParameterSpec<Effect>> specs,
                         DenseParameterPreflight<Effect> preflight = {}) {
  auto callback = [specs, preflight](std::shared_ptr<Effect> self,
                                     py::array input, double sampleRate,
                                     unsigned int bufferSize, bool reset,
                                     py::object automation) {
    if (automation.is_none())
      return process(input, sampleRate, {self}, bufferSize, reset);
    if (!py::isinstance<py::dict>(automation))
      throw py::type_error("automation must be a dict of NumPy curves.");
    auto mapping = automation.cast<py::dict>();
    if (mapping.empty())
      return process(input, sampleRate, {self}, bufferSize, reset);
    for (auto entry : mapping) {
      if (!py::isinstance<py::str>(entry.first))
        throw py::type_error("Automation parameter names must be strings.");
      const auto name = py::cast<std::string>(entry.first);
      if (std::none_of(specs.begin(), specs.end(),
                       [&](const auto &spec) { return name == spec.name; }))
        throw py::value_error("Unsupported automation parameter: " + name);
    }
    if (!std::isfinite(sampleRate) || sampleRate <= 0 || bufferSize == 0 ||
        bufferSize > (unsigned int)std::numeric_limits<int>::max())
      throw py::value_error("sample_rate and buffer_size must be positive and "
                            "finite, and buffer_size must fit a native int.");

    py::array_t<float, py::array::c_style> audio;
    if (input.dtype().char_() == 'f')
      audio = input;
    else if (input.dtype().char_() == 'd')
      audio = input.attr("astype")("float32");
    else
      throw py::type_error("Audio must use float32 or float64.");
    ChannelLayout layout;
    {
      std::scoped_lock lock(self->mutex);
      layout = self->parseAndCacheChannelLayout(audio);
    }
    const auto info = audio.request();
    const size_t frames =
        info.ndim == 1
            ? info.shape[0]
            : info.shape[layout == ChannelLayout::Interleaved ? 0 : 1];
    const size_t channels =
        info.ndim == 1
            ? 1
            : info.shape[layout == ChannelLayout::Interleaved ? 1 : 0];
    if (frames > (size_t)std::numeric_limits<int>::max() ||
        channels > (size_t)std::numeric_limits<int>::max())
      throw py::value_error("Audio dimensions exceed native buffer capacity.");

    std::vector<DenseParameterControl> controls;
    for (const auto &spec : specs) {
      if (!mapping.contains(spec.name))
        continue;
      auto control = mapping[spec.name];
      if (!py::isinstance<py::array>(control))
        throw py::type_error("Each automation curve must be a NumPy array.");
      auto curve = py::reinterpret_borrow<py::array>(control);
      const std::string kind = py::str(curve.dtype().attr("kind"));
      if (kind != "f" && kind != "i" && kind != "u")
        throw py::type_error("Automation curves must contain real numbers.");
      if (curve.ndim() != 1 || (size_t)curve.shape(0) != frames)
        throw py::value_error(
            "Curves must be one-dimensional with one value per input frame.");
      auto numbers =
          py::array_t<double,
                      py::array::c_style | py::array::forcecast>::ensure(curve);
      if (!numbers)
        throw py::type_error(
            "Unable to convert automation curve to real values.");
      auto data = numbers.template unchecked<1>();
      std::vector<float> values;
      values.reserve(frames);
      for (size_t i = 0; i < frames; ++i) {
        const double d = data(i);
        const float value = (float)d;
        if (!validDenseParameter(d, value, spec.domain, sampleRate))
          throw py::value_error(std::string("Invalid automation value for ") +
                                spec.name);
        values.push_back(value);
      }
      auto update = spec.update;
      controls.push_back({std::move(values),
                          [self, update](float value) { update(*self, value); },
                          {},
                          spec.name});
    }
    if (!frames || !channels)
      return process(audio, sampleRate, {}, bufferSize, false);
    DenseControlPreflight validation;
    if (preflight)
      validation = [self, sampleRate, channels, preflight](const auto &values) {
        preflight(*self, values, sampleRate, (unsigned int)channels);
      };
    auto adapter = std::make_shared<DenseParameterProcessor>(
        self, std::move(controls), layout, channels, std::move(validation));
    return process(audio, sampleRate, {adapter}, bufferSize, reset);
  };
  for (const char *method : {"process", "__call__"})
    binding.def(
        method, callback, py::arg("input_array"), py::arg("sample_rate"),
        py::arg("buffer_size") = DEFAULT_BUFFER_SIZE, py::arg("reset") = true,
        py::kw_only(), py::arg("automation") = py::none(),
        R"(Process audio with optional synchronized native parameter automation.

The automation dict maps supported parameter names to one-dimensional real NumPy
arrays, one target per input frame shared by all channels. Curves validate and
copy before processing. All changes at a frame apply before its audio is processed.
The first values initialize reset/prepare; reset=False retains processing memory.
No extra interpolation or smoothing is added. Final values persist for scalar
calls. Empty curves with empty audio leave DSP unchanged. Omit automation for
the existing scalar path; slice audio and curves together to continue a stream.)");
}

// Keep the existing one-control bindings as consumers of the same scheduler.
template <typename Effect, typename... Options, typename Setter>
void bindDenseParameter(py::class_<Effect, Options...> &binding,
                        const char *parameterName, Setter update,
                        bool decibels = true) {
  bindDenseParameters(binding, {{parameterName, update,
                                 decibels ? DenseParameterDomain::Decibels
                                          : DenseParameterDomain::BitDepth}});
}
} // namespace Pedalboard
