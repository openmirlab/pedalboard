/*
 * Native dense parameter scheduling for the first stateless-effect batch.
 * Reuses Pedalboard's audio conversion, tree locking and scalar DSP.
 * Licensed under the GNU General Public License, Version 3.0.
 */
#pragma once

#include "process.h"
#include <cmath>
#include <functional>
#include <limits>

namespace Pedalboard {

// This private container makes the existing processing lock cover the child.
// Only zero-latency effects are admitted by the explicit bindings below.
class DenseParameterProcessor : public PluginContainer {
public:
  DenseParameterProcessor(std::shared_ptr<Plugin> effect,
                          std::vector<float> values,
                          std::function<void(float)> setter,
                          ChannelLayout layout, unsigned int channels)
      : PluginContainer({effect}), values(std::move(values)),
        setter(std::move(setter)) {
    lastChannelLayout = layout;
    lastSpec.numChannels = channels;
  }

  void prepare(const juce::dsp::ProcessSpec &spec) override {
    plugins.front()->prepare(spec);
    lastSpec = spec;
  }

  void reset() override { plugins.front()->reset(); }

  int process(
      const juce::dsp::ProcessContextReplacing<float> &context) override {
    auto block = context.getOutputBlock();
    const size_t count = block.getNumSamples();
    if (cursor + count > values.size())
      throw std::runtime_error(
          "Automation consumed more frames than supplied.");

    size_t start = 0;
    while (start < count) {
      const float value = values[cursor + start];
      size_t end = start + 1;
      while (end < count && values[cursor + end] == value)
        ++end;
      if (!lastValue || *lastValue != value) {
        setter(value);
        lastValue = value;
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
  std::vector<float> values;
  std::function<void(float)> setter;
  std::optional<float> lastValue;
  size_t cursor = 0;
};

// Attach the same optional keyword to process and its callable alias, without
// changing Plugin's public interface or affecting unsupported processors.
template <typename Effect, typename... Options, typename Setter>
void bindDenseParameter(py::class_<Effect, Options...> &binding,
                        const char *parameterName, Setter update,
                        bool decibels = true) {
  auto callback = [=](std::shared_ptr<Effect> self, py::array input,
                      double sampleRate, unsigned int bufferSize, bool reset,
                      py::object automation) {
    if (automation.is_none())
      return process(input, sampleRate, {self}, bufferSize, reset);
    if (!py::isinstance<py::dict>(automation))
      throw py::type_error("automation must be a dict of NumPy curves.");
    auto mapping = automation.cast<py::dict>();
    if (mapping.empty())
      return process(input, sampleRate, {self}, bufferSize, reset);
    if (mapping.size() != 1 || !mapping.contains(parameterName))
      throw py::value_error(std::string("Supported automation parameter: ") +
                            parameterName);
    if (!std::isfinite(sampleRate) || sampleRate <= 0 || bufferSize == 0 ||
        bufferSize > (unsigned int)std::numeric_limits<int>::max())
      throw py::value_error("sample_rate and buffer_size must be positive and "
                            "finite, and buffer_size must fit a native int.");
    auto control = mapping[parameterName];
    if (!py::isinstance<py::array>(control))
      throw py::type_error("Each automation curve must be a NumPy array.");
    auto curve = py::reinterpret_borrow<py::array>(control);
    const std::string kind = py::str(curve.dtype().attr("kind"));
    if (kind != "f" && kind != "i" && kind != "u")
      throw py::type_error("Automation curves must contain real numbers.");
    if (curve.ndim() != 1)
      throw py::value_error("Automation curves must be one-dimensional.");

    py::array_t<float, py::array::c_style> audio;
    if (input.dtype().char_() == 'f')
      audio = input;
    else if (input.dtype().char_() == 'd')
      audio = input.attr("astype")("float32");
    else
      throw py::type_error("Audio must use float32 or float64.");

    // Match the original layout inference, including cached tiny/square
    // streams.
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
    if ((size_t)curve.shape(0) != frames)
      throw py::value_error("Curve length must equal input audio frames.");
    if (frames > (size_t)std::numeric_limits<int>::max() ||
        channels > (size_t)std::numeric_limits<int>::max())
      throw py::value_error("Audio dimensions exceed native buffer capacity.");

    auto numbers =
        py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
            curve);
    if (!numbers)
      throw py::type_error(
          "Unable to convert automation curve to real values.");
    auto data = numbers.template unchecked<1>();
    std::vector<float> values;
    values.reserve(frames);
    for (size_t i = 0; i < frames; ++i) {
      const double d = data(i);
      const float value = (float)d;
      if (decibels) {
        if (std::isnan(d) || d == std::numeric_limits<double>::infinity() ||
            (std::isfinite(d) && !std::isfinite(value)) ||
            !std::isfinite(juce::Decibels::decibelsToGain(value)))
          throw py::value_error("dB curves require representable gains; only "
                                "negative infinity is allowed for silence.");
      } else if (!std::isfinite(d) || d < 0 || d > 32) {
        throw py::value_error(
            "bit_depth curves must be finite and in [0, 32].");
      }
      values.push_back(value);
    }
    // Empty requests must not reset DSP or alter its parameter.
    if (!frames || !channels)
      return process(audio, sampleRate, {}, bufferSize, false);

    auto adapter = std::make_shared<DenseParameterProcessor>(
        self, std::move(values),
        [self, update](float value) { update(*self, value); }, layout,
        channels);
    return process(audio, sampleRate, {adapter}, bufferSize, reset);
  };
  for (const char *method : {"process", "__call__"}) {
    binding.def(
        method, callback, py::arg("input_array"), py::arg("sample_rate"),
        py::arg("buffer_size") = DEFAULT_BUFFER_SIZE, py::arg("reset") = true,
        py::kw_only(), py::arg("automation") = py::none(),
        R"(Process audio with optional native parameter automation.

The automation dict maps this effect's supported parameter to a one-dimensional
real NumPy array, one target per input frame (shared across channels). Curves are
validated and copied before processing. No extra interpolation or smoothing is
added. The final value persists for later scalar calls. Empty curves with empty
audio do not change the parameter. Omit automation to use the existing fast path.
Use reset=False with matching curve slices to continue a stream.)");
  }
}
} // namespace Pedalboard
