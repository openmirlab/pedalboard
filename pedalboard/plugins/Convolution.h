/*
 * pedalboard
 * Copyright 2021 Spotify AB
 *
 * Licensed under the GNU Public License, Version 3.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    https://www.gnu.org/licenses/gpl-3.0.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "../JucePlugin.h"
#include "../parameter_automation.h"

#include "../juce_overrides/juce_BlockingConvolution.h"

namespace Pedalboard {

/**
 * Quick wrapper around juce::dsp::BlockingConvolution to
 * allow mixing at arbitrary levels.
 */
class ConvolutionWithMix {
public:
  ConvolutionWithMix() = default;

  juce::dsp::BlockingConvolution &getConvolution() { return convolution; }

  const juce::dsp::BlockingConvolution &getConvolution() const {
    return convolution;
  }

  void setMix(double newMix) noexcept {
    mixer.setWetMixProportion(newMix);
    mix = newMix;
  }

  double getMix() const noexcept { return mix; }

  void setImpulseResponseFilename(std::string &filename) {
    impulseResponseFilename = filename;
  }

  const std::optional<std::string> &getImpulseResponseFilename() const {
    return impulseResponseFilename;
  }

  void setImpulseResponse(juce::AudioBuffer<float> &&ir) {
    impulseResponse = {std::move(ir)};
  }

  const std::optional<juce::AudioBuffer<float>> &getImpulseResponse() const {
    return {impulseResponse};
  }

  void setSampleRate(double sr) { sampleRate = {sr}; }

  const std::optional<double> &getSampleRate() const { return sampleRate; }

  void prepare(const juce::dsp::ProcessSpec &spec) {
    convolution.prepare(spec);
    mixer.prepare(spec);
    mixer.setWetMixProportion(mix);
  }

  void reset() noexcept {
    convolution.reset();
    mixer.reset();
    mixer.setWetMixProportion(mix);
  }

  template <typename ProcessContext>
  void process(const ProcessContext &context) noexcept {
    mixer.pushDrySamples(context.getInputBlock());
    convolution.process(context);
    mixer.mixWetSamples(context.getOutputBlock());
  }

  void
  processAutomated(const juce::dsp::ProcessContextReplacing<float> &context,
                   const std::vector<float> &curve, size_t cursor) {
    // mix does not affect wet-engine input or history. Keep FFT processing
    // bulk.
    mixer.pushDrySamples(context.getInputBlock());
    convolution.process(context);
    auto wet = context.getOutputBlock();
    for (size_t start = 0; start < wet.getNumSamples();) {
      setMix(curve[cursor + start]);
      size_t end = start + 1;
      while (end < wet.getNumSamples() &&
             curve[cursor + end] == curve[cursor + start])
        ++end;
      mixer.mixWetSamples(wet.getSubBlock(start, end - start));
      start = end;
    }
  }

private:
  juce::dsp::BlockingConvolution convolution;
  juce::dsp::DryWetMixer<float> mixer;
  float mix = 1.0;
  std::optional<std::string> impulseResponseFilename;
  std::optional<juce::AudioBuffer<float>> impulseResponse;
  std::optional<double> sampleRate;
};

class Convolution : public JucePlugin<ConvolutionWithMix> {
public:
  void setMix(double value) { getDSP().setMix(value); }

  void validateAutomation(double rate, unsigned int channels) const {
    if (!std::isfinite(rate) || rate < 100 || channels < 1 || channels > 2 ||
        std::floor(rate * 0.05) > std::numeric_limits<int>::max())
      throw std::invalid_argument(
          "Convolution curves require mono/stereo, finite rate >=100Hz and "
          "representable ramps.");
    if (getDSP().getConvolution().getLatency() != 0)
      throw std::invalid_argument(
          "Convolution curves require zero engine latency.");
    const double previousRate =
        lastSpec.sampleRate > 0 ? lastSpec.sampleRate : 44100.0;
    const double projectedIR =
        ((double)getDSP().getConvolution().getCurrentIRSize() + 1) * rate /
            previousRate +
        1;
    if (!std::isfinite(projectedIR) ||
        projectedIR > std::numeric_limits<int>::max() / 4)
      throw std::invalid_argument(
          "Resampled convolution impulse response exceeds native size bounds.");
  }

  void prepare(const juce::dsp::ProcessSpec &spec) override {
    if (lastSpec.sampleRate != spec.sampleRate ||
        lastSpec.numChannels != spec.numChannels) {
      // FFT partition size is significant: retain the normal8192 default,
      // bound pathological tiny/large requests, and never resize on block
      // growth.
      preparedFrames = std::clamp(spec.maximumBlockSize, 128u, 8192u);
      auto bounded = spec;
      bounded.maximumBlockSize = preparedFrames;
      JucePlugin<ConvolutionWithMix>::prepare(bounded);
    }
    lastSpec = spec;
  }

  int process(
      const juce::dsp::ProcessContextReplacing<float> &context) override {
    return processBlocks(context, nullptr, 0);
  }

  int processAutomated(const juce::dsp::ProcessContextReplacing<float> &context,
                       const std::vector<float> &curve, size_t cursor) {
    return processBlocks(context, &curve, cursor);
  }

private:
  int processBlocks(const juce::dsp::ProcessContextReplacing<float> &context,
                    const std::vector<float> *curve, size_t cursor) {
    if (preparedFrames == 0)
      throw std::runtime_error(
          "Convolution must be prepared before processing.");
    auto block = context.getOutputBlock();
    for (size_t start = 0; start < block.getNumSamples();
         start += preparedFrames) {
      auto span =
          block.getSubBlock(start, std::min((size_t)preparedFrames,
                                            block.getNumSamples() - start));
      juce::dsp::ProcessContextReplacing<float> sub(span);
      sub.isBypassed = context.isBypassed;
      if (curve)
        getDSP().processAutomated(sub, *curve, cursor + start);
      else
        getDSP().process(sub);
    }
    return (int)block.getNumSamples();
  }
  unsigned int preparedFrames = 0;
};

inline void init_convolution(py::module &m) {
  auto binding =
      py::class_<Convolution, Plugin, std::shared_ptr<Convolution>>(
          m, "Convolution",
          "An audio convolution, suitable for things like speaker simulation "
          "or "
          "reverb modeling.\n\n"
          "The convolution impulse response can be specified either by "
          "filename or "
          "as a 32-bit floating point NumPy array. If a NumPy array is "
          "provided, "
          "the ``sample_rate`` argument must also be provided to indicate the "
          "sample rate of the impulse response.\n\n*Support for passing NumPy "
          "arrays as impulse responses introduced in v0.9.10.*")
          .def(py::init([](std::variant<std::string,
                                        py::array_t<float, py::array::c_style>>
                               impulseResponse,
                           float mix, std::optional<double> sampleRate) {
                 auto plugin = std::make_unique<Convolution>();

                 if (auto *impulseResponseFilename =
                         std::get_if<std::string>(&impulseResponse)) {
                   py::gil_scoped_release release;
                   // Load the IR file on construction, to handle errors
                   auto inputFile = juce::File(*impulseResponseFilename);
                   // Test opening the file before we pass it to
                   // loadImpulseResponse, which reloads it in the background:
                   {
                     juce::FileInputStream stream(inputFile);
                     if (!stream.openedOk()) {
                       throw std::runtime_error(
                           "Unable to load impulse response: " +
                           *impulseResponseFilename);
                     }
                   }

                   plugin->getDSP().getConvolution().loadImpulseResponse(
                       inputFile, juce::dsp::Convolution::Stereo::yes,
                       juce::dsp::Convolution::Trim::no, 0);
                   plugin->getDSP().setImpulseResponseFilename(
                       *impulseResponseFilename);
                 } else if (auto *inputArray = std::get_if<
                                py::array_t<float, py::array::c_style>>(
                                &impulseResponse)) {
                   if (!sampleRate) {
                     throw std::runtime_error("sample_rate must be provided "
                                              "when passing a numpy array "
                                              "as an impulse response.");
                   }
                   plugin->getDSP().getConvolution().loadImpulseResponse(
                       std::move(copyPyArrayIntoJuceBuffer(*inputArray)),
                       *sampleRate, juce::dsp::Convolution::Stereo::yes,
                       juce::dsp::Convolution::Trim::no,
                       juce::dsp::Convolution::Normalise::yes);

                   plugin->getDSP().setImpulseResponse(
                       copyPyArrayIntoJuceBuffer(*inputArray));
                   plugin->getDSP().setSampleRate(*sampleRate);
                 }
                 plugin->getDSP().setMix(mix);
                 return plugin;
               }),
               py::arg("impulse_response_filename"), py::arg("mix") = 1.0,
               py::arg("sample_rate") = py::none())
          .def("__repr__",
               [](Convolution &plugin) {
                 std::ostringstream ss;
                 ss << "<pedalboard.Convolution";
                 if (plugin.getDSP().getImpulseResponseFilename()) {
                   ss << " impulse_response_filename="
                      << *plugin.getDSP().getImpulseResponseFilename();
                 } else if (plugin.getDSP().getImpulseResponse()) {
                   ss << " impulse_response=<"
                      << plugin.getDSP().getImpulseResponse()->getNumSamples()
                      << " samples of "
                      << plugin.getDSP().getImpulseResponse()->getNumChannels()
                      << "-channel audio at "
                      << *plugin.getDSP().getSampleRate() << " Hz>";
                 }

                 ss << " mix=" << plugin.getDSP().getMix();
                 ss << " at " << &plugin;
                 ss << ">";
                 return ss.str();
               })
          .def_property_readonly(
              "impulse_response_filename",
              [](Convolution &plugin) {
                return plugin.getDSP().getImpulseResponseFilename();
              })
          .def_property_readonly(
              "impulse_response",
              [](Convolution &plugin)
                  -> std::optional<py::array_t<float, py::array::c_style>> {
                if (plugin.getDSP().getImpulseResponse()) {
                  return {copyJuceBufferIntoPyArray(
                      *plugin.getDSP().getImpulseResponse(),
                      ChannelLayout::NotInterleaved, 0, 2)};
                } else {
                  return {};
                }
              })
          .def_property(
              "mix",
              [](Convolution &plugin) { return plugin.getDSP().getMix(); },
              [](Convolution &plugin, double newMix) {
                return plugin.getDSP().setMix(newMix);
              });
  DenseParameterPreflight<Convolution> preflight =
      [](const Convolution &effect, const std::vector<DenseParameterControl> &,
         double rate,
         unsigned int channels) { effect.validateAutomation(rate, channels); };
  DenseParameterProcess<Convolution> bulk = [](Convolution &effect,
                                               const auto &context,
                                               const auto &controls,
                                               size_t cursor) {
    return effect.processAutomated(context, controls.front().values, cursor);
  };
  bindDenseParameters(
      binding,
      {{"mix", &Convolution::setMix, DenseParameterDomain::UnitInterval}},
      std::move(preflight), {}, std::move(bulk));
}
}; // namespace Pedalboard