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

namespace Pedalboard {
template <typename SampleType>
class HighpassFilter : public JucePlugin<juce::dsp::ProcessorDuplicator<
                           juce::dsp::IIR::Filter<SampleType>,
                           juce::dsp::IIR::Coefficients<SampleType>>> {
public:
  void setCutoffFrequencyHz(float f) noexcept { cutoffFrequencyHz = f; }
  float getCutoffFrequencyHz() const noexcept { return cutoffFrequencyHz; }

  // The scalar setter keeps its existing deferred-prepare behavior.
  void setAutomatedCutoffFrequencyHz(float value) {
    cutoffFrequencyHz = value;
    coefficientsDirty = true;
  }

  void prepare(const juce::dsp::ProcessSpec &spec) override {
    updateCoefficients(spec.sampleRate);
    this->prepareWithoutBlockSizeReset(spec);
    coefficientsDirty = false;
  }

  int process(
      const juce::dsp::ProcessContextReplacing<float> &context) override {
    // First targets can arrive before prepare() on a rate change. Refresh only
    // here, after preparation has established the current processing rate.
    if (coefficientsDirty) {
      updateCoefficients(this->lastSpec.sampleRate);
      coefficientsDirty = false;
    }
    this->getDSP().process(context);
    return (int)context.getOutputBlock().getNumSamples();
  }

private:
  void updateCoefficients(double sampleRate) {
    // JUCE's array form uses the same arithmetic without creating a new
    // ref-counted coefficient object for every changing control sample.
    *this->getDSP().state =
        juce::dsp::IIR::ArrayCoefficients<SampleType>::makeFirstOrderHighPass(
            sampleRate, cutoffFrequencyHz);
  }

  float cutoffFrequencyHz;
  bool coefficientsDirty = false;
};

inline void init_highpass(py::module &m) {
  auto binding =
      py::class_<HighpassFilter<float>, Plugin,
                 std::shared_ptr<HighpassFilter<float>>>(
          m, "HighpassFilter",
          "Apply a first-order high-pass filter with a roll-off of 6dB/octave. "
          "The cutoff frequency will be attenuated by -3dB (i.e.: "
          R"(:math:`\\frac{1}{\\sqrt{2}}` as loud, expressed as a gain factor))"
          " and lower frequencies will be attenuated by a further 6dB per "
          "octave.)")
          .def(py::init([](float cutoff_frequency_hz) {
                 auto plugin = std::make_unique<HighpassFilter<float>>();
                 plugin->setCutoffFrequencyHz(cutoff_frequency_hz);
                 return plugin;
               }),
               py::arg("cutoff_frequency_hz") = 50)
          .def("__repr__",
               [](const HighpassFilter<float> &plugin) {
                 std::ostringstream ss;
                 ss << "<pedalboard.Highpass";
                 ss << " cutoff_frequency_hz=" << plugin.getCutoffFrequencyHz();
                 ss << " at " << &plugin;
                 ss << ">";
                 return ss.str();
               })
          .def_property("cutoff_frequency_hz",
                        &HighpassFilter<float>::getCutoffFrequencyHz,
                        &HighpassFilter<float>::setCutoffFrequencyHz);
  bindDenseParameters(binding,
                      {{"cutoff_frequency_hz",
                        &HighpassFilter<float>::setAutomatedCutoffFrequencyHz,
                        DenseParameterDomain::CutoffFrequency}});
}
}; // namespace Pedalboard
