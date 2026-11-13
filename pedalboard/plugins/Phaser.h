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
class Phaser
    : public FixedBlockJucePlugin<juce::dsp::Phaser<SampleType>, SampleType> {
public:
  // JUCE prepare changes its rate without remapping the cached centre.
  // The dense path refreshes it after prepare, including omitted scalar values.
  void prepareAutomationCentre() {
    this->getDSP().setCentreFrequency(getCentreFrequency());
  }

  void validateAutomation(const std::vector<DenseParameterControl> &controls,
                          double sampleRate) const {
    // The LFO runs at rate/4; bound its phase wrapping and 50ms int ramps.
    const float oscillatorRate = (float)(sampleRate / 4.0);
    if (!std::isfinite(sampleRate) || sampleRate < 400 ||
        !std::isfinite(oscillatorRate) ||
        std::floor(sampleRate * 0.05) > std::numeric_limits<int>::max() ||
        std::floor((double)oscillatorRate * 0.05) >
            std::numeric_limits<int>::max())
      throw std::invalid_argument(
          "Phaser automation requires a finite sample rate >= 400Hz with "
          "representable native 50ms ramps.");
    float rate = getRate(), depth = getDepth(), centre = getCentreFrequency(),
          feedback = getFeedback(), mix = getMix();
    for (const auto &control : controls) {
      const float v = control.values.front();
      if (control.name == "rate_hz")
        rate = v;
      else if (control.name == "depth")
        depth = v;
      else if (control.name == "centre_frequency_hz")
        centre = v;
      else if (control.name == "feedback")
        feedback = v;
      else if (control.name == "mix")
        mix = v;
    }
    if (!validDenseParameter(rate, rate, DenseParameterDomain::ModulationRate,
                             sampleRate) ||
        !validDenseParameter(depth, depth, DenseParameterDomain::UnitInterval,
                             sampleRate) ||
        !validDenseParameter(centre, centre,
                             DenseParameterDomain::PhaserFrequency,
                             sampleRate) ||
        !validDenseParameter(feedback, feedback,
                             DenseParameterDomain::SignedUnitInterval,
                             sampleRate) ||
        !validDenseParameter(mix, mix, DenseParameterDomain::UnitInterval,
                             sampleRate))
      throw std::invalid_argument("Invalid effective Phaser controls.");
  }

  DEFINE_DSP_SETTER_AND_GETTER(SampleType, Rate, {});
  DEFINE_DSP_SETTER_AND_GETTER(SampleType, Depth, {});
  DEFINE_DSP_SETTER_AND_GETTER(SampleType, CentreFrequency, {});
  DEFINE_DSP_SETTER_AND_GETTER(SampleType, Feedback, {});
  DEFINE_DSP_SETTER_AND_GETTER(SampleType, Mix, {});
};

inline void init_phaser(py::module &m) {

  auto binding =
      py::class_<Phaser<float>, Plugin, std::shared_ptr<Phaser<float>>>(
          m, "Phaser",
          "A 6 stage phaser that modulates first order all-pass filters to "
          "create "
          "sweeping notches in the magnitude frequency response. This audio "
          "effect "
          "can be controlled with standard phaser parameters: the speed and "
          "depth "
          "of the LFO controlling the frequency response, a mix control, a "
          "feedback control, and the centre frequency of the modulation.")
          .def(py::init([](float rateHz, float depth, float centreFrequency,
                           float feedback, float mix) {
                 auto plugin = std::make_unique<Phaser<float>>();
                 plugin->setRate(rateHz);
                 plugin->setDepth(depth);
                 plugin->setCentreFrequency(centreFrequency);
                 plugin->setFeedback(feedback);
                 plugin->setMix(mix);
                 return plugin;
               }),
               py::arg("rate_hz") = 1.0, py::arg("depth") = 0.5,
               py::arg("centre_frequency_hz") = 1300.0,
               py::arg("feedback") = 0.0, py::arg("mix") = 0.5)
          .def("__repr__",
               [](const Phaser<float> &plugin) {
                 std::ostringstream ss;
                 ss << "<pedalboard.Phaser";
                 ss << " rate_hz=" << plugin.getRate();
                 ss << " depth=" << plugin.getDepth();
                 ss << " centre_frequency_hz=" << plugin.getCentreFrequency();
                 ss << " feedback=" << plugin.getFeedback();
                 ss << " mix=" << plugin.getMix();
                 ss << " at " << &plugin;
                 ss << ">";
                 return ss.str();
               })
          .def_property("rate_hz", &Phaser<float>::getRate,
                        &Phaser<float>::setRate)
          .def_property("depth", &Phaser<float>::getDepth,
                        &Phaser<float>::setDepth)
          .def_property("centre_frequency_hz",
                        &Phaser<float>::getCentreFrequency,
                        &Phaser<float>::setCentreFrequency)
          .def_property("feedback", &Phaser<float>::getFeedback,
                        &Phaser<float>::setFeedback)
          .def_property("mix", &Phaser<float>::getMix, &Phaser<float>::setMix);
  DenseParameterPreflight<Phaser<float>> preflight =
      [](const Phaser<float> &effect,
         const std::vector<DenseParameterControl> &controls, double rate,
         unsigned int) { effect.validateAutomation(controls, rate); };
  bindDenseParameters(
      binding,
      {{"rate_hz", &Phaser<float>::setRate,
        DenseParameterDomain::ModulationRate},
       {"depth", &Phaser<float>::setDepth, DenseParameterDomain::UnitInterval},
       {"centre_frequency_hz", &Phaser<float>::setCentreFrequency,
        DenseParameterDomain::PhaserFrequency},
       {"feedback", &Phaser<float>::setFeedback,
        DenseParameterDomain::SignedUnitInterval},
       {"mix", &Phaser<float>::setMix, DenseParameterDomain::UnitInterval}},
      std::move(preflight),
      DenseParameterPrepared<Phaser<float>>(
          [](Phaser<float> &effect) { effect.prepareAutomationCentre(); }));
}
}; // namespace Pedalboard