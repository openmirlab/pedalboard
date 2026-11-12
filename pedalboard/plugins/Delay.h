/*
 * pedalboard
 * Copyright 2022 Spotify AB
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

#include "../JucePlugin.h"
#include "../parameter_automation.h"

namespace Pedalboard {
template <typename SampleType>
class Delay : public JucePlugin<juce::dsp::DelayLine<
                  SampleType, juce::dsp::DelayLineInterpolationTypes::None>> {
public:
  void validateAutomation(const std::vector<DenseParameterControl> &controls,
                          double rate) const {
    // Both ring index sums (position + size - 1 and read + delay) use int.
    if (!std::isfinite(rate) || rate <= 0 ||
        MAXIMUM_DELAY_TIME_SECONDS * rate >
            (std::numeric_limits<int>::max() - 1) / 2)
      throw std::invalid_argument("Delay automation sample rate exceeds safe "
                                  "native ring-buffer arithmetic.");
    auto seconds = getDelaySeconds(), feedback = getFeedback(), mix = getMix();
    for (const auto &control : controls) {
      const float v = control.values.front();
      if (control.name == "delay_seconds")
        seconds = v;
      else if (control.name == "feedback")
        feedback = v;
      else if (control.name == "mix")
        mix = v;
    }
    if (!std::isfinite(seconds) || seconds < 0 ||
        seconds > MAXIMUM_DELAY_TIME_SECONDS || !std::isfinite(feedback) ||
        feedback < 0 || feedback > 1 || !std::isfinite(mix) || mix < 0 ||
        mix > 1)
      throw std::invalid_argument(
          "Delay controls must be finite: delay_seconds in [0, 30], "
          "feedback and mix in [0, 1].");
  }

  SampleType getDelaySeconds() const { return delaySeconds; }
  void setDelaySeconds(const SampleType value) {
    if (value < 0.0 || value > MAXIMUM_DELAY_TIME_SECONDS) {
      throw std::range_error("Delay (in seconds) must be between 0.0s and " +
                             std::to_string(MAXIMUM_DELAY_TIME_SECONDS) + "s.");
    }
    delaySeconds = value;
  };
  SampleType getFeedback() const { return feedback; }
  void setFeedback(const SampleType value) {
    if (value < 0.0 || value > 1.0) {
      throw std::range_error("Feedback must be between 0.0 and 1.0.");
    }
    feedback = value;
  };
  SampleType getMix() const { return mix; }
  void setMix(const SampleType value) {
    if (value < 0.0 || value > 1.0) {
      throw std::range_error("Mix must be between 0.0 and 1.0.");
    }
    mix = value;
  };

  virtual void prepare(const juce::dsp::ProcessSpec &spec) override {
    if (this->lastSpec.sampleRate != spec.sampleRate ||
        spec.numChannels != this->lastSpec.numChannels) {
      this->getDSP().setMaximumDelayInSamples(
          (int)(MAXIMUM_DELAY_TIME_SECONDS * spec.sampleRate));
      this->getDSP().prepare(spec);
    }
    this->lastSpec = spec;

    this->getDSP().setDelay((int)(delaySeconds * spec.sampleRate));
  }

  virtual void reset() override {
    JucePlugin<juce::dsp::DelayLine<
        SampleType, juce::dsp::DelayLineInterpolationTypes::None>>::reset();
    this->getDSP().reset();
  }

  virtual int process(
      const juce::dsp::ProcessContextReplacing<SampleType> &context) override {
    // TODO: More advanced mixing rules than "linear?"
    SampleType dryVolume = 1.0f - getMix();
    SampleType wetVolume = getMix();

    if (delaySeconds == 0.0f) {
      // Special case where DelayLine doesn't do anything for us.
      // Regardless of the mix or feedback parameters, the input will sound
      // identical.
      return context.getInputBlock().getNumSamples();
    }

    this->getDSP().setDelay((int)(delaySeconds * this->lastSpec.sampleRate));

    // Pass samples through the delay line with feedback:
    for (size_t c = 0; c < context.getInputBlock().getNumChannels(); c++) {
      jassert(context.getInputBlock().getChannelPointer(c) ==
              context.getOutputBlock().getChannelPointer(c));
      SampleType *channelBuffer = context.getOutputBlock().getChannelPointer(c);

      for (size_t i = 0; i < context.getInputBlock().getNumSamples(); i++) {
        SampleType delayOutput = this->getDSP().popSample(c);
        this->getDSP().pushSample(c, channelBuffer[i] +
                                         (getFeedback() * delayOutput));
        channelBuffer[i] =
            (channelBuffer[i] * dryVolume) + (wetVolume * delayOutput);
      }
    }
    return context.getInputBlock().getNumSamples();
  }

private:
  SampleType delaySeconds = 1.0f;
  SampleType feedback = 0.0f;
  SampleType mix = 1.0f;
  static constexpr int MAXIMUM_DELAY_TIME_SECONDS = 30;
};

inline void init_delay(py::module &m) {
  auto binding =
      py::class_<Delay<float>, Plugin, std::shared_ptr<Delay<float>>>(
          m, "Delay",
          "A digital delay plugin with controllable delay time, feedback "
          "percentage, and dry/wet mix.")
          .def(py::init([](float delaySeconds, float feedback, float mix) {
                 auto delay = std::make_unique<Delay<float>>();
                 delay->setDelaySeconds(delaySeconds);
                 delay->setFeedback(feedback);
                 delay->setMix(mix);
                 return delay;
               }),
               py::arg("delay_seconds") = 0.5, py::arg("feedback") = 0.0,
               py::arg("mix") = 0.5)
          .def("__repr__",
               [](const Delay<float> &plugin) {
                 std::ostringstream ss;
                 ss << "<pedalboard.Delay";
                 ss << " delay_seconds=" << plugin.getDelaySeconds();
                 ss << " feedback=" << plugin.getFeedback();
                 ss << " mix=" << plugin.getMix();
                 ss << " at " << &plugin;
                 ss << ">";
                 return ss.str();
               })
          .def_property("delay_seconds", &Delay<float>::getDelaySeconds,
                        &Delay<float>::setDelaySeconds)
          .def_property("feedback", &Delay<float>::getFeedback,
                        &Delay<float>::setFeedback)
          .def_property("mix", &Delay<float>::getMix, &Delay<float>::setMix);
  DenseParameterPreflight<Delay<float>> preflight =
      [](const Delay<float> &effect,
         const std::vector<DenseParameterControl> &controls, double rate,
         unsigned int) { effect.validateAutomation(controls, rate); };
  bindDenseParameters(
      binding,
      {{"delay_seconds", &Delay<float>::setDelaySeconds,
        DenseParameterDomain::DelaySeconds},
       {"feedback", &Delay<float>::setFeedback,
        DenseParameterDomain::UnitInterval},
       {"mix", &Delay<float>::setMix, DenseParameterDomain::UnitInterval}},
      std::move(preflight));
}
}; // namespace Pedalboard
