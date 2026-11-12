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

/**
 * Given a cutoff frequency and sample rate, clamp the cutoff frequency to
 * [e, sr / 2 - e] for some small epsilon, to ensure that filter frequency
 * response is stable.
 */
inline float clampCutoffFrequency(float cutoffFrequencyHz, float sampleRate) {
  return juce::jlimit(1e-2f, (sampleRate / 2) - 1e2f, cutoffFrequencyHz);
}

/**
 * A base class for all IIR filter classes.
 */
template <typename SampleType>
class IIRFilter : public JucePlugin<juce::dsp::ProcessorDuplicator<
                      juce::dsp::IIR::Filter<SampleType>,
                      juce::dsp::IIR::Coefficients<SampleType>>> {
public:
  void setCutoffFrequencyHz(float f) {
    if (f <= 0)
      throw std::domain_error("Cutoff frequency must be greater than 0Hz.");
    cutoffFrequencyHz = f;
  }
  float getCutoffFrequencyHz() const noexcept { return cutoffFrequencyHz; }

  void setQ(float q) {
    if (q <= 0)
      throw std::domain_error("Q value must be greater than 0.");
    Q = q;
  }
  float getQ() const noexcept { return Q; }

  void setGainDecibels(float dB) {
    gainFactor = juce::Decibels::decibelsToGain<SampleType>(dB);
  }
  float getGainDecibels() const noexcept {
    return juce::Decibels::gainToDecibels<SampleType>(gainFactor);
  }

  void setAutomatedCutoffFrequencyHz(float value) {
    setCutoffFrequencyHz(value);
    coefficientsDirty = true;
  }
  void setAutomatedQ(float value) {
    setQ(value);
    coefficientsDirty = true;
  }
  void setAutomatedGainDecibels(float value) {
    setGainDecibels(value);
    coefficientsDirty = true;
  }

  void prepare(const juce::dsp::ProcessSpec &spec) override {
    updateCoefficients(spec.sampleRate);
    this->prepareWithoutBlockSizeReset(spec);
    coefficientsDirty = false;
  }

  int process(
      const juce::dsp::ProcessContextReplacing<float> &context) override {
    // All same-frame controls are stored before one coefficient refresh. The
    // rate is already established by prepare(), including format changes.
    if (coefficientsDirty) {
      updateCoefficients(this->lastSpec.sampleRate);
      coefficientsDirty = false;
    }
    this->getDSP().process(context);
    return (int)context.getOutputBlock().getNumSamples();
  }

  void validateAutomation(const std::vector<DenseParameterControl> &controls,
                          double sampleRate) const {
    if (!std::isfinite((float)sampleRate) ||
        !((float)sampleRate * 0.5f - 100.0f > 0.01f))
      throw std::invalid_argument(
          "EQ curves require a finite audio sample rate "
          "above 200.02 Hz.");
    const std::vector<float> *frequencies = nullptr, *gains = nullptr,
                             *qualities = nullptr;
    for (const auto &control : controls) {
      if (control.name == "cutoff_frequency_hz")
        frequencies = &control.values;
      else if (control.name == "gain_db")
        gains = &control.values;
      else if (control.name == "q")
        qualities = &control.values;
    }
    float previousFrequency = 0, previousGain = 0, previousQ = 0;
    for (size_t i = 0; i < controls.front().values.size(); ++i) {
      const float frequency =
          frequencies ? (*frequencies)[i] : cutoffFrequencyHz;
      const float gain =
          gains ? juce::Decibels::decibelsToGain((*gains)[i]) : gainFactor;
      const float quality = qualities ? (*qualities)[i] : Q;
      if (!(std::isfinite(frequency) && frequency > 0 && std::isfinite(gain) &&
            gain > 0 && std::isfinite(quality) && quality > 0))
        throw std::invalid_argument("EQ controls must form a finite positive "
                                    "frequency/Q/gain combination.");
      if (i && frequency == previousFrequency && gain == previousGain &&
          quality == previousQ)
        continue;
      const auto values = coefficientsFor(sampleRate, frequency, quality, gain);
      if (!(values[3] > 0) || !std::isfinite(values[3]))
        throw std::invalid_argument("Invalid EQ coefficient denominator.");
      const float inverse = 1.0f / values[3];
      std::array<float, 6> normalized;
      for (size_t j = 0; j < values.size(); ++j) {
        normalized[j] = values[j] * inverse;
        if (!std::isfinite(normalized[j]))
          throw std::invalid_argument(
              "EQ curves produce nonfinite coefficients.");
      }
      // Jury conditions for z^2 + a1*z + a2, evaluated in double on the
      // actual normalized float32 coefficients to avoid cancellation here.
      const double a1 = normalized[4], a2 = normalized[5];
      if (!(std::abs(a2) < 1 && 1.0 + a1 + a2 > 0 && 1.0 - a1 + a2 > 0))
        throw std::invalid_argument("EQ curves produce unstable or marginal "
                                    "float32 coefficients.");
      previousFrequency = frequency;
      previousGain = gain;
      previousQ = quality;
    }
  }

protected:
  virtual std::array<SampleType, 6> makeCoefficients(double sampleRate,
                                                     SampleType frequency,
                                                     SampleType quality,
                                                     SampleType gain) const = 0;

  float cutoffFrequencyHz;
  float Q;
  float gainFactor;

private:
  std::array<SampleType, 6> coefficientsFor(double sampleRate, float frequency,
                                            float quality, float gain) const {
    return makeCoefficients(
        sampleRate, clampCutoffFrequency(frequency, sampleRate), quality, gain);
  }
  void updateCoefficients(double sampleRate) {
    *this->getDSP().state =
        coefficientsFor(sampleRate, cutoffFrequencyHz, Q, gainFactor);
  }
  bool coefficientsDirty = false;
};

template <typename SampleType>
class HighShelfFilter : public IIRFilter<SampleType> {
protected:
  std::array<SampleType, 6> makeCoefficients(double rate, SampleType frequency,
                                             SampleType quality,
                                             SampleType gain) const override {
    return juce::dsp::IIR::ArrayCoefficients<SampleType>::makeHighShelf(
        rate, frequency, quality, gain);
  }
};

template <typename SampleType>
class LowShelfFilter : public IIRFilter<SampleType> {
protected:
  std::array<SampleType, 6> makeCoefficients(double rate, SampleType frequency,
                                             SampleType quality,
                                             SampleType gain) const override {
    return juce::dsp::IIR::ArrayCoefficients<SampleType>::makeLowShelf(
        rate, frequency, quality, gain);
  }
};

template <typename SampleType> class PeakFilter : public IIRFilter<SampleType> {
protected:
  std::array<SampleType, 6> makeCoefficients(double rate, SampleType frequency,
                                             SampleType quality,
                                             SampleType gain) const override {
    return juce::dsp::IIR::ArrayCoefficients<SampleType>::makePeakFilter(
        rate, frequency, quality, gain);
  }
};

template <typename Effect, typename... Options>
void bindEQAutomation(py::class_<Effect, Options...> &binding) {
  DenseParameterPreflight<Effect> preflight =
      [](const Effect &effect,
         const std::vector<DenseParameterControl> &controls, double rate,
         unsigned int) { effect.validateAutomation(controls, rate); };
  bindDenseParameters(
      binding,
      {{"cutoff_frequency_hz", &Effect::setAutomatedCutoffFrequencyHz,
        DenseParameterDomain::Positive},
       {"gain_db", &Effect::setAutomatedGainDecibels,
        DenseParameterDomain::EQGain},
       {"q", &Effect::setAutomatedQ, DenseParameterDomain::Positive}},
      std::move(preflight));
}

inline void init_iir_filters(py::module &m) {
  py::class_<IIRFilter<float>, Plugin, std::shared_ptr<IIRFilter<float>>>(
      m, "IIRFilter",
      "An abstract class that implements various kinds of infinite impulse "
      "response (IIR) filter designs. This should not be used directly; use "
      ":class:`HighShelfFilter`, :class:`LowShelfFilter`, or "
      ":class:`PeakFilter` directly instead.")
      .def(py::init([]() {
        throw std::runtime_error(
            "IIRFilter is not designed to be instantiated directly: "
            "use HighShelfFilter, LowShelfFilter, or PeakFilter instead.");
        return nullptr;
      }));

  auto highShelf =
      py::class_<HighShelfFilter<float>, IIRFilter<float>,
                 std::shared_ptr<HighShelfFilter<float>>>(
          m, "HighShelfFilter",
          "A high shelf filter plugin with variable Q and gain, as would be "
          "used "
          "in an equalizer. Frequencies above the cutoff frequency will be "
          "boosted "
          "(or cut) by the provided gain (in decibels).")
          .def(py::init([](float cutoffFrequencyHz, float gaindB, float Q) {
                 auto plugin = std::make_unique<HighShelfFilter<float>>();
                 plugin->setCutoffFrequencyHz(cutoffFrequencyHz);
                 plugin->setGainDecibels(gaindB);
                 plugin->setQ(Q);
                 return plugin;
               }),
               py::arg("cutoff_frequency_hz") = 440, py::arg("gain_db") = 0.0,
               py::arg("q") = (juce::MathConstants<float>::sqrt2 / 2.0))
          .def("__repr__",
               [](const HighShelfFilter<float> &plugin) {
                 std::ostringstream ss;
                 ss << "<pedalboard.HighShelfFilter";
                 ss << " cutoff_frequency_hz=" << plugin.getCutoffFrequencyHz();
                 ss << " gain_db=" << plugin.getGainDecibels();
                 ss << " q=" << plugin.getQ();
                 ss << " at " << &plugin;
                 ss << ">";
                 return ss.str();
               })
          .def_property("cutoff_frequency_hz",
                        &HighShelfFilter<float>::getCutoffFrequencyHz,
                        &HighShelfFilter<float>::setCutoffFrequencyHz)
          .def_property("gain_db", &HighShelfFilter<float>::getGainDecibels,
                        &HighShelfFilter<float>::setGainDecibels)
          .def_property("q", &HighShelfFilter<float>::getQ,
                        &HighShelfFilter<float>::setQ);
  bindEQAutomation(highShelf);

  auto lowShelf =
      py::class_<LowShelfFilter<float>, IIRFilter<float>,
                 std::shared_ptr<LowShelfFilter<float>>>(
          m, "LowShelfFilter",
          "A low shelf filter with variable Q and gain, as would be used in an "
          "equalizer. Frequencies below the cutoff frequency will be boosted "
          "(or "
          "cut) by the provided gain value.")
          .def(py::init([](float cutoffFrequencyHz, float gaindB, float Q) {
                 auto plugin = std::make_unique<LowShelfFilter<float>>();
                 plugin->setCutoffFrequencyHz(cutoffFrequencyHz);
                 plugin->setGainDecibels(gaindB);
                 plugin->setQ(Q);
                 return plugin;
               }),
               py::arg("cutoff_frequency_hz") = 440, py::arg("gain_db") = 0.0,
               py::arg("q") = (juce::MathConstants<float>::sqrt2 / 2.0))
          .def("__repr__",
               [](const LowShelfFilter<float> &plugin) {
                 std::ostringstream ss;
                 ss << "<pedalboard.LowShelfFilter";
                 ss << " cutoff_frequency_hz=" << plugin.getCutoffFrequencyHz();
                 ss << " gain_db=" << plugin.getGainDecibels();
                 ss << " q=" << plugin.getQ();
                 ss << " at " << &plugin;
                 ss << ">";
                 return ss.str();
               })
          .def_property("cutoff_frequency_hz",
                        &LowShelfFilter<float>::getCutoffFrequencyHz,
                        &LowShelfFilter<float>::setCutoffFrequencyHz)
          .def_property("gain_db", &LowShelfFilter<float>::getGainDecibels,
                        &LowShelfFilter<float>::setGainDecibels)
          .def_property("q", &LowShelfFilter<float>::getQ,
                        &LowShelfFilter<float>::setQ);
  bindEQAutomation(lowShelf);

  auto peak =
      py::class_<PeakFilter<float>, IIRFilter<float>,
                 std::shared_ptr<PeakFilter<float>>>(
          m, "PeakFilter",
          "A peak (or notch) filter with variable Q and gain, as would be used "
          "in "
          "an equalizer. Frequencies around the cutoff frequency will be "
          "boosted "
          "(or cut) by the provided gain value.")
          .def(py::init([](float cutoffFrequencyHz, float gaindB, float Q) {
                 auto plugin = std::make_unique<PeakFilter<float>>();
                 plugin->setCutoffFrequencyHz(cutoffFrequencyHz);
                 plugin->setGainDecibels(gaindB);
                 plugin->setQ(Q);
                 return plugin;
               }),
               py::arg("cutoff_frequency_hz") = 440, py::arg("gain_db") = 0.0,
               py::arg("q") = (juce::MathConstants<float>::sqrt2 / 2.0))
          .def("__repr__",
               [](const PeakFilter<float> &plugin) {
                 std::ostringstream ss;
                 ss << "<pedalboard.PeakFilter";
                 ss << " cutoff_frequency_hz=" << plugin.getCutoffFrequencyHz();
                 ss << " gain_db=" << plugin.getGainDecibels();
                 ss << " q=" << plugin.getQ();
                 ss << " at " << &plugin;
                 ss << ">";
                 return ss.str();
               })
          .def_property("cutoff_frequency_hz",
                        &PeakFilter<float>::getCutoffFrequencyHz,
                        &PeakFilter<float>::setCutoffFrequencyHz)
          .def_property("gain_db", &PeakFilter<float>::getGainDecibels,
                        &PeakFilter<float>::setGainDecibels)
          .def_property("q", &PeakFilter<float>::getQ,
                        &PeakFilter<float>::setQ);
  bindEQAutomation(peak);
}
}; // namespace Pedalboard
