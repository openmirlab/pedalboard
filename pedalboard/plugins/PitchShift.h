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

#include "../RubberbandPlugin.h"
#include "../parameter_automation.h"
#include "../plugin_templates/PrimeWithSilence.h"
#include <atomic>
#include <cstring>

namespace Pedalboard {

/*
 * Modifies pitch of an audio without affecting duration.
 */
class PitchShift : public PrimeWithSilence<RubberbandPlugin> {
private:
  std::atomic<double> _semitones{0.0};
  unsigned int capacity_ = 0;
  std::vector<double> history_, targets_;
  size_t historyWrite_ = 0, historyFrames_ = 0;
  bool automationActive_ = false, fineScheduling_ = false;
  mutable bool incomingVaries_ = false, automationPending_ = false;
  void recordHistory(double value, size_t count) {
    if (count >= history_.size()) {
      std::fill(history_.begin(), history_.end(), value);
      historyWrite_ = (historyWrite_ + count) % history_.size();
      historyFrames_ = history_.size();
    } else
      for (size_t i = 0; i < count; ++i) {
        history_[historyWrite_] = value;
        historyWrite_ = (historyWrite_ + 1) % history_.size();
        historyFrames_ = std::min(historyFrames_ + 1, history_.size());
      }
  }
  int processBlocks(const juce::dsp::ProcessContextReplacing<float> &context,
                    const std::vector<DenseParameterControl> *controls,
                    size_t cursor) {
    auto block = context.getOutputBlock();
    const size_t count = block.getNumSamples();
    const double scalar = getSemitones();
    size_t returned = 0;
    for (size_t start = 0; start < count; start += capacity_) {
      const size_t length = std::min((size_t)capacity_, count - start);
      if (automationActive_)
        for (size_t i = 0; i < length; ++i) {
          double source =
              controls
                  ? controls->front()
                        .values[std::min(cursor + start + i,
                                         controls->front().values.size() - 1)]
                  : scalar;
          targets_[i] = history_[historyWrite_];
          history_[historyWrite_] = source;
          historyWrite_ = (historyWrite_ + 1) % history_.size();
          historyFrames_ = std::min(historyFrames_ + 1, history_.size());
        }
      for (size_t offset = 0; offset < length;) {
        size_t end = length;
        if (automationActive_) {
          end = offset + 1;
          while (!fineScheduling_ && end < length &&
                 targets_[end] == targets_[offset])
            ++end;
          getNestedPlugin().getStretcher().setPitchScale(
              std::pow(2.0, double(targets_[offset]) / 12.0));
        }
        auto span = block.getSubBlock(start + offset, end - offset);
        juce::dsp::ProcessContextReplacing<float> subContext(span);
        int valid = PrimeWithSilence<RubberbandPlugin>::process(subContext);
        if (valid < 0 || valid > (int)(end - offset))
          throw std::runtime_error("Invalid PitchShift frame count");
        // process.h requires all valid output at the right of the outer block.
        // Compact only processed regions first, then move the packed output
        // once.
        for (size_t ch = 0; ch < block.getNumChannels(); ++ch)
          std::memmove(block.getChannelPointer(ch) + returned,
                       block.getChannelPointer(ch) + start + end - valid,
                       valid * sizeof(float));
        returned += valid;
        offset = end;
      }
    }
    if (!automationActive_)
      recordHistory(scalar, count);
    for (size_t ch = 0; ch < block.getNumChannels(); ++ch)
      std::memmove(block.getChannelPointer(ch) + count - returned,
                   block.getChannelPointer(ch), returned * sizeof(float));
    if (controls && count)
      _semitones = controls->front().values[std::min(
          cursor + count - 1, controls->front().values.size() - 1)];
    return (int)returned;
  }

  // Allow pitch shifting by up to 6 octaves up or down:
  static constexpr int MIN_SEMITONES = -6 * 12;
  static constexpr int MAX_SEMITONES = 6 * 12;

  double getScaleFactor() { return pow(2, (getSemitones() / 12)); }

public:
  void setSemitones(double semitones) {
    if (semitones < MIN_SEMITONES || semitones > MAX_SEMITONES) {
      throw std::range_error("Semitones of pitch must be a value between " +
                             std::to_string(MIN_SEMITONES) + "st and " +
                             std::to_string(MAX_SEMITONES) + "st.");
    }

    _semitones = semitones;
  }

  double getSemitones() const {
    return _semitones.load(std::memory_order_relaxed);
  }

  void prepare(const juce::dsp::ProcessSpec &spec) override final {
    const bool changed = !capacity_ || lastSpec.sampleRate != spec.sampleRate ||
                         lastSpec.numChannels != spec.numChannels;
    if (changed) {
      capacity_ = std::max(1u, std::min(8192u, spec.maximumBlockSize));
      history_.assign(std::max(1, (int)spec.sampleRate), getSemitones());
      targets_.resize(capacity_);
      historyWrite_ = historyFrames_ = 0;
      automationActive_ = false;
      fineScheduling_ = false;
    }
    auto boundedSpec = spec;
    boundedSpec.maximumBlockSize = capacity_;
    setSilenceLengthSamples(spec.sampleRate);
    PrimeWithSilence<RubberbandPlugin>::prepare(boundedSpec);
    if (!automationActive_) {
      const double enginePitch = automationPending_ && historyFrames_
                                     ? history_[historyWrite_]
                                     : getSemitones();
      getNestedPlugin().getStretcher().setPitchScale(
          std::pow(2.0, enginePitch / 12.0));
    }
  }
  void reset() override {
    PrimeWithSilence<RubberbandPlugin>::reset();
    getNestedPlugin().clearPreparation();
    automationActive_ = false;
    fineScheduling_ = false;
    historyWrite_ = historyFrames_ = 0;
    std::fill(history_.begin(), history_.end(), getSemitones());
  }
  void validateAutomation(const std::vector<DenseParameterControl> &controls,
                          double rate) const {
    if (!std::isfinite(rate) || rate < 1 || rate > INT_MAX / 128 ||
        controls.front().values.size() + rate * 128 >= INT_MAX)
      throw std::range_error("Unsafe PitchShift sample rate or frame count");
  }
  void requestAutomation(const std::vector<DenseParameterControl> &controls) {
    const auto &values = controls.front().values;
    automationPending_ = true;
    incomingVaries_ =
        std::any_of(values.begin(), values.end(),
                    [&](float value) { return value != values.front(); });
  }
  void enableAutomationTimeline() {
    automationPending_ = false;
    fineScheduling_ =
        fineScheduling_ || incomingVaries_ ||
        std::any_of(history_.begin(), history_.end(),
                    [this](double value) { return value != getSemitones(); });
    automationActive_ = true;
    if (!historyFrames_)
      std::fill(history_.begin(), history_.end(), getSemitones());
  }
  int process(
      const juce::dsp::ProcessContextReplacing<float> &context) override {
    return processBlocks(context, nullptr, 0);
  }
  int processAutomated(const juce::dsp::ProcessContextReplacing<float> &context,
                       const std::vector<DenseParameterControl> &controls,
                       size_t cursor) {
    return processBlocks(context, &controls, cursor);
  }
};

inline void init_pitch_shift(py::module &m) {
  auto binding =
      py::class_<PitchShift, Plugin, std::shared_ptr<PitchShift>>(
          m, "PitchShift",
          "A pitch shifting effect that can change the pitch of audio without "
          "affecting its duration.\n\nThis effect uses `Chris Cannam's "
          "wonderful "
          "*Rubber Band* library <https://breakfastquay.com/rubberband/>`_ "
          "audio "
          "stretching library.")
          .def(py::init([](double scale) {
                 auto plugin = std::make_unique<PitchShift>();
                 plugin->setSemitones(scale);
                 return plugin;
               }),
               py::arg("semitones") = 0.0)
          .def("__repr__",
               [](const PitchShift &plugin) {
                 std::ostringstream ss;
                 ss << "<pedalboard.PitchShift";
                 ss << " semitones=" << plugin.getSemitones();
                 ss << " at " << &plugin;
                 ss << ">";
                 return ss.str();
               })
          .def_property("semitones", &PitchShift::getSemitones,
                        &PitchShift::setSemitones);
  bindDenseParameters<PitchShift>(
      binding,
      {{"semitones",
        [](auto &effect, float value) { effect.setSemitones(value); },
        DenseParameterDomain::PitchSemitones}},
      [](const auto &effect, const auto &controls, double rate, unsigned int) {
        effect.validateAutomation(controls, rate);
      },
      [](auto &effect) { effect.enableAutomationTimeline(); },
      [](auto &effect, const auto &context, const auto &controls,
         size_t cursor) {
        return effect.processAutomated(context, controls, cursor);
      },
      DenseParameterLatency::Buffered, {},
      [](auto &effect, const auto &controls) {
        effect.requestAutomation(controls);
      });
}
}; // namespace Pedalboard
