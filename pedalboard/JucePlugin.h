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

#pragma once

#include "JuceHeader.h"

#include "Plugin.h"

// A macro to make it easier to add simple parameter validation to existing
// JUCE DSP types, without fully wrapping them in JUCE's `AudioParameter`
// machinery.
#define DEFINE_DSP_SETTER_AND_GETTER(type, CamelCaseParameterName,             \
                                     setterValidation)                         \
private:                                                                       \
  type _##CamelCaseParameterName;                                              \
                                                                               \
public:                                                                        \
  type get##CamelCaseParameterName() const {                                   \
    return _##CamelCaseParameterName;                                          \
  }                                                                            \
  void set##CamelCaseParameterName(const type value) {                         \
    {setterValidation};                                                        \
    _##CamelCaseParameterName = value;                                         \
    this->getDSP().set##CamelCaseParameterName(value);                         \
  }

namespace Pedalboard {
/**
 * A template class to adapt an arbitrary juce::dsp block to a Plugin.
 * Could technically be used with any type that provides prepare,
 * process, and reset methods.
 */
template <typename DSPType> class JucePlugin : public Plugin {
public:
  virtual ~JucePlugin() {};

  void prepare(const juce::dsp::ProcessSpec &spec) override {
    if (lastSpec.sampleRate != spec.sampleRate ||
        lastSpec.maximumBlockSize < spec.maximumBlockSize ||
        spec.numChannels != lastSpec.numChannels) {
      dspBlock.prepare(spec);
      lastSpec = spec;
    }
  }

  int process(
      const juce::dsp::ProcessContextReplacing<float> &context) override {
    dspBlock.process(context);
    return context.getOutputBlock().getNumSamples();
  }

  void reset() override { dspBlock.reset(); }

  DSPType &getDSP() { return dspBlock; };
  const DSPType &getDSP() const { return dspBlock; };

protected:
  // Opt in only when DSP has no block-sized scratch and prepare() would reset
  // it. A larger audio block alone then needs no reallocation or
  // reinitialization.
  void prepareWithoutBlockSizeReset(const juce::dsp::ProcessSpec &spec) {
    if (lastSpec.sampleRate != spec.sampleRate ||
        spec.numChannels != lastSpec.numChannels)
      dspBlock.prepare(spec);
    lastSpec = spec;
  }

private:
  DSPType dspBlock;
};
// For DSPs whose private scratch cannot grow without resetting their histories.
// Prepare a fixed capacity on format changes; larger calls reuse bounded spans.
template <typename DSPType, typename SampleType, unsigned int Capacity = 1024>
class FixedBlockJucePlugin : public JucePlugin<DSPType> {
  static_assert(Capacity > 0, "DSP scratch capacity must be positive.");

public:
  void prepare(const juce::dsp::ProcessSpec &spec) override {
    if (this->lastSpec.sampleRate != spec.sampleRate ||
        this->lastSpec.numChannels != spec.numChannels) {
      auto boundedSpec = spec;
      boundedSpec.maximumBlockSize = Capacity;
      JucePlugin<DSPType>::prepare(boundedSpec);
    }
    this->lastSpec = spec;
  }

  int process(
      const juce::dsp::ProcessContextReplacing<SampleType> &context) override {
    auto block = context.getOutputBlock();
    for (size_t start = 0; start < block.getNumSamples(); start += Capacity) {
      auto span = block.getSubBlock(
          start, std::min((size_t)Capacity, block.getNumSamples() - start));
      juce::dsp::ProcessContextReplacing<SampleType> boundedContext(span);
      boundedContext.isBypassed = context.isBypassed;
      this->getDSP().process(boundedContext);
    }
    return (int)block.getNumSamples();
  }
};
} // namespace Pedalboard
