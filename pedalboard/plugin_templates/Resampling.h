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
#pragma once
#include "../JuceHeader.h"
#include "../juce_overrides/juce_FastWindowedSincInterpolators.h"
#include <cstdint>
#include <variant>
namespace Pedalboard {
struct SamplePosition {
  int64_t frame = 0;
  double fraction = 0;
};
/**
 * The various levels of resampler quality available from JUCE.
 * More could be added here, but these should cover the vast
 * majority of use cases.
 */
enum class ResamplingQuality {
  // These types are the legacy JUCE versions, which all
  // introduce aliasing when used for downsampling:
  ZeroOrderHold = 0,
  Linear = 1,
  CatmullRom = 2,
  Lagrange = 3,
  WindowedSinc = 4,
  // These resamplers are faster than the default WindowedSinc
  // counterpart, and all properly handle downsampling without
  // aliasing. They also include significant speedups for the cases
  // in which the speed ratio is a nice fraction like 2/3 or 3/2.
  WindowedSinc256 = 5,
  WindowedSinc128 = 6,
  WindowedSinc64 = 7,
  WindowedSinc32 = 8, // This is the new default as of Pedalboard v0.9.15
  WindowedSinc16 = 9,
  WindowedSinc8 = 10,
};

/**
 * A wrapper class that allows changing the quality of a resampler,
 * as the JUCE GenericInterpolator implementations are each separate classes.
 */
class VariableQualityResampler {
public:
  void setQuality(const ResamplingQuality newQuality) {
    positionInitialized = false;
    positionConsumed = 0;
    switch (newQuality) {
    case ResamplingQuality::ZeroOrderHold:
      interpolator = juce::Interpolators::ZeroOrderHold();
      break;
    case ResamplingQuality::Linear:
      interpolator = juce::Interpolators::Linear();
      break;
    case ResamplingQuality::CatmullRom:
      interpolator = juce::Interpolators::CatmullRom();
      break;
    case ResamplingQuality::Lagrange:
      interpolator = juce::Interpolators::Lagrange();
      break;
    case ResamplingQuality::WindowedSinc:
      interpolator = juce::Interpolators::WindowedSinc();
      break;
    case ResamplingQuality::WindowedSinc256:
      interpolator = juce::FastInterpolators::WindowedSinc256();
      break;
    case ResamplingQuality::WindowedSinc128:
      interpolator = juce::FastInterpolators::WindowedSinc128();
      break;
    case ResamplingQuality::WindowedSinc64:
      interpolator = juce::FastInterpolators::WindowedSinc64();
      break;
    case ResamplingQuality::WindowedSinc32:
      interpolator = juce::FastInterpolators::WindowedSinc32();
      break;
    case ResamplingQuality::WindowedSinc16:
      interpolator = juce::FastInterpolators::WindowedSinc16();
      break;
    case ResamplingQuality::WindowedSinc8:
      interpolator = juce::FastInterpolators::WindowedSinc8();
      break;
    default:
      throw std::domain_error("Unknown resampler quality received!");
    }
  }

  ResamplingQuality getQuality() const {
    return (ResamplingQuality)interpolator.index();
  }

  float getBaseLatency() const {
    // Unfortunately, std::visit cannot be used here due to macOS version
    // issues: https://stackoverflow.com/q/52310835/679081
    if (auto *i =
            std::get_if<juce::Interpolators::ZeroOrderHold>(&interpolator)) {
      return i->getBaseLatency();
    } else if (auto *i =
                   std::get_if<juce::Interpolators::Linear>(&interpolator)) {
      return i->getBaseLatency();
    } else if (auto *i = std::get_if<juce::Interpolators::CatmullRom>(
                   &interpolator)) {
      return i->getBaseLatency();
    } else if (auto *i =
                   std::get_if<juce::Interpolators::Lagrange>(&interpolator)) {
      return i->getBaseLatency();
    } else if (auto *i = std::get_if<juce::Interpolators::WindowedSinc>(
                   &interpolator)) {
      return i->getBaseLatency();
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc256>(
                   &interpolator)) {
      return i->getBaseLatency();
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc128>(
                   &interpolator)) {
      return i->getBaseLatency();
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc64>(
                   &interpolator)) {
      return i->getBaseLatency();
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc32>(
                   &interpolator)) {
      return i->getBaseLatency();
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc16>(
                   &interpolator)) {
      return i->getBaseLatency();
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc8>(
                   &interpolator)) {
      return i->getBaseLatency();
    } else {
      throw std::runtime_error("Unknown resampler quality!");
    }
  }

  void reset() noexcept {
    positionInitialized = false;
    positionConsumed = 0;
    // Unfortunately, std::visit cannot be used here due to macOS version
    // issues: https://stackoverflow.com/q/52310835/679081
    if (auto *i =
            std::get_if<juce::Interpolators::ZeroOrderHold>(&interpolator)) {
      i->reset();
    } else if (auto *i =
                   std::get_if<juce::Interpolators::Linear>(&interpolator)) {
      i->reset();
    } else if (auto *i = std::get_if<juce::Interpolators::CatmullRom>(
                   &interpolator)) {
      i->reset();
    } else if (auto *i =
                   std::get_if<juce::Interpolators::Lagrange>(&interpolator)) {
      i->reset();
    } else if (auto *i = std::get_if<juce::Interpolators::WindowedSinc>(
                   &interpolator)) {
      i->reset();
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc256>(
                   &interpolator)) {
      i->reset();
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc128>(
                   &interpolator)) {
      i->reset();
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc64>(
                   &interpolator)) {
      i->reset();
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc32>(
                   &interpolator)) {
      i->reset();
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc16>(
                   &interpolator)) {
      i->reset();
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc8>(
                   &interpolator)) {
      i->reset();
    } else {
      throw std::runtime_error("Unknown resampler quality!");
    }
  }

  int process(double speedRatio, const float *inputSamples,
              float *outputSamples, int numOutputSamplesToProduce) noexcept {
    // Unfortunately, std::visit cannot be used here due to macOS version
    // issues: https://stackoverflow.com/q/52310835/679081
    if (auto *i =
            std::get_if<juce::Interpolators::ZeroOrderHold>(&interpolator)) {
      return i->process(speedRatio, inputSamples, outputSamples,
                        numOutputSamplesToProduce);
    } else if (auto *i =
                   std::get_if<juce::Interpolators::Linear>(&interpolator)) {
      return i->process(speedRatio, inputSamples, outputSamples,
                        numOutputSamplesToProduce);
    } else if (auto *i = std::get_if<juce::Interpolators::CatmullRom>(
                   &interpolator)) {
      return i->process(speedRatio, inputSamples, outputSamples,
                        numOutputSamplesToProduce);
    } else if (auto *i =
                   std::get_if<juce::Interpolators::Lagrange>(&interpolator)) {
      return i->process(speedRatio, inputSamples, outputSamples,
                        numOutputSamplesToProduce);
    } else if (auto *i = std::get_if<juce::Interpolators::WindowedSinc>(
                   &interpolator)) {
      return i->process(speedRatio, inputSamples, outputSamples,
                        numOutputSamplesToProduce);
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc256>(
                   &interpolator)) {
      return i->process(speedRatio, inputSamples, outputSamples,
                        numOutputSamplesToProduce);
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc128>(
                   &interpolator)) {
      return i->process(speedRatio, inputSamples, outputSamples,
                        numOutputSamplesToProduce);
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc64>(
                   &interpolator)) {
      return i->process(speedRatio, inputSamples, outputSamples,
                        numOutputSamplesToProduce);
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc32>(
                   &interpolator)) {
      return i->process(speedRatio, inputSamples, outputSamples,
                        numOutputSamplesToProduce);
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc16>(
                   &interpolator)) {
      return i->process(speedRatio, inputSamples, outputSamples,
                        numOutputSamplesToProduce);
    } else if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc8>(
                   &interpolator)) {
      return i->process(speedRatio, inputSamples, outputSamples,
                        numOutputSamplesToProduce);
    } else {
      throw std::runtime_error("Unknown resampler quality!");
    }
  }

  // Position is split to avoid losing sub-sample precision on long streams.
  int processAtPosition(SamplePosition position, double filterRatio,
                        const float *input, float *output) {
    if (auto *i =
            std::get_if<juce::Interpolators::ZeroOrderHold>(&interpolator)) {
      float discarded;
      const int initialRead = positionInitialized ? 0 : 1;
      const double before = positionInitialized ? legacyFraction : 0.0;
      const double desired =
          cursorFor(position, positionConsumed + initialRead);
      double advance = desired - before;
      for (int attempt = 0; attempt < 3 && before + advance != desired;
           ++attempt)
        advance = std::nextafter(
            advance, before + advance < desired
                         ? std::numeric_limits<double>::infinity()
                         : -std::numeric_limits<double>::infinity());
      int used = i->process(advance, input, &discarded, 1);
      double actualCursor = before + advance;
      used += i->process(0.0, input + used, output, 1);
      while (actualCursor >= 1.0)
        actualCursor -= 1.0;
      legacyFraction = actualCursor;
      positionInitialized = true;
      positionConsumed += used;
      return used;
    }
    if (auto *i = std::get_if<juce::Interpolators::Linear>(&interpolator)) {
      float discarded;
      const int initialRead = positionInitialized ? 0 : 1;
      const double before = positionInitialized ? legacyFraction : 0.0;
      const double desired =
          cursorFor(position, positionConsumed + initialRead);
      double advance = desired - before;
      for (int attempt = 0; attempt < 3 && before + advance != desired;
           ++attempt)
        advance = std::nextafter(
            advance, before + advance < desired
                         ? std::numeric_limits<double>::infinity()
                         : -std::numeric_limits<double>::infinity());
      int used = i->process(advance, input, &discarded, 1);
      double actualCursor = before + advance;
      used += i->process(0.0, input + used, output, 1);
      while (actualCursor >= 1.0)
        actualCursor -= 1.0;
      legacyFraction = actualCursor;
      positionInitialized = true;
      positionConsumed += used;
      return used;
    }
    if (auto *i = std::get_if<juce::Interpolators::CatmullRom>(&interpolator)) {
      float discarded;
      const int initialRead = positionInitialized ? 0 : 1;
      const double before = positionInitialized ? legacyFraction : 0.0;
      const double desired =
          cursorFor(position, positionConsumed + initialRead);
      double advance = desired - before;
      for (int attempt = 0; attempt < 3 && before + advance != desired;
           ++attempt)
        advance = std::nextafter(
            advance, before + advance < desired
                         ? std::numeric_limits<double>::infinity()
                         : -std::numeric_limits<double>::infinity());
      int used = i->process(advance, input, &discarded, 1);
      double actualCursor = before + advance;
      used += i->process(0.0, input + used, output, 1);
      while (actualCursor >= 1.0)
        actualCursor -= 1.0;
      legacyFraction = actualCursor;
      positionInitialized = true;
      positionConsumed += used;
      return used;
    }
    if (auto *i = std::get_if<juce::Interpolators::Lagrange>(&interpolator)) {
      float discarded;
      const int initialRead = positionInitialized ? 0 : 1;
      const double before = positionInitialized ? legacyFraction : 0.0;
      const double desired =
          cursorFor(position, positionConsumed + initialRead);
      double advance = desired - before;
      for (int attempt = 0; attempt < 3 && before + advance != desired;
           ++attempt)
        advance = std::nextafter(
            advance, before + advance < desired
                         ? std::numeric_limits<double>::infinity()
                         : -std::numeric_limits<double>::infinity());
      int used = i->process(advance, input, &discarded, 1);
      double actualCursor = before + advance;
      used += i->process(0.0, input + used, output, 1);
      while (actualCursor >= 1.0)
        actualCursor -= 1.0;
      legacyFraction = actualCursor;
      positionInitialized = true;
      positionConsumed += used;
      return used;
    }
    if (auto *i =
            std::get_if<juce::Interpolators::WindowedSinc>(&interpolator)) {
      float discarded;
      const int initialRead = positionInitialized ? 0 : 1;
      const double before = positionInitialized ? legacyFraction : 0.0;
      const double desired =
          cursorFor(position, positionConsumed + initialRead);
      double advance = desired - before;
      for (int attempt = 0; attempt < 3 && before + advance != desired;
           ++attempt)
        advance = std::nextafter(
            advance, before + advance < desired
                         ? std::numeric_limits<double>::infinity()
                         : -std::numeric_limits<double>::infinity());
      int used = i->process(advance, input, &discarded, 1);
      double actualCursor = before + advance;
      used += i->process(0.0, input + used, output, 1);
      while (actualCursor >= 1.0)
        actualCursor -= 1.0;
      legacyFraction = actualCursor;
      positionInitialized = true;
      positionConsumed += used;
      return used;
    }
    if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc256>(
            &interpolator)) {
      const int used = i->processSampleAtCursor(
          cursorFor(position, positionConsumed), filterRatio, input, output);
      positionConsumed += used;
      return used;
    }
    if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc128>(
            &interpolator)) {
      const int used = i->processSampleAtCursor(
          cursorFor(position, positionConsumed), filterRatio, input, output);
      positionConsumed += used;
      return used;
    }
    if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc64>(
            &interpolator)) {
      const int used = i->processSampleAtCursor(
          cursorFor(position, positionConsumed), filterRatio, input, output);
      positionConsumed += used;
      return used;
    }
    if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc32>(
            &interpolator)) {
      const int used = i->processSampleAtCursor(
          cursorFor(position, positionConsumed), filterRatio, input, output);
      positionConsumed += used;
      return used;
    }
    if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc16>(
            &interpolator)) {
      const int used = i->processSampleAtCursor(
          cursorFor(position, positionConsumed), filterRatio, input, output);
      positionConsumed += used;
      return used;
    }
    if (auto *i = std::get_if<juce::FastInterpolators::WindowedSinc8>(
            &interpolator)) {
      const int used = i->processSampleAtCursor(
          cursorFor(position, positionConsumed), filterRatio, input, output);
      positionConsumed += used;
      return used;
    }
    throw std::runtime_error("Unknown resampler quality!");
  }

private:
  static double cursorFor(SamplePosition position, int64_t consumed) {
    const auto needed = position.frame + 1 - consumed;
    double cursor = double(needed) + position.fraction;
    if (position.fraction && cursor >= double(needed + 1))
      cursor = std::nextafter(double(needed + 1),
                              -std::numeric_limits<double>::infinity());
    return cursor;
  }
  double legacyFraction = 0;
  bool positionInitialized = false;
  int64_t positionConsumed = 0;
  std::variant<juce::Interpolators::ZeroOrderHold, juce::Interpolators::Linear,
               juce::Interpolators::CatmullRom, juce::Interpolators::Lagrange,
               juce::Interpolators::WindowedSinc,
               juce::FastInterpolators::WindowedSinc256,
               juce::FastInterpolators::WindowedSinc128,
               juce::FastInterpolators::WindowedSinc64,
               juce::FastInterpolators::WindowedSinc32,
               juce::FastInterpolators::WindowedSinc16,
               juce::FastInterpolators::WindowedSinc8>
      interpolator;
};

} // namespace Pedalboard
