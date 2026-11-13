/*
 * Clocked resampling with bounded, source-timestamped queues.
 * Licensed under the GNU General Public License, Version 3.0.
 * See LICENSE for details.
 */
#pragma once
#include "Resampling.h"
#include <deque>

namespace Pedalboard {

// A source-time clock shared by the downsampler and reconstructor. The original
// interpolation kernels remain the sole owner of sample/filter arithmetic.
class ClockedResample {
public:
  static constexpr int ratioLimit = 64;
  void invalidate() { prepared = false; }
  int latency() const { return delay; }

  void prepare(const juce::dsp::ProcessSpec &spec, ResamplingQuality quality,
               float target) {
    if (prepared && rate == spec.sampleRate && channels == spec.numChannels &&
        selectedQuality == quality)
      return;
    prepared = false;
    rate = spec.sampleRate;
    channels = spec.numChannels;
    selectedQuality = quality;
    down.clear();
    up.clear();
    down.resize(channels);
    up.resize(channels);
    for (unsigned c = 0; c < channels; ++c) {
      down[c].setQuality(quality);
      up[c].setQuality(quality);
    }
    base = (int)down.front().getBaseLatency();
    delay = base * (ratioLimit + 1) + 2;
    initialRatio = rate / target;
    const double originalDelay = base * initialRatio + base;
    roundingOffset = std::round(originalDelay) - originalDelay;
    nativeCapacity = ratioLimit * 2 + 8;
    targetCapacity = delay + 2 * base + ratioLimit + 8;
    clockCapacity = delay + 4;
    input.assign(channels, std::vector<float>(2 * nativeCapacity));
    intermediate.assign(channels, std::vector<float>(2 * targetCapacity));
    clocks.assign(clockCapacity, {});
    events.clear();
    inputFrames = downConsumed = targetProduced = upConsumed = 0;
    const double initialCycles = base / initialRatio;
    ticks = (int64_t)std::floor(initialCycles);
    phase = initialCycles - ticks;
    for (int64_t j = 0; j <= ticks; ++j) {
      double time = -double(base) + j * initialRatio;
      if (j == ticks && phase == 0)
        time = 0;
      const double whole = std::floor(time);
      events.push_back({{(int64_t)whole, time - whole}, initialRatio, j});
    }
    pendingEndpoint = false;
    prepared = true;
  }

  int process(const juce::dsp::ProcessContextReplacing<float> &context,
              const std::vector<float> *curve, size_t cursor,
              float heldTarget) {
    auto block = context.getOutputBlock();
    int valid = 0;
    for (size_t i = 0; i < block.getNumSamples(); ++i) {
      const float target =
          curve ? (*curve)[std::min(cursor + i, curve->size() - 1)]
                : heldTarget;
      const double increment = target / rate;
      const int64_t n = inputFrames++;
      for (unsigned c = 0; c < channels; ++c)
        store(input[c], nativeCapacity, n, block.getChannelPointer(c)[i]);
      clocks[n % clockCapacity] = {ticks, phase, increment};
      if (pendingEndpoint) {
        events.push_back({{n, 0}, 1.0 / increment, ticks});
        pendingEndpoint = false;
      }
      const double next = phase + increment;
      if (next >= 1.0) {
        const double fraction = (1.0 - phase) / increment;
        ++ticks;
        // Integration roundoff at an integer source timestamp must not select
        // the adjacent sample (particularly for zero-order hold).
        constexpr double endpointTolerance = 1e-12;
        if (fraction >= 1.0 - endpointTolerance)
          pendingEndpoint = true;
        else if (fraction <= endpointTolerance)
          events.push_back({{n, 0}, 1.0 / increment, ticks});
        else
          events.push_back({{n, fraction}, 1.0 / increment, ticks});
        phase = next - 1.0;
      } else
        phase = next;

      while (!events.empty()) {
        const auto &event = events.front();
        const SamplePosition position{event.time.frame + base,
                                      event.time.fraction};
        if (position.frame >= inputFrames)
          break;
        if (event.index != targetProduced ||
            position.frame + 1 < downConsumed ||
            inputFrames - downConsumed > nativeCapacity)
          throw std::runtime_error(
              "Invalid downsampling clock or input ring bounds");
        int consumed = -1;
        for (unsigned c = 0; c < channels; ++c) {
          float sample;
          const int used = down[c].processAtPosition(
              position, event.ratio,
              input[c].data() + downConsumed % nativeCapacity, &sample);
          if (consumed >= 0 && consumed != used)
            throw std::runtime_error("Channel clock mismatch");
          consumed = used;
          store(intermediate[c], targetCapacity, targetProduced, sample);
        }
        if (consumed != position.frame + 1 - downConsumed)
          throw std::runtime_error("Unexpected downsampling input consumption");
        downConsumed += consumed;
        ++targetProduced;
        events.pop_front();
      }
      if (targetProduced - upConsumed > targetCapacity)
        throw std::runtime_error("Target ring overflow");
      if (n < delay)
        continue;
      const auto clock = clockAt(n - delay, roundingOffset);
      const SamplePosition outputPosition{clock.cycle + base, clock.fraction};
      if (outputPosition.frame >= targetProduced ||
          outputPosition.frame + 1 < upConsumed)
        throw std::runtime_error(
            "Output clock requires unavailable target data");
      int consumed = -1;
      for (unsigned c = 0; c < channels; ++c) {
        const int used = up[c].processAtPosition(
            outputPosition, clock.increment,
            intermediate[c].data() + upConsumed % targetCapacity,
            block.getChannelPointer(c) + i);
        if (consumed >= 0 && consumed != used)
          throw std::runtime_error("Channel clock mismatch");
        consumed = used;
      }
      if (consumed != outputPosition.frame + 1 - upConsumed)
        throw std::runtime_error("Unexpected reconstruction input consumption");
      upConsumed += consumed;
      ++valid;
    }
    return valid;
  }

private:
  struct Clock {
    int64_t cycle = 0;
    double fraction = 0, increment = 0;
  };
  struct Event {
    SamplePosition time;
    double ratio;
    int64_t index;
  };
  Clock clockAt(int64_t frame, double offset) const {
    if (offset < 0) {
      --frame;
      offset += 1;
    }
    Clock clock;
    if (frame < 0) {
      const double cycles = (double(frame) + offset + base) / initialRatio;
      clock.cycle = (int64_t)std::floor(cycles);
      clock.fraction = cycles - clock.cycle;
      clock.increment = 1.0 / initialRatio;
      return clock;
    }
    clock = clocks[frame % clockCapacity];
    clock.fraction += offset * clock.increment;
    if (clock.fraction >= 1) {
      ++clock.cycle;
      clock.fraction -= 1;
    }
    return clock;
  }
  static void store(std::vector<float> &ring, int capacity, int64_t index,
                    float value) {
    const auto slot = index % capacity;
    ring[slot] = ring[slot + capacity] = value;
  }
  bool prepared = false, pendingEndpoint = false;
  double rate = 0, initialRatio = 1, roundingOffset = 0, phase = 0;
  unsigned channels = 0;
  int base = 0, delay = 0, nativeCapacity = 0, targetCapacity = 0,
      clockCapacity = 0;
  int64_t inputFrames = 0, downConsumed = 0, targetProduced = 0, upConsumed = 0,
          ticks = 0;
  ResamplingQuality selectedQuality = ResamplingQuality::WindowedSinc32;
  std::vector<VariableQualityResampler> down, up;
  std::vector<std::vector<float>> input, intermediate;
  std::vector<Clock> clocks;
  std::deque<Event> events;
};
} // namespace Pedalboard
