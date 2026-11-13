/* Caller-owned hosted transport and callback lifetime. GPL-3.0-only. */
#pragma once
#include "JuceHeader.h"
#include <climits>
#include <cmath>
#include <limits>
#include <mutex>
#include <set>
#if JUCE_MAC && JUCE_PLUGINHOST_AU
#include <AudioToolbox/AudioToolbox.h>
#endif

namespace Pedalboard {

struct HostedTransport {
  static constexpr double maxFrame = 9007199254740990.0;
  bool enabled = false, playing = true, recording = false, looping = false;
  double rate, bpm = 120, ppq = 0, step = 0, loopStart = 0, loopEnd = 0;
  double fraction = 0;
  juce::int64 sample = 0;
  int numerator = 4, denominator = 4;

  static double number(py::handle value, const std::string &name) {
    if ((!py::isinstance<py::float_>(value) &&
         !py::isinstance<py::int_>(value)) ||
        py::isinstance<py::bool_>(value))
      throw py::type_error("transport " + name + " must be a real number");
    const double result = py::cast<double>(value);
    if (!std::isfinite(result))
      throw py::value_error("transport " + name + " must be finite");
    return result;
  }
  static juce::int64 integer(py::handle value, const std::string &name) {
    if (!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value))
      throw py::type_error("transport " + name + " must be an integer");
    const auto result = py::cast<juce::int64>(value);
    if (result < -maxFrame || result > maxFrame)
      throw py::value_error("transport " + name + " exceeds exact frame range");
    return result;
  }
  static bool flag(py::handle value, const std::string &name) {
    if (!py::isinstance<py::bool_>(value))
      throw py::type_error("transport " + name + " must be a bool");
    return py::cast<bool>(value);
  }
  HostedTransport(py::object value, double rate, int frames) : rate(rate) {
    if (value.is_none())
      return;
    if (!py::isinstance<py::dict>(value))
      throw py::type_error("transport must be a dict or None");
    enabled = true;
    const auto map = value.cast<py::dict>();
    const std::set<std::string> names{
        "bpm",     "sample_position", "ppq_position",
        "playing", "recording",       "time_signature",
        "looping", "loop_start_ppq",  "loop_end_ppq"};
    for (const auto item : map) {
      if (!py::isinstance<py::str>(item.first) ||
          !names.count(py::cast<std::string>(item.first)))
        throw py::value_error("Unknown transport field");
    }
    if (map.contains("bpm"))
      bpm = number(map["bpm"], "bpm");
    if (!(bpm > 0))
      throw py::value_error("transport bpm must be positive");
    step = bpm / (60.0 * rate);
    if (!(step > 0) || !std::isfinite(step) || !std::isfinite(1.0 / step) ||
        1.0 / step > UINT32_MAX)
      throw py::value_error(
          "Unsafe transport beat increment or native beat-period range");
    if (map.contains("sample_position"))
      sample = integer(map["sample_position"], "sample_position");
    ppq = map.contains("ppq_position")
              ? number(map["ppq_position"], "ppq_position")
              : sample * step;
    if (map.contains("playing"))
      playing = flag(map["playing"], "playing");
    if (map.contains("recording"))
      recording = flag(map["recording"], "recording");
    if (map.contains("looping"))
      looping = flag(map["looping"], "looping");
    if (map.contains("time_signature")) {
      auto meter = map["time_signature"];
      if (!py::isinstance<py::tuple>(meter) && !py::isinstance<py::list>(meter))
        throw py::type_error(
            "transport time_signature must be a pair of integers");
      const auto values = py::reinterpret_borrow<py::sequence>(meter);
      if (values.size() != 2)
        throw py::value_error("transport time_signature requires two integers");
      const auto n = integer(values[0], "time_signature numerator");
      const auto d = integer(values[1], "time_signature denominator");
      if (n < 1 || n > INT_MAX || d < 1 || d > INT_MAX || (d & (d - 1)))
        throw py::value_error("Unsafe transport meter; denominator must be a "
                              "positive power of two");
      numerator = (int)n;
      denominator = (int)d;
    }
    const bool hasStart = map.contains("loop_start_ppq"),
               hasEnd = map.contains("loop_end_ppq");
    if (hasStart != hasEnd || (looping && !hasStart))
      throw py::value_error("transport loop requires both bounds");
    if (hasStart) {
      loopStart = number(map["loop_start_ppq"], "loop_start_ppq");
      loopEnd = number(map["loop_end_ppq"], "loop_end_ppq");
      if (!(loopEnd > loopStart) || !std::isfinite(loopEnd - loopStart))
        throw py::value_error("transport loop end must exceed start");
    }
    if (playing && ppq + step == ppq)
      throw py::value_error(
          "Transport PPQ exceeds sample-resolution precision");
    if (looping && playing && ppq >= loopEnd)
      throw py::value_error("Initial playing PPQ must precede loop end");
    // Validate all derived fields before MIDI iteration or native preparation.
    position(0);
    position(frames);
    if (looping) {
      const double first = sample + (loopStart - ppq) / step;
      if (!std::isfinite(first) || std::abs(first) > maxFrame)
        throw py::value_error("Unsafe transport loop sample position");
    }
  }
  double tolerance(double beat) const {
    return std::min(step * .125,
                    std::numeric_limits<double>::epsilon() * 8 *
                        std::max({1.0, std::abs(beat), std::abs(ppq)}));
  }
  double beatAt(size_t offset) const {
    double value = ppq + (playing ? (double)offset * step : 0);
    if (playing && looping && value >= loopEnd - tolerance(value)) {
      const double length = loopEnd - loopStart;
      double phase = std::fmod(value - loopStart, length);
      if (length - phase <= tolerance(value) || phase < 0)
        phase = 0;
      value = loopStart + phase;
    }
    return value;
  }
  double displacementAt(size_t offset) const {
    const double unwrapped = ppq + (playing ? (double)offset * step : 0);
    if (playing && looping && unwrapped >= loopEnd - tolerance(unwrapped))
      return fraction + (beatAt(offset) - ppq) / step;
    return fraction + (playing ? (double)offset : 0);
  }
  double exactSampleAt(size_t offset) const {
    return sample + displacementAt(offset);
  }
  juce::int64 roundedDisplacement(size_t offset) const {
    const double exact = displacementAt(offset), beat = beatAt(offset);
    const double lower = std::floor(exact);
    // Round relative displacement before adding the integer origin. Otherwise
    // a large origin inflates the tie bound and can turn .499 into one frame.
    const double error = std::min(
        .125, std::numeric_limits<double>::epsilon() * 16 *
                  std::max({1.0, std::abs(exact), (double)offset,
                            std::abs(ppq / step), std::abs(beat / step)}));
    return (juce::int64)(lower + (exact - lower >= .5 - error ? 1 : 0));
  }
  juce::AudioPlayHead::CurrentPositionInfo position(size_t offset) const {
    juce::AudioPlayHead::CurrentPositionInfo result;
    const double exact = exactSampleAt(offset), beat = beatAt(offset);
    const double bar = 4.0 * numerator / denominator;
    const double downbeat = std::floor(beat / bar) * bar;
    if (!std::isfinite(exact) || std::abs(exact) > maxFrame ||
        !std::isfinite(beat) || !std::isfinite(downbeat))
      throw std::invalid_argument("Unsafe derived transport position");
    result.bpm = bpm;
    // Ties toward positive infinity advance one frame even across zero.
    result.timeInSamples = sample + roundedDisplacement(offset);
    result.timeInSeconds = result.timeInSamples / rate;
    result.ppqPosition = beat;
    result.ppqPositionOfLastBarStart = downbeat;
    result.timeSigNumerator = numerator;
    result.timeSigDenominator = denominator;
    result.isPlaying = playing;
    result.isRecording = recording;
    result.isLooping = looping;
    result.ppqLoopStart = loopStart;
    result.ppqLoopEnd = loopEnd;
    return result;
  }
  int span(size_t start, int requested) const {
    if (!enabled || !playing || !looping)
      return requested;
    const double distance = (loopEnd - beatAt(start)) / step;
    // Propagate the beat-phase rounding bound into frame coordinates. A
    // distance-only epsilon misses accumulated fmod error after later wraps.
    const double eps = std::max(tolerance(ppq + start * step) / step,
                                std::numeric_limits<double>::epsilon() * 8 *
                                    std::max(1.0, std::abs(distance)));
    if (distance >= requested + eps)
      return requested;
    return std::max(1, std::min(requested, (int)std::ceil(distance - eps)));
  }
};

class HostedClock final : public juce::AudioPlayHead {
public:
  using Position = juce::AudioPlayHead::CurrentPositionInfo;
  struct Snapshot {
    Position position;
    double rate;
    bool changed;
  };
  void reset() {
    std::scoped_lock lock(mutex);
    active = hasNext = installed = nextWrapped = false;
    fallback = nullptr;
  }
  static bool same(const Position &a, const Position &b, double tolerance) {
    return a.timeInSamples == b.timeInSamples &&
           std::abs(a.ppqPosition - b.ppqPosition) <= tolerance &&
           a.bpm == b.bpm && a.isPlaying == b.isPlaying &&
           a.isRecording == b.isRecording && a.isLooping == b.isLooping &&
           a.timeSigNumerator == b.timeSigNumerator &&
           a.timeSigDenominator == b.timeSigDenominator &&
           a.ppqLoopStart == b.ppqLoopStart && a.ppqLoopEnd == b.ppqLoopEnd;
  }
  void begin(HostedTransport &request, bool continuation,
             juce::AudioPlayHead *previous) {
    std::scoped_lock lock(mutex);
    fallback = previous == this ? nullptr : previous;
    active = request.enabled;
    if (!request.enabled) {
      hasNext = false;
      return;
    }
    if (continuation && hasNext && nextRate == request.rate &&
        same(next, request.position(0), request.tolerance(request.ppq)))
      request.fraction = nextFraction;
    else
      hasNext = false;
    current = request.position(0);
    rate = request.rate;
    changed = !hasNext || nextWrapped;
  }
  void update(const HostedTransport &request, size_t offset, int count) {
    if (!request.enabled)
      return;
    const auto now = request.position(offset),
               after = request.position(offset + count);
    std::scoped_lock lock(mutex);
    changed = !hasNext || nextWrapped || nextRate != request.rate ||
              !same(next, now, request.tolerance(now.ppqPosition));
    current = now;
    rate = request.rate;
    next = after;
    nextRate = rate;
    nextFraction = request.displacementAt(offset + count) -
                   request.roundedDisplacement(offset + count);
    nextWrapped =
        request.playing && request.looping &&
        count * request.step >= request.loopEnd - now.ppqPosition -
                                    request.tolerance(now.ppqPosition);
    hasNext = true;
  }
  void end() {
    std::scoped_lock lock(mutex);
    active = false;
  }
  Snapshot snapshot() {
    juce::AudioPlayHead *other = nullptr;
    double fallbackRate;
    {
      std::scoped_lock lock(mutex);
      if (active)
        return {current, rate, changed};
      other = fallback;
      fallbackRate = rate;
    }
    Position value;
    if (other)
      other->getCurrentPosition(value);
    return {value, fallbackRate, false};
  }
  bool getCurrentPosition(Position &result) override {
    result = snapshot().position;
    return true;
  }
#if JUCE_MAC && JUCE_PLUGINHOST_AU
  bool installed = false;
  static OSStatus beat(void *user, Float64 *ppq, Float64 *bpm) {
    const auto s = static_cast<HostedClock *>(user)->snapshot();
    if (ppq)
      *ppq = s.position.ppqPosition;
    if (bpm)
      *bpm = s.position.bpm;
    return noErr;
  }
  static OSStatus musical(void *user, UInt32 *delta, Float32 *num, UInt32 *den,
                          Float64 *bar) {
    const auto s = static_cast<HostedClock *>(user)->snapshot();
    if (num)
      *num = s.position.timeSigNumerator;
    if (den)
      *den = s.position.timeSigDenominator;
    if (bar)
      *bar = s.position.ppqPositionOfLastBarStart;
    const double phase =
        std::ceil(s.position.ppqPosition) - s.position.ppqPosition;
    const double samples = std::ceil(phase * 60.0 * s.rate / s.position.bpm);
    if (!std::isfinite(samples) || samples > UINT32_MAX)
      return kAudioUnitErr_InvalidPropertyValue;
    if (delta)
      *delta = (UInt32)samples;
    return noErr;
  }
  static OSStatus transport2(void *user, Boolean *playing, Boolean *recording,
                             Boolean *changed, Float64 *sample,
                             Boolean *looping, Float64 *start, Float64 *end) {
    const auto s = static_cast<HostedClock *>(user)->snapshot();
    if (playing)
      *playing = s.position.isPlaying;
    if (recording)
      *recording = s.position.isRecording;
    if (changed)
      *changed = s.changed;
    if (sample)
      *sample = s.position.timeInSamples;
    if (looping)
      *looping = s.position.isLooping;
    if (start)
      *start = s.position.ppqLoopStart;
    if (end)
      *end = s.position.ppqLoopEnd;
    return noErr;
  }
  static OSStatus transport(void *user, Boolean *playing, Boolean *changed,
                            Float64 *sample, Boolean *looping, Float64 *start,
                            Float64 *end) {
    return transport2(user, playing, nullptr, changed, sample, looping, start,
                      end);
  }
  void install(AudioUnit unit) {
    HostCallbackInfo callbacks{this, beat, musical, transport, transport2};
    const auto status = AudioUnitSetProperty(
        unit, kAudioUnitProperty_HostCallbacks, kAudioUnitScope_Global, 0,
        &callbacks, sizeof(callbacks));
    if (status != noErr)
      throw std::runtime_error(
          "Audio Unit rejected host transport callbacks: " +
          std::to_string(status));
    installed = true;
  }
#else
  bool installed = false;
#endif
private:
  std::mutex mutex;
  Position current, next;
  double rate = 44100, nextRate = 0, nextFraction = 0;
  bool active = false, hasNext = false, changed = false, nextWrapped = false;
  juce::AudioPlayHead *fallback = nullptr;
};

struct ScopedHostedClock {
  juce::AudioPluginInstance &instance;
  HostedClock &clock;
  juce::AudioPlayHead *previous;
  ScopedHostedClock(juce::AudioPluginInstance &instance, HostedClock &clock,
                    HostedTransport &request, bool continuation)
      : instance(instance), clock(clock), previous(instance.getPlayHead()) {
    clock.begin(request, continuation, previous);
    if (request.enabled)
      instance.setPlayHead(&clock);
  }
  ~ScopedHostedClock() {
    instance.setPlayHead(previous);
    clock.end();
  }
};

} // namespace Pedalboard
