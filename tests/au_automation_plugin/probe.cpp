/* Actual AUv2 diagnostic DSP, GPL-3.0-only. Apple utility sources retain their
 * license. */
#include "AUPlugInDispatch.h"
#include "MusicDeviceBase.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr AudioUnitParameterID gainId = 0x1001, frequencyId = 0x314159,
                               bypassId = 0xABC, blocksId = 0x998,
                               brokenId = 0x777, nonRealtimeId = 0x999;
int kindOf(AudioComponentInstance instance) {
  AudioComponentDescription description{};
  AudioComponentGetDescription(AudioComponentInstanceGetComponent(instance),
                               &description);
  const int suffix = description.componentSubType & 255;
  return suffix >= 'a' ? suffix - 'a' + 10 : suffix - '0';
}
bool hasBuses(int kind) { return kind >= 9 && kind <= 11; }
bool isInstrument(int kind) { return kind == 2 || (kind >= 6 && kind <= 8); }
int latency(int kind) {
  return kind == 1 || kind == 6 || kind == 11 ? 20
         : kind == 7                          ? 4097
         : kind == 8                          ? 16385
                                              : 0;
}
class Probe : public MusicDeviceBase {
public:
  explicit Probe(AudioComponentInstance instance)
      : MusicDeviceBase(instance,
                        hasBuses(kindOf(instance))       ? 2
                        : isInstrument(kindOf(instance)) ? 0
                                                         : 1,
                        hasBuses(kindOf(instance)) ? 2 : 1),
        kind(kindOf(instance)) {
    CreateElements();
    Globals()->SetParameter(gainId, kind == 4 ? 1 : 0);
    Globals()->SetParameter(frequencyId, kind >= 10 ? 20000 : 20);
    Globals()->SetParameter(bypassId, 0);
    Globals()->SetParameter(blocksId, 0);
    Globals()->SetParameter(brokenId, 0);
    Globals()->SetParameter(nonRealtimeId, 0);
  }
  bool CanScheduleParameters() const override { return false; }
  bool StreamFormatWritable(AudioUnitScope, AudioUnitElement) override {
    return true;
  }
  UInt32 SupportedNumChannels(const AUChannelInfo **out) override {
    if (hasBuses(kind)) {
      static const AUChannelInfo buses[] = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
      if (out)
        *out = buses;
      return 4;
    }
    static const AUChannelInfo effect[] = {{1, 1}, {2, 2}};
    static const AUChannelInfo instrument[] = {{0, 1}, {0, 2}};
    if (out)
      *out = isInstrument(kind) ? instrument : effect;
    return 2;
  }
  OSStatus GetParameterInfo(AudioUnitScope scope, AudioUnitParameterID id,
                            AudioUnitParameterInfo &info) override {
    if (scope != kAudioUnitScope_Global)
      return kAudioUnitErr_InvalidScope;
    info = {};
    const char *name = nullptr;
    info.flags =
        kAudioUnitParameterFlag_IsReadable | kAudioUnitParameterFlag_IsWritable;
    if (id == gainId) {
      name = "Gain";
      info.minValue = kind == 4 ? 0 : -60;
      info.maxValue = kind == 4 ? 2 : 12;
      info.defaultValue = kind == 4 ? 1 : 0;
      info.unit = kind == 4 ? kAudioUnitParameterUnit_LinearGain
                            : kAudioUnitParameterUnit_Decibels;
      if (kind == 4 || kind == 5)
        info.flags |= kAudioUnitParameterFlag_ValuesHaveStrings;
    } else if (id == frequencyId) {
      name = "Frequency";
      info.minValue = 20;
      info.maxValue =
          kind == 3 ? GetOutput(0)->GetStreamFormat().mSampleRate * .5 : 20000;
      info.defaultValue = 20;
      info.unit = kAudioUnitParameterUnit_Hertz;
    } else if (id == bypassId) {
      name = "Bypass";
      info.maxValue = 1;
      info.unit = kAudioUnitParameterUnit_Boolean;
    } else if (id == blocksId) {
      name = "Blocks";
      info.maxValue = 1e9;
      info.flags = kAudioUnitParameterFlag_IsReadable;
    } else if (id == brokenId) {
      name = "Broken";
      info.maxValue = 1;
      info.flags |= kAudioUnitParameterFlag_ValuesHaveStrings;
    } else if (id == nonRealtimeId) {
      name = "Setup";
      info.maxValue = 1;
      info.flags |= kAudioUnitParameterFlag_NonRealTime;
    } else
      return kAudioUnitErr_InvalidParameter;
    std::snprintf(info.name, sizeof(info.name), "%s", name);
    return noErr;
  }
  OSStatus GetPropertyInfo(AudioUnitPropertyID id, AudioUnitScope scope,
                           AudioUnitElement element, UInt32 &size,
                           Boolean &writable) override {
    if (id == kAudioUnitProperty_ParameterStringFromValue) {
      size = sizeof(AudioUnitParameterStringFromValue);
      writable = false;
      return noErr;
    }
    if (id == kAudioUnitProperty_ParameterValueFromString) {
      size = sizeof(AudioUnitParameterValueFromString);
      writable = false;
      return noErr;
    }
    return MusicDeviceBase::GetPropertyInfo(id, scope, element, size, writable);
  }
  OSStatus GetProperty(AudioUnitPropertyID id, AudioUnitScope scope,
                       AudioUnitElement element, void *data) override {
    if (id == kAudioUnitProperty_ParameterStringFromValue) {
      auto &value = *static_cast<AudioUnitParameterStringFromValue *>(data);
      if (value.inParamID == brokenId) {
        value.outString = CFStringCreateWithCString(nullptr, "unsupported",
                                                    kCFStringEncodingUTF8);
        return noErr;
      }
      if (value.inParamID == gainId && kind == 5) {
        if (value.inValue && *value.inValue <= -60) {
          value.outString =
              CFStringCreateWithCString(nullptr, "-inf", kCFStringEncodingUTF8);
          return noErr;
        }
        return kAudioUnitErr_InvalidProperty;
      }
      if (value.inParamID != gainId || kind != 4)
        return kAudioUnitErr_InvalidProperty;
      const float raw =
          value.inValue ? *value.inValue : Globals()->GetParameter(gainId);
      char text[96];
      std::snprintf(text, sizeof(text), "%.9f dB", 20 * std::log10(raw));
      value.outString =
          CFStringCreateWithCString(nullptr, text, kCFStringEncodingUTF8);
      return noErr;
    }
    if (id == kAudioUnitProperty_ParameterValueFromString) {
      auto &value = *static_cast<AudioUnitParameterValueFromString *>(data);
      if (value.inParamID != gainId || kind != 4)
        return kAudioUnitErr_InvalidProperty;
      char text[128]{};
      CFStringGetCString(value.inString, text, sizeof(text),
                         kCFStringEncodingUTF8);
      value.outValue = std::pow(10.0, std::strtod(text, nullptr) / 20);
      return noErr;
    }
    return MusicDeviceBase::GetProperty(id, scope, element, data);
  }
  OSStatus Initialize() override {
    const auto status = MusicDeviceBase::Initialize();
    clear();
    return status;
  }
  OSStatus Reset(AudioUnitScope scope, AudioUnitElement element) override {
    clear();
    return MusicDeviceBase::Reset(scope, element);
  }
  Float64 GetLatency() override {
    return latency(kind) / GetOutput(0)->GetStreamFormat().mSampleRate;
  }
  OSStatus MIDIEvent(UInt32 status, UInt32 data1, UInt32 velocity,
                     UInt32 offset) override {
    if ((status & 0xf0) == 0x90 || (status & 0xf0) == 0x80)
      notes.push_back({offset, (status & 0xf0) == 0x90 && velocity > 0});
    if (kind >= 10 && (status & 0xf0) == 0xb0 && data1 == 1)
      controllers.push_back({offset, velocity >= 64});
    return noErr;
  }
  OSStatus Render(AudioUnitRenderActionFlags &flags, const AudioTimeStamp &time,
                  UInt32 frames) override {
    if (hasBuses(kind)) {
      for (int bus = 0; bus < 2; ++bus) {
        const auto status = PullInput(bus, flags, time, frames);
        if (status != noErr)
          return status;
      }
      // RenderBus calls Render once per timestamp; initialize every output.
      GetOutput(1)->PrepareBuffer(frames);
      const auto &main = GetInput(0)->GetBufferList();
      const auto &key = GetInput(1)->GetBufferList();
      auto &output = GetOutput(0)->GetBufferList();
      auto &aux = GetOutput(1)->GetBufferList();
      size_t event = 0, controller = 0;
      const double gain =
          std::pow(10.0, Globals()->GetParameter(gainId) / 20.0);
      const double frequency = Globals()->GetParameter(frequencyId);
      for (UInt32 i = 0; i < frames; ++i) {
        while (event < notes.size() && notes[event].first <= i)
          note = notes[event++].second;
        while (controller < controllers.size() &&
               controllers[controller].first <= i)
          Globals()->SetParameter(bypassId,
                                  controllers[controller++].second ? 1 : 0);
        const float duck =
            std::abs(static_cast<float *>(key.mBuffers[0].mData)[i]) > .5f
                ? .25f
                : 1.f;
        for (UInt32 c = 0; c < aux.mNumberBuffers; ++c)
          static_cast<float *>(aux.mBuffers[c].mData)[i] =
              .5f * static_cast<float *>(
                        main.mBuffers[c % main.mNumberBuffers].mData)[i];
        for (UInt32 c = 0; c < output.mNumberBuffers; ++c) {
          const float input = static_cast<float *>(
              main.mBuffers[c % main.mNumberBuffers].mData)[i];
          float value = kind >= 10 && Globals()->GetParameter(bypassId) >= .5
                            ? input
                        : kind == 9 ? duck * input
                                    : (float)(duck * input * (note ? 1 : 0) *
                                              gain * frequency / 20000.0);
          if (latency(kind)) {
            const float old = delay[c][position];
            delay[c][position] = value;
            value = old;
          }
          static_cast<float *>(output.mBuffers[c].mData)[i] = value;
        }
        position = (position + 1) % std::max(20, latency(kind));
      }
      notes.clear();
      controllers.clear();
      Globals()->SetParameter(blocksId, ++blocks);
      return noErr;
    }
    if (!isInstrument(kind)) {
      const auto status = PullInput(0, flags, time, frames);
      if (status != noErr)
        return status;
    }
    auto &output = GetOutput(0)->GetBufferList();
    const double gain =
        kind == 4 ? Globals()->GetParameter(gainId)
                  : std::pow(10.0, Globals()->GetParameter(gainId) / 20.0);
    const double actualGain =
        kind == 5 && Globals()->GetParameter(gainId) <= -60 ? 0 : gain;
    const double frequency = Globals()->GetParameter(frequencyId);
    const bool bypass = Globals()->GetParameter(bypassId) >= .5;
    size_t event = 0;
    for (UInt32 i = 0; i < frames; ++i) {
      while (event < notes.size() && notes[event].first <= i)
        note = notes[event++].second;
      for (UInt32 c = 0; c < output.mNumberBuffers; ++c) {
        const float input =
            isInstrument(kind)
                ? (note ? 1.0f : 0.0f)
                : static_cast<float *>(
                      GetInput(0)->GetBufferList().mBuffers[c].mData)[i];
        float value =
            bypass ? input : (float)(input * actualGain * frequency / 20000.0);
        if (latency(kind)) {
          const float old = delay[c][position];
          delay[c][position] = value;
          value = old;
        }
        static_cast<float *>(output.mBuffers[c].mData)[i] = value;
      }
      position = (position + 1) % std::max(20, latency(kind));
    }
    notes.clear();
    Globals()->SetParameter(blocksId, ++blocks);
    return noErr;
  }

private:
  void clear() {
    for (auto &channel : delay)
      channel.fill(0);
    position = 0;
    note = false;
    notes.clear();
    controllers.clear();
    blocks = 0;
  }
  int kind, position = 0, blocks = 0;
  bool note = false;
  std::vector<std::pair<UInt32, bool>> notes, controllers;
  std::array<std::array<float, 16385>, 2> delay{};
};
} // namespace
extern "C" __attribute__((visibility("default"))) void *
AutomationAUFactory(const AudioComponentDescription *description) {
  return AUMusicDeviceFactory<Probe>::Factory(description);
}
