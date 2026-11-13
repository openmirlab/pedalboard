/* Native Audio Unit parameter metadata and pure conversions. GPL-3.0-only. */
#pragma once
#include "JuceHeader.h"
#if JUCE_MAC && JUCE_PLUGINHOST_AU
#include <AudioToolbox/AudioToolbox.h>
#include <AudioToolbox/AudioUnitUtilities.h>

namespace Pedalboard {
struct AudioUnitAutomationBackend {
  struct Parameter {
    uint32_t id;
    juce::String name, units;
    int steps;
    bool automatable, bypass = false;
    float minimum, maximum;
    AudioUnitParameterOptions flags;
    AudioUnitParameterUnit unit;
    bool isV3;
  };
  static AudioUnit handle(juce::AudioPluginInstance &p) {
    auto au = static_cast<AudioUnit>(p.getPlatformSpecificData());
    if (!au)
      throw std::invalid_argument("Audio Unit has no native instance");
    return au;
  }
  static Parameter describeById(juce::AudioPluginInstance &p, uint32_t id) {
    AudioUnitParameterInfo info{};
    UInt32 size = sizeof(info);
    if (AudioUnitGetProperty(handle(p), kAudioUnitProperty_ParameterInfo,
                             kAudioUnitScope_Global, id, &info, &size) != noErr)
      throw std::invalid_argument("Audio Unit parameter ID is unavailable");
    juce::String name(info.name, sizeof(info.name));
    if (info.flags & kAudioUnitParameterFlag_HasCFNameString) {
      name = juce::String::fromCFString(info.cfNameString);
      if (info.cfNameString &&
          (info.flags & kAudioUnitParameterFlag_CFNameRelease))
        CFRelease(info.cfNameString);
    }
    juce::String units;
    switch (info.unit) {
    case kAudioUnitParameterUnit_Decibels:
      units = "dB";
      break;
    case kAudioUnitParameterUnit_Hertz:
      units = "Hz";
      break;
    case kAudioUnitParameterUnit_Percent:
      units = "%";
      break;
    case kAudioUnitParameterUnit_Seconds:
      units = "s";
      break;
    case kAudioUnitParameterUnit_Milliseconds:
      units = "ms";
      break;
    case kAudioUnitParameterUnit_CustomUnit:
      if (info.unitName)
        units = juce::String::fromCFString(info.unitName);
      break;
    default:
      break;
    }
    const bool discrete = info.unit == kAudioUnitParameterUnit_Boolean ||
                          info.unit == kAudioUnitParameterUnit_Indexed;
    const bool writable = (info.flags & kAudioUnitParameterFlag_IsWritable) &&
                          !(info.flags & kAudioUnitParameterFlag_NonRealTime);
    if (!std::isfinite(info.minValue) || !std::isfinite(info.maxValue) ||
        info.minValue >= info.maxValue)
      throw std::invalid_argument(
          "Audio Unit has no finite increasing parameter range");
    const double steps = (double)info.maxValue - info.minValue + 1;
    AudioComponentDescription component{};
    if (AudioComponentGetDescription(
            AudioComponentInstanceGetComponent(handle(p)), &component) != noErr)
      throw std::invalid_argument(
          "Audio Unit component description is unavailable");
    return {id,
            name,
            units,
            discrete && steps < INT_MAX ? (int)steps - 1 : INT_MAX,
            writable,
            false,
            info.minValue,
            info.maxValue,
            info.flags,
            info.unit,
            (component.componentFlags & kAudioComponentFlag_IsV3AudioUnit) !=
                0};
  }

  static Parameter describe(juce::AudioPluginInstance &p, int index) {
    auto *parameter = p.getHostedParameter(index);
    if (!parameter)
      throw std::invalid_argument("Audio Unit has no stable parameter ID");
    const auto string = parameter->getParameterID().toStdString();
    size_t consumed = 0;
    const auto id = std::stoull(string, &consumed);
    if (consumed != string.size() || id > UINT32_MAX)
      throw std::invalid_argument(
          "Audio Unit parameter ID is not a global UInt32 ID");
    return describeById(p, (uint32_t)id);
  }
  static double plain(juce::AudioPluginInstance &, const Parameter &p,
                      double raw) {
    return p.minimum + ((double)p.maximum - p.minimum) * raw;
  }
  static double normalized(juce::AudioPluginInstance &, const Parameter &p,
                           double value) {
    return (value - p.minimum) / ((double)p.maximum - p.minimum);
  }
  static double neighbor(juce::AudioPluginInstance &instance,
                         const Parameter &p, double raw, int direction) {
    const float physical = (float)plain(instance, p, raw);
    const float adjacent = std::nextafter(
        physical, direction < 0 ? -std::numeric_limits<float>::infinity()
                                : std::numeric_limits<float>::infinity());
    return std::clamp(normalized(instance, p, adjacent), 0.0, 1.0);
  }
  static double precision(double value) {
    return std::isfinite(value) ? std::numeric_limits<float>::epsilon() *
                                      std::max(1.0, std::abs(value))
                                : 0;
  }
  static juce::String text(juce::AudioPluginInstance &instance,
                           const Parameter &p, double raw) {
    const float value = (float)plain(instance, p, raw);
    if (p.flags & kAudioUnitParameterFlag_ValuesHaveStrings) {
      AudioUnitParameterStringFromValue conversion{p.id, &value, nullptr};
      UInt32 size = sizeof(conversion);
      const auto status = AudioUnitGetProperty(
          handle(instance), kAudioUnitProperty_ParameterStringFromValue,
          kAudioUnitScope_Global, 0, &conversion, &size);
      if (status == noErr && conversion.outString) {
        auto result = juce::String::fromCFString(conversion.outString);
        CFRelease(conversion.outString);
        return result;
      }
      // AU permits special display strings for only some values, e.g. -inf
      // at the minimum. Remaining values use their metadata's physical units.
      if (conversion.outString)
        CFRelease(conversion.outString);
    }
    if (p.unit == kAudioUnitParameterUnit_Indexed) {
      CFArrayRef labels = nullptr;
      UInt32 size = sizeof(labels);
      if (AudioUnitGetProperty(
              handle(instance), kAudioUnitProperty_ParameterValueStrings,
              kAudioUnitScope_Global, p.id, &labels, &size) == noErr &&
          labels) {
        const auto count = CFArrayGetCount(labels);
        const auto index =
            count ? std::clamp((CFIndex)std::llround(value - p.minimum),
                               (CFIndex)0, count - 1)
                  : 0;
        const auto result =
            count ? juce::String::fromCFString(static_cast<CFStringRef>(
                        CFArrayGetValueAtIndex(labels, index)))
                  : juce::String();
        CFRelease(labels);
        if (result.isNotEmpty())
          return result;
      }
    }
    // Metadata values already carry physical units. No inferred range or
    // setter.
    return juce::String(plain(instance, p, raw), 17);
  }
  static double fromText(juce::AudioPluginInstance &instance,
                         const Parameter &p, const juce::String &text) {
    auto string = text.toCFString();
    AudioUnitParameterValueFromString conversion{p.id, string, 0};
    UInt32 size = sizeof(conversion);
    const auto status = AudioUnitGetProperty(
        handle(instance), kAudioUnitProperty_ParameterValueFromString,
        kAudioUnitScope_Global, 0, &conversion, &size);
    CFRelease(string);
    if (status == noErr)
      return normalized(instance, p, conversion.outValue);
    // The shared converter still verifies the resulting actual display. This
    // handles plain physical values when an AU only names special endpoints;
    // nonlinear/ambiguous mappings must still pass the round-trip check.
    return normalized(instance, p, text.getDoubleValue());
  }
  static void setPrepared(juce::AudioPluginInstance &instance,
                          const Parameter &p, float raw) {
    const float value = p.minimum + (p.maximum - p.minimum) * raw;
    auto au = handle(instance);
    const auto status =
        AudioUnitSetParameter(au, p.id, kAudioUnitScope_Global, 0, value, 0);
    if (status != noErr)
      throw std::runtime_error("Audio Unit rejected an automation target: " +
                               std::to_string(status));
    // AUv3's bridge already delivers AUParameter updates. The legacy global
    // listener broadcast echoes back through the bridge, adding ~33ms per
    // target. Retain native physical ranges: JUCE's AU parameter can cache a
    // stale range after format changes, so its normalized setter is not safe.
    if (!p.isV3) {
      AudioUnitParameter parameter{au, p.id, kAudioUnitScope_Global, 0};
      AUParameterListenerNotify(nullptr, nullptr, &parameter);
    }
  }

  static void set(juce::AudioPluginInstance &instance, uint32_t id, float raw) {
    setPrepared(instance, describeById(instance, id), raw);
  }
};
} // namespace Pedalboard
#endif
