/* A deterministic VST3 host-test fixture, not a production audio effect.
 * Reads real IParameterChanges; supports input-clock interpolation or
 * last-point processing, fixed latency and a note-gated instrument.
 * GPL-3.0-only. */
#include "../hosted_transport_fixture.h"
#include "pluginterfaces/base/ibstream.h"
#include "pluginterfaces/base/ipluginbase.h"
#include "pluginterfaces/base/ustring.h"
#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/ivsteditcontroller.h"
#include "pluginterfaces/vst/ivstevents.h"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "pluginterfaces/vst/ivstprocesscontext.h"
#include "pluginterfaces/vst/vstspeaker.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace Steinberg;
using namespace Steinberg::Vst;
namespace {
const std::array<FUID, 16> ids = {
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A001),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A002),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A003),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A004),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A005),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A006),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A007),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A008),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A009),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A00A),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A00B),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A00C),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A00D),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A00E),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A00F),
    FUID(0x49CF8011, 0xE7504992, 0x81784123, 0xFF00A010)};
const char *names[] = {"Automation Probe",
                       "Automation Probe Last Point",
                       "Automation Probe Delayed",
                       "Automation Probe Instrument",
                       "Automation Probe Infinite Gain",
                       "Automation Probe Rate Dependent",
                       "Automation Probe Delayed Instrument",
                       "Automation Probe Long Delayed Instrument",
                       "Automation Probe Huge Delayed Instrument",
                       "Automation Probe Buses",
                       "Automation Probe Audio MIDI",
                       "Automation Probe Delayed Audio MIDI",
                       "Automation Probe Clock",
                       "Automation Probe Meter",
                       "Automation Probe Loop",
                       "Automation Probe Position"};
static_assert(std::size(names) == std::size(ids));
constexpr ParamID gainId = 0x1001, frequencyId = 0x314159, bypassId = 0xABC,
                  blocksId = 0x998, brokenId = 0x777, misparsedId = 0x999;
const ParamID parameterIds[] = {gainId,   frequencyId, bypassId,
                                blocksId, brokenId,    misparsedId};
void text(TChar *out, const char *in) { UString(out, 128).fromAscii(in); }
int parameterIndex(ParamID id) {
  for (int i = 0; i < 6; ++i)
    if (parameterIds[i] == id)
      return i;
  return -1;
}

bool hasBuses(int kind) { return kind >= 9 && kind <= 15; }

bool isInstrument(int kind) { return kind == 3 || (kind >= 6 && kind <= 8); }
int latency(int kind) {
  return kind == 2 || kind == 6 || kind == 11 ? 20
         : kind == 7                          ? 4097
         : kind == 8                          ? 16385
                                              : 0;
}

class Probe final : public IComponent,
                    public IAudioProcessor,
                    public IEditController,
                    public IMidiMapping {
public:
  explicit Probe(int kind) : kind(kind) {
    if (kind >= 10)
      ui[1] = dsp[1] = 1;
    if (kind == 4)
      ui[0] = dsp[0] = .5;
  }
  tresult PLUGIN_API queryInterface(const TUID iid, void **out) override {
    *out = nullptr;
    if (FUnknownPrivate::iidEqual(iid, IComponent::iid) ||
        FUnknownPrivate::iidEqual(iid, IPluginBase::iid) ||
        FUnknownPrivate::iidEqual(iid, FUnknown::iid))
      *out = static_cast<IComponent *>(this);
    else if (FUnknownPrivate::iidEqual(iid, IAudioProcessor::iid))
      *out = static_cast<IAudioProcessor *>(this);
    else if (FUnknownPrivate::iidEqual(iid, IEditController::iid))
      *out = static_cast<IEditController *>(this);
    if (kind >= 10 && FUnknownPrivate::iidEqual(iid, IMidiMapping::iid))
      *out = static_cast<IMidiMapping *>(this);
    if (!*out)
      return kNoInterface;
    addRef();
    return kResultOk;
  }
  uint32 PLUGIN_API addRef() override { return ++references; }
  uint32 PLUGIN_API release() override {
    const auto count = --references;
    if (!count)
      delete this;
    return count;
  }
  tresult PLUGIN_API initialize(FUnknown *) override { return kResultOk; }
  tresult PLUGIN_API terminate() override { return kResultOk; }
  tresult PLUGIN_API getControllerClassId(TUID) override {
    return kResultFalse;
  }
  tresult PLUGIN_API setIoMode(IoMode) override { return kResultOk; }
  int32 PLUGIN_API getBusCount(MediaType type,
                               BusDirection direction) override {
    if (hasBuses(kind))
      return type == kAudio ? 2 : 1;
    if (type == kAudio)
      return direction == kOutput || !isInstrument(kind) ? 1 : 0;
    return isInstrument(kind) ? 1 : 0;
  }
  tresult PLUGIN_API getBusInfo(MediaType type, BusDirection direction,
                                int32 index, BusInfo &info) override {
    if (index < 0 || index >= getBusCount(type, direction))
      return kInvalidArgument;
    info = {};
    info.mediaType = type;
    info.direction = direction;
    info.channelCount =
        type == kAudio
            ? (hasBuses(kind) ? busChannels[direction][index] : channels)
            : 16;
    info.busType = index ? kAux : kMain;
    info.flags = BusInfo::kDefaultActive;
    text(info.name,
         type == kAudio
             ? (index ? (direction == kInput ? "External Key" : "Aux Output")
                      : "Audio")
             : "Notes");
    return kResultOk;
  }
  tresult PLUGIN_API getRoutingInfo(RoutingInfo &, RoutingInfo &) override {
    return kNotImplemented;
  }
  tresult PLUGIN_API activateBus(MediaType type, BusDirection, int32 index,
                                 TBool enabled) override {
    if (hasBuses(kind) && type == kAudio && index == 0 && !enabled)
      return kResultFalse;
    return kResultOk;
  }
  tresult PLUGIN_API setActive(TBool active) override {
    if (!active)
      clear();
    return kResultOk;
  }
  tresult PLUGIN_API setState(IBStream *state) override {
    if (!state)
      return kResultFalse;
    std::array<double, 6> values{};
    int32 count = 0;
    if (state->read(values.data(), sizeof(values), &count) != kResultOk ||
        count != sizeof(values))
      return kResultFalse;
    ui = dsp = values;
    return kResultOk;
  }
  tresult PLUGIN_API getState(IBStream *state) override {
    return state ? state->write(dsp.data(), sizeof(dsp), nullptr)
                 : kResultFalse;
  }
  tresult PLUGIN_API setComponentState(IBStream *state) override {
    return setState(state);
  }
  tresult PLUGIN_API setBusArrangements(SpeakerArrangement *inputs,
                                        int32 numInputs,
                                        SpeakerArrangement *outputs,
                                        int32 numOutputs) override {
    if (hasBuses(kind)) {
      if (numInputs != 2 || numOutputs != 2 || inputs[0] != outputs[0])
        return kResultFalse;
      for (int direction = 0; direction < 2; ++direction)
        for (int index = 0; index < 2; ++index) {
          const auto arrangement = (direction ? outputs : inputs)[index];
          const int count = SpeakerArr::getChannelCount(arrangement);
          if (count > 2 || (!index && count < 1))
            return kResultFalse;
        }
      for (int direction = 0; direction < 2; ++direction)
        for (int index = 0; index < 2; ++index)
          busChannels[direction][index] = SpeakerArr::getChannelCount(
              (direction ? outputs : inputs)[index]);
      return kResultOk;
    }
    if (numOutputs != 1 || numInputs != (isInstrument(kind) ? 0 : 1))
      return kResultFalse;
    const auto count = SpeakerArr::getChannelCount(outputs[0]);
    if (count < 1 || count > 2 ||
        (!isInstrument(kind) && inputs[0] != outputs[0]))
      return kResultFalse;
    channels = count;
    return kResultOk;
  }
  tresult PLUGIN_API getBusArrangement(BusDirection direction, int32 index,
                                       SpeakerArrangement &arr) override {
    if (hasBuses(kind)) {
      if (index < 0 || index > 1)
        return kInvalidArgument;
      const int count = busChannels[direction][index];
      arr = count == 0   ? SpeakerArr::kEmpty
            : count == 1 ? SpeakerArr::kMono
                         : SpeakerArr::kStereo;
      return kResultOk;
    }
    if (index != 0 || (isInstrument(kind) && direction == kInput))
      return kResultFalse;
    arr = channels == 1 ? SpeakerArr::kMono : SpeakerArr::kStereo;
    return kResultOk;
  }
  tresult PLUGIN_API canProcessSampleSize(int32 size) override {
    return size == kSample32 ? kResultTrue : kResultFalse;
  }
  uint32 PLUGIN_API getLatencySamples() override { return latency(kind); }
  tresult PLUGIN_API setupProcessing(ProcessSetup &setup) override {
    maxBlockSize = setup.maxSamplesPerBlock;
    if (kind == 5)
      frequencyMaximum = setup.sampleRate * .5;
    clear();
    return kResultOk;
  }
  tresult PLUGIN_API setProcessing(TBool state) override {
    if (!state)
      clear();
    return kResultOk;
  }
  uint32 PLUGIN_API getTailSamples() override { return 0; }

  tresult PLUGIN_API process(ProcessData &data) override {
    if (data.numSamples > maxBlockSize)
      return kInvalidArgument;
    ++blocks;
    struct Cursor {
      IParamValueQueue *queue = nullptr;
      int32 point = 0, previousOffset = -1, nextOffset = 0;
      double previous = 0, next = 0;
      bool hasNext = false;
      void advance() {
        hasNext = queue && point < queue->getPointCount() &&
                  queue->getPoint(point++, nextOffset, next) == kResultOk;
      }
      double at(int32 frame, bool discrete = false) {
        while (hasNext && nextOffset <= frame) {
          previous = next;
          previousOffset = nextOffset;
          advance();
        }
        return hasNext && !discrete
                   ? previous + (next - previous) * (frame - previousOffset) /
                                    (nextOffset - previousOffset)
                   : previous;
      }
    };
    std::array<Cursor, 6> controls;
    for (int p = 0; p < 6; ++p)
      controls[p].previous = dsp[p];
    if (data.inputParameterChanges) {
      for (int32 i = 0; i < data.inputParameterChanges->getParameterCount();
           ++i) {
        auto *queue = data.inputParameterChanges->getParameterData(i);
        const int p = parameterIndex(queue->getParameterId());
        if (p < 0)
          continue;
        if (kind == 1) {
          int32 ignored;
          double value;
          if (queue->getPoint(queue->getPointCount() - 1, ignored, value) ==
              kResultOk)
            controls[p].previous = value;
        } else {
          controls[p].queue = queue;
          controls[p].advance();
        }
      }
    }
    Event nextEvent{};
    int32 eventIndex = 0;
    const int32 eventCount =
        data.inputEvents ? data.inputEvents->getEventCount() : 0;
    bool hasEvent =
        eventCount && data.inputEvents->getEvent(0, nextEvent) == kResultOk;
    for (int32 i = 0; i < data.numSamples; ++i) {
      while (hasEvent && nextEvent.sampleOffset <= i) {
        if (nextEvent.type == Event::kNoteOnEvent)
          note = nextEvent.noteOn.velocity > 0;
        if (nextEvent.type == Event::kNoteOffEvent)
          note = false;
        if (kind >= 10 && data.outputEvents &&
            (nextEvent.type == Event::kNoteOnEvent ||
             nextEvent.type == Event::kNoteOffEvent)) {
          auto emitted = nextEvent;
          if (emitted.type == Event::kNoteOnEvent)
            emitted.noteOn.pitch += 12;
          else
            emitted.noteOff.pitch += 12;
          data.outputEvents->addEvent(emitted);
        }
        hasEvent =
            ++eventIndex < eventCount &&
            data.inputEvents->getEvent(eventIndex, nextEvent) == kResultOk;
      }
      for (int p = 0; p < 6; ++p)
        dsp[p] = controls[p].at(i, kind >= 10 && p == 2);
      const double gain =
          kind == 4
              ? 2 * dsp[0]
              : std::pow(10.0, normalizedParamToPlain(gainId, dsp[0]) / 20.0);
      const double frequency = normalizedParamToPlain(frequencyId, dsp[1]);
      if (kind >= 12) {
        const auto *time = data.processContext;
        std::array<float, 4> values{-1, -1, -1, -1};
        if (time) {
          TransportObservation observed;
          observed.playing = time->state & ProcessContext::kPlaying;
          observed.tempo = time->tempo;
          observed.ppq = time->projectTimeMusic;
          observed.sample = time->projectTimeSamples;
          observed.rate = time->sampleRate;
          observed.bar = time->barPositionMusic;
          observed.numerator = time->timeSigNumerator;
          observed.denominator = time->timeSigDenominator;
          observed.start = time->cycleStartMusic;
          observed.end = time->cycleEndMusic;
          observed.flags =
              (observed.playing ? 1 : 0) |
              ((time->state & ProcessContext::kRecording) ? 2 : 0) |
              ((time->state & ProcessContext::kCycleActive) ? 4 : 0) |
              ((time->state & ProcessContext::kTempoValid) ? 8 : 0) |
              ((time->state & ProcessContext::kProjectTimeMusicValid) ? 16
                                                                      : 0) |
              ((time->state & ProcessContext::kBarPositionValid) ? 32 : 0) |
              ((time->state & ProcessContext::kCycleValid) ? 64 : 0) |
              ((time->state & ProcessContext::kTimeSigValid) ? 128 : 0);
          observed.nextBeat = (uint32_t)std::ceil(
              (std::ceil(time->projectTimeMusic) - time->projectTimeMusic) *
              60 * time->sampleRate / time->tempo);
          values = observed.audio(kind - 12, i);
        }
        for (int bus = 0; bus < data.numOutputs && bus < 2; ++bus)
          for (int c = 0; c < data.outputs[bus].numChannels && c < 2; ++c)
            data.outputs[bus].channelBuffers32[c][i] = values[bus * 2 + c];
        continue;
      }
      if (hasBuses(kind)) {
        const float key = data.numInputs > 1 && data.inputs[1].numChannels
                              ? data.inputs[1].channelBuffers32[0][i]
                              : 0;
        const float duck = std::abs(key) > .5f ? .25f : 1.0f;
        // Read the source before writing any possibly aliased auxiliary output.
        std::array<float, 2> source{};
        for (int c = 0; c < data.inputs[0].numChannels; ++c)
          source[c] = data.inputs[0].channelBuffers32[c][i];
        if (data.numOutputs > 1)
          for (int c = 0; c < data.outputs[1].numChannels; ++c)
            data.outputs[1].channelBuffers32[c][i] =
                .5f * source[c % data.inputs[0].numChannels];
        for (int c = 0; c < data.outputs[0].numChannels; ++c) {
          float value = kind >= 10 && dsp[2] >= .5 ? source[c]
                        : kind == 9
                            ? source[c] * duck
                            : (float)(source[c] * duck * (note ? 1 : 0) * gain *
                                      frequency / 20000.0);
          if (latency(kind)) {
            const float old = delay[c][position];
            delay[c][position] = value;
            value = old;
          }
          data.outputs[0].channelBuffers32[c][i] = value;
        }
        position = (position + 1) % std::max(20, latency(kind));
        continue;
      }
      for (int32 c = 0; data.numOutputs && c < data.outputs[0].numChannels;
           ++c) {
        float input = isInstrument(kind)
                          ? (note ? 1.0f : 0.0f)
                          : data.inputs[0].channelBuffers32[c][i];
        float output =
            dsp[2] >= .5 ? input : (float)(input * gain * frequency / 20000.0);
        if (latency(kind)) {
          const float old = delay[c][position];
          delay[c][position] = output;
          output = old;
        }
        data.outputs[0].channelBuffers32[c][i] = output;
      }
      position = (position + 1) % std::max(20, latency(kind));
    }
    // Returned MIDI must never become input to a later host sub-block. An
    // accidental reuse of these note-offs would cut subsequent held notes.
    if ((isInstrument(kind) || kind >= 10) && data.outputEvents &&
        data.numSamples > 0) {
      Event event{};
      event.type = Event::kNoteOffEvent;
      event.noteOff.pitch = 60;
      event.noteOff.noteId = -1;
      data.outputEvents->addEvent(event);
    }
    return kResultOk;
  }

  tresult PLUGIN_API getMidiControllerAssignment(int32 bus, int16 channel,
                                                 CtrlNumber number,
                                                 ParamID &id) override {
    if (kind < 10 || bus != 0 || channel != 0 || number != 1)
      return kResultFalse;
    id = bypassId;
    return kResultOk;
  }

  int32 PLUGIN_API getParameterCount() override { return 6; }
  tresult PLUGIN_API getParameterInfo(int32 index,
                                      ParameterInfo &info) override {
    if (index < 0 || index >= 6)
      return kInvalidArgument;
    info = {};
    info.id = parameterIds[index];
    info.defaultNormalizedValue =
        index == 0 ? (kind == 4 ? .5 : 60.0 / 72.0) : 0;
    const char *titles[] = {"Gain",   "Frequency", "Bypass",
                            "Blocks", "Broken",    "Misparsed"};
    text(info.title, titles[index]);
    text(info.shortTitle, titles[index]);
    text(info.units, index == 0 ? "dB" : index == 1 ? "Hz" : "");
    info.flags =
        index == 3 ? ParameterInfo::kIsReadOnly : ParameterInfo::kCanAutomate;
    if (index == 2) {
      info.stepCount = 1;
      info.flags |= ParameterInfo::kIsBypass;
    }
    return kResultOk;
  }
  tresult PLUGIN_API getParamStringByValue(ParamID id, ParamValue value,
                                           String128 out) override {
    const int index = parameterIndex(id);
    if (index < 0)
      return kInvalidArgument;
    char buffer[96];
    if (id == blocksId)
      std::snprintf(buffer, sizeof(buffer), "%llu", (unsigned long long)blocks);
    else if (id == brokenId)
      std::snprintf(buffer, sizeof(buffer), "unsupported");
    else
      std::snprintf(buffer, sizeof(buffer), "%.6f",
                    id == gainId && kind == 4
                        ? 20 * std::log10(2 * value)
                        : normalizedParamToPlain(id, value));
    text(out, buffer);
    return kResultOk;
  }
  tresult PLUGIN_API getParamValueByString(ParamID id, TChar *in,
                                           ParamValue &value) override {
    if (id == brokenId)
      return kNotImplemented;
    if (id == misparsedId) {
      value = 0;
      return kResultOk;
    }
    char buffer[128]{};
    UString(in, 128).toAscii(buffer, sizeof(buffer));
    char *end = nullptr;
    const double plain = std::strtod(buffer, &end);
    if (end == buffer || !std::isfinite(plain))
      return kResultFalse;
    value = plainParamToNormalized(
        id, id == gainId && kind == 4 ? std::pow(10.0, plain / 20) : plain);
    return kResultOk;
  }
  ParamValue PLUGIN_API normalizedParamToPlain(ParamID id,
                                               ParamValue value) override {
    if (id == gainId && kind == 4)
      return 2 * value;
    return id == gainId        ? -60 + 72 * value
           : id == frequencyId ? 20 * std::pow(frequencyMaximum / 20.0, value)
                               : value;
  }
  ParamValue PLUGIN_API plainParamToNormalized(ParamID id,
                                               ParamValue value) override {
    if (id == gainId && kind == 4)
      return std::clamp(value / 2, 0.0, 1.0);
    if (id == misparsedId)
      return 0;
    return std::clamp(id == gainId ? (value + 60) / 72
                      : id == frequencyId
                          ? std::log(std::max(20.0, value) / 20) /
                                std::log(frequencyMaximum / 20.0)
                          : value,
                      0.0, 1.0);
  }
  ParamValue PLUGIN_API getParamNormalized(ParamID id) override {
    const int i = parameterIndex(id);
    return i < 0 ? 0 : ui[i];
  }
  tresult PLUGIN_API setParamNormalized(ParamID id, ParamValue value) override {
    const int i = parameterIndex(id);
    if (i < 0)
      return kInvalidArgument;
    ui[i] = value;
    return kResultOk;
  }
  tresult PLUGIN_API setComponentHandler(IComponentHandler *) override {
    return kResultOk;
  }
  IPlugView *PLUGIN_API createView(FIDString) override { return nullptr; }

private:
  void clear() {
    for (auto &channel : delay)
      channel.fill(0);
    position = 0;
    note = false;
    blocks = 0;
  }
  std::atomic<uint32> references{1};
  int kind, channels = 2, position = 0, maxBlockSize = 8192;
  int busChannels[2][2] = {{2, 2}, {2, 2}};
  bool note = false;
  uint64 blocks = 0;
  double frequencyMaximum = 20000;
  std::array<double, 6> ui{60.0 / 72.0, 0, 0, 0, 0, 0}, dsp{ui};
  std::array<std::array<float, 16385>, 2> delay{};
};

class Factory final : public IPluginFactory2 {
public:
  tresult PLUGIN_API queryInterface(const TUID iid, void **out) override {
    *out = nullptr;
    if (FUnknownPrivate::iidEqual(iid, IPluginFactory::iid) ||
        FUnknownPrivate::iidEqual(iid, IPluginFactory2::iid) ||
        FUnknownPrivate::iidEqual(iid, FUnknown::iid))
      *out = static_cast<IPluginFactory2 *>(this);
    if (!*out)
      return kNoInterface;
    addRef();
    return kResultOk;
  }
  uint32 PLUGIN_API addRef() override { return ++references; }
  uint32 PLUGIN_API release() override {
    const auto count = --references;
    if (!count)
      delete this;
    return count;
  }
  tresult PLUGIN_API getFactoryInfo(PFactoryInfo *info) override {
    if (!info)
      return kInvalidArgument;
    *info =
        PFactoryInfo("OpenMIRLab", "https://github.com/openmirlab/pedalboard",
                     "bernie40916@gmail.com", 0);
    return kResultOk;
  }
  int32 PLUGIN_API countClasses() override { return (int32)ids.size(); }
  tresult PLUGIN_API getClassInfo(int32 index, PClassInfo *info) override {
    if (index < 0 || index >= (int32)ids.size() || !info)
      return kInvalidArgument;
    *info = PClassInfo(ids[index].toTUID(), PClassInfo::kManyInstances,
                       kVstAudioEffectClass, names[index]);
    return kResultOk;
  }
  tresult PLUGIN_API getClassInfo2(int32 index, PClassInfo2 *info) override {
    if (index < 0 || index >= (int32)ids.size() || !info)
      return kInvalidArgument;
    *info = {};
    ids[index].toTUID(info->cid);
    info->cardinality = PClassInfo::kManyInstances;
    std::strncpy(info->category, kVstAudioEffectClass,
                 sizeof(info->category) - 1);
    std::strncpy(info->name, names[index], sizeof(info->name) - 1);
    std::strncpy(info->subCategories,
                 isInstrument(index) ? "Instrument|Synth" : "Fx",
                 sizeof(info->subCategories) - 1);
    std::strncpy(info->vendor, "OpenMIRLab", sizeof(info->vendor) - 1);
    std::strncpy(info->version, "1.0.0", sizeof(info->version) - 1);
    std::strncpy(info->sdkVersion, "VST 3", sizeof(info->sdkVersion) - 1);
    return kResultOk;
  }
  tresult PLUGIN_API createInstance(FIDString cid, FIDString iid,
                                    void **out) override {
    for (int i = 0; i < (int32)ids.size(); ++i)
      if (std::memcmp(cid, ids[i].toTUID(), 16) == 0) {
        auto *probe = new Probe(i);
        const auto result = probe->queryInterface(iid, out);
        probe->release();
        return result;
      }
    *out = nullptr;
    return kNoInterface;
  }

private:
  std::atomic<uint32> references{1};
};
} // namespace

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
extern "C" EXPORT IPluginFactory *GetPluginFactory() { return new Factory(); }
extern "C" EXPORT bool bundleEntry(void *) { return true; }
extern "C" EXPORT bool bundleExit() { return true; }
extern "C" EXPORT bool ModuleEntry(void *) { return true; }
extern "C" EXPORT bool ModuleExit() { return true; }
extern "C" EXPORT bool InitDll() { return true; }
extern "C" EXPORT bool ExitDll() { return true; }
