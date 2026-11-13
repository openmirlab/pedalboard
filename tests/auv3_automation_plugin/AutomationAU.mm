/* Original AUv3 diagnostic extension, GPL-3.0-only. */
#include "../hosted_transport_fixture.h"
#import <AVFoundation/AVFoundation.h>
#import <AudioToolbox/AudioToolbox.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <unistd.h>

namespace {
constexpr AUParameterAddress gainAddress = 0x1001;
constexpr AUParameterAddress frequencyAddress = 0x100000002ULL;
constexpr AUParameterAddress passAddress = 0xABC;
constexpr AUParameterAddress pidAddress = 0x998;
constexpr AUParameterAddress instanceAddress = 0x999;
constexpr AUParameterAddress identityAddress = 0xFA;
constexpr AUParameterAddress lowAddress = 2;
constexpr AUParameterAddress framesAddress = 0x99A, capacityAddress = 0x99B;
constexpr AUParameterAddress busMaskAddress = 0x99C;
std::atomic<unsigned> nextInstance{1};
struct DSP {
  std::atomic<float> gain{0}, frequency{20}, pass{0}, identity{0}, decoy{0};
  std::atomic<unsigned> renderFrames{0}, renderCapacity{0};
  std::atomic<unsigned> renderBusMask{0};
  bool note = false;
  unsigned position = 0;
  std::array<std::array<float, 16385>, 2> delay{};
  std::array<std::array<float, 8192>, 4> busInput{};
  void clear() {
    renderBusMask.store(0);
    note = false;
    position = 0;
    for (auto &channel : delay)
      channel.fill(0);
  }
};
inline void applyEvent(DSP *dsp, const AURenderEvent *events,
                       bool controllers = false) {
  if (events->head.eventType == AURenderEventParameter) {
    const auto &p = events->parameter;
    switch (p.parameterAddress) {
    case gainAddress:
      dsp->gain.store(p.value);
      break;
    case frequencyAddress:
      dsp->frequency.store(p.value);
      break;
    case passAddress:
      dsp->pass.store(p.value);
      break;
    case identityAddress:
      dsp->identity.store(p.value);
      break;
    case lowAddress:
      dsp->decoy.store(p.value);
      break;
    default:
      break;
    }
  } else if (events->head.eventType == AURenderEventMIDI) {
    const auto *data = events->MIDI.data;
    if ((data[0] & 0xf0) == 0x90)
      dsp->note = data[2] > 0;
    if ((data[0] & 0xf0) == 0x80)
      dsp->note = false;
    if (controllers && events->MIDI.length >= 3 && (data[0] & 0xf0) == 0xb0 &&
        data[1] == 1)
      dsp->pass.store(data[2] >= 64 ? 1 : 0);
  }
}
} // namespace

@interface OMAutomationAU : AUAudioUnit
@end
@implementation OMAutomationAU {
  AUAudioUnitBusArray *_inputs, *_outputs;
  AUParameterTree *_parameters;
  DSP _dsp;
  unsigned _instance;
  BOOL _instrument;
  BOOL _multipleBuses, _busIdentity, _audioMidi;
  int _clockMode;
  unsigned _latency;
}
- (instancetype)
    initWithComponentDescription:(AudioComponentDescription)description
                         options:(AudioComponentInstantiationOptions)options
                           error:(NSError **)error {
  self = [super initWithComponentDescription:description
                                     options:options
                                       error:error];
  if (!self)
    return nil;
  _instrument = description.componentType == kAudioUnitType_MusicDevice;
  const auto variant = description.componentSubType & 255;
  _clockMode = variant >= 'a' && variant <= 'd' ? (int)(variant - 'a') : -1;
  _audioMidi = variant == '8' || variant == '9';
  _multipleBuses =
      variant == '6' || variant == '7' || _audioMidi || _clockMode >= 0;
  _busIdentity = variant == '7';
  _latency = variant == '1' || variant == '3' || variant == '9' ? 20
             : variant == '4'                                   ? 4097
             : variant == '5'                                   ? 16385
                                                                : 0;
  _instance = nextInstance.fetch_add(1);
  AVAudioFormat *format =
      [[AVAudioFormat alloc] initStandardFormatWithSampleRate:44100 channels:2];
  AUAudioUnitBus *input = [[AUAudioUnitBus alloc] initWithFormat:format
                                                           error:error];
  AUAudioUnitBus *output = [[AUAudioUnitBus alloc] initWithFormat:format
                                                            error:error];
  if (!input || !output)
    return nil;
  AUAudioUnitBus *key = [[AUAudioUnitBus alloc] initWithFormat:format
                                                         error:error];
  AUAudioUnitBus *aux = [[AUAudioUnitBus alloc] initWithFormat:format
                                                         error:error];
  _inputs = [[AUAudioUnitBusArray alloc]
      initWithAudioUnit:self
                busType:AUAudioUnitBusTypeInput
                 busses:_instrument      ? @[]
                        : _multipleBuses ? @[ input, key ]
                                         : @[ input ]];
  _outputs = [[AUAudioUnitBusArray alloc]
      initWithAudioUnit:self
                busType:AUAudioUnitBusTypeOutput
                 busses:_multipleBuses ? @[ output, aux ] : @[ output ]];
  auto make = ^AUParameter *(
      NSString *identifier, NSString *name, AUParameterAddress address,
      AUValue low, AUValue high, AudioUnitParameterUnit unit, BOOL writable) {
    return [AUParameterTree
        createParameterWithIdentifier:identifier
                                 name:name
                              address:address
                                  min:low
                                  max:high
                                 unit:unit
                             unitName:nil
                                flags:kAudioUnitParameterFlag_IsReadable |
                                      (writable
                                           ? kAudioUnitParameterFlag_IsWritable
                                           : 0)valueStrings:nil
                  dependentParameters:nil];
  };
  AUParameter *gain = make(@"gain", @"Gain", gainAddress, -60, 12,
                           kAudioUnitParameterUnit_Decibels, YES);
  AUParameter *frequency = make(@"frequency", @"Frequency", frequencyAddress,
                                20, 20000, kAudioUnitParameterUnit_Hertz, YES);
  AUParameter *pass = make(@"pass", @"Pass Through", passAddress, 0, 1,
                           kAudioUnitParameterUnit_Boolean, YES);
  AUParameter *pid = make(@"render_pid", @"Render PID", pidAddress, 0, 1e9,
                          kAudioUnitParameterUnit_Generic, NO);
  AUParameter *instance = make(@"instance_id", @"Instance ID", instanceAddress,
                               0, 1e9, kAudioUnitParameterUnit_Generic, NO);
  AUParameter *identity =
      make(@"identity_probe", @"Identity Probe", identityAddress, 0, 1,
           kAudioUnitParameterUnit_Boolean, YES);
  AUParameter *decoy =
      make(@"low_address_decoy", @"Low Address Decoy", lowAddress, 0, 1,
           kAudioUnitParameterUnit_Boolean, YES);
  AUParameter *frames = make(@"render_frames", @"Render Frames", framesAddress,
                             0, 1e9, kAudioUnitParameterUnit_Generic, NO);
  AUParameter *capacity =
      make(@"render_capacity", @"Render Capacity", capacityAddress, 0, 1e9,
           kAudioUnitParameterUnit_Generic, NO);
  AUParameter *busMask =
      make(@"render_bus_mask", @"Render Bus Mask", busMaskAddress, 0, 3,
           kAudioUnitParameterUnit_Generic, NO);
  identity.value = decoy.value = 0;
  gain.value = 0;
  frequency.value = _audioMidi ? 20000 : 20;
  _dsp.frequency.store(_audioMidi ? 20000 : 20);
  pass.value = 0;
  pid.value = getpid();
  instance.value = _instance;
  NSMutableArray *children = [@[
    gain, frequency, pass, pid, instance, identity, decoy, frames, capacity
  ] mutableCopy];
  if (_multipleBuses)
    [children addObject:busMask];
  _parameters = [AUParameterTree createTreeWithChildren:children];
  DSP *dsp = &_dsp;
  const float creatorPID = getpid(), instanceID = _instance;
  _parameters.implementorValueObserver =
      ^(AUParameter *parameter, AUValue value) {
        switch (parameter.address) {
        case gainAddress:
          dsp->gain.store(value);
          break;
        case frequencyAddress:
          dsp->frequency.store(value);
          break;
        case passAddress:
          dsp->pass.store(value);
          break;
        case identityAddress:
          dsp->identity.store(value);
          break;
        case lowAddress:
          dsp->decoy.store(value);
          break;
        default:
          break;
        }
      };
  _parameters.implementorValueProvider = ^AUValue(AUParameter *parameter) {
    switch (parameter.address) {
    case gainAddress:
      return dsp->gain.load();
    case frequencyAddress:
      return dsp->frequency.load();
    case passAddress:
      return dsp->pass.load();
    case identityAddress:
      return dsp->identity.load();
    case lowAddress:
      return dsp->decoy.load();
    case pidAddress:
      return creatorPID;
    case instanceAddress:
      return instanceID;
    case framesAddress:
      return dsp->renderFrames.load();
    case capacityAddress:
      return dsp->renderCapacity.load();
    case busMaskAddress:
      return dsp->renderBusMask.load();
    default:
      return 0;
    }
  };
  _parameters.implementorStringFromValueCallback =
      ^NSString *(AUParameter *parameter, const AUValue *value) {
        return [NSString
            stringWithFormat:@"%.9f", value ? *value : parameter.value];
      };
  self.maximumFramesToRender = 8192;
  return self;
}
- (AUAudioUnitBusArray *)inputBusses {
  return _inputs;
}
- (AUAudioUnitBusArray *)outputBusses {
  return _outputs;
}
- (AUParameterTree *)parameterTree {
  return _parameters;
}
- (NSArray<NSNumber *> *)channelCapabilities {
  if (_multipleBuses)
    return @[ @1, @1, @1, @2, @2, @1, @2, @2 ];
  return _instrument ? @[ @0, @1, @0, @2 ] : @[ @1, @1, @2, @2 ];
}
- (BOOL)canProcessInPlace {
  return !_multipleBuses;
}
- (NSTimeInterval)latency {
  return (double)_latency / _outputs[0].format.sampleRate;
}
- (BOOL)allocateRenderResourcesAndReturnError:(NSError **)error {
  if (![super allocateRenderResourcesAndReturnError:error])
    return NO;
  _dsp.clear();
  return YES;
}
- (void)reset {
  [super reset];
  _dsp.clear();
}
- (NSDictionary *)fullState {
  NSMutableDictionary *state =
      [[super fullState] mutableCopy] ?: [NSMutableDictionary dictionary];
  state[@"probe_gain"] = @(_dsp.gain.load());
  state[@"probe_frequency"] = @(_dsp.frequency.load());
  state[@"probe_pass"] = @(_dsp.pass.load());
  return state;
}
- (void)setFullState:(NSDictionary *)state {
  [super setFullState:state];
  if (state[@"probe_gain"])
    _dsp.gain.store([state[@"probe_gain"] floatValue]);
  if (state[@"probe_frequency"])
    _dsp.frequency.store([state[@"probe_frequency"] floatValue]);
  if (state[@"probe_pass"])
    _dsp.pass.store([state[@"probe_pass"] floatValue]);
}
- (AUInternalRenderBlock)internalRenderBlock {
  DSP *dsp = &_dsp;
  const BOOL instrument = _instrument;
  const int clockMode = _clockMode;
  __unsafe_unretained OMAutomationAU *unit = self;
  const double renderRate = _outputs[0].format.sampleRate;
  const BOOL multipleBuses = _multipleBuses;
  const BOOL busIdentity = _busIdentity;
  const BOOL audioMidi = _audioMidi;
  const unsigned mainChannels =
      _instrument ? 0 : _inputs[0].format.channelCount;
  const unsigned keyChannels =
      _multipleBuses ? _inputs[1].format.channelCount : 0;
  const unsigned latency = _latency;
  const AUAudioFrameCount capacity = self.maximumFramesToRender;
  return ^AUAudioUnitStatus(
      AudioUnitRenderActionFlags *flags, const AudioTimeStamp *timestamp,
      AUAudioFrameCount frames, NSInteger bus, AudioBufferList *output,
      const AURenderEvent *events, AURenderPullInputBlock pull) {
    dsp->renderFrames.store(frames);
    dsp->renderCapacity.store(capacity);
    dsp->renderBusMask.fetch_or(1u << bus);
    if (frames > capacity)
      return (AUAudioUnitStatus)kAudioUnitErr_TooManyFramesToProcess;
    if (clockMode >= 0) {
      TransportObservation observed;
      observed.rate = renderRate;
      AUHostMusicalContextBlock musical = unit.musicalContextBlock;
      AUHostTransportStateBlock transport = unit.transportStateBlock;
      double num = 0;
      NSInteger den = 0, delta = 0;
      const bool musicOK =
          musical && musical(&observed.tempo, &num, &den, &observed.ppq, &delta,
                             &observed.bar);
      observed.numerator = num;
      observed.denominator = (int)den;
      observed.nextBeat = (uint32_t)delta;
      AUHostTransportStateFlags state = 0;
      double frame = 0;
      const bool transportOK =
          transport &&
          transport(&state, &frame, &observed.start, &observed.end);
      observed.sample = (int64_t)frame;
      observed.playing = state & AUHostTransportStateMoving;
      observed.changed = state & AUHostTransportStateChanged;
      observed.flags = (observed.playing ? 1 : 0) |
                       ((state & AUHostTransportStateRecording) ? 2 : 0) |
                       ((state & AUHostTransportStateCycling) ? 4 : 0) |
                       (musicOK ? 8 + 16 + 32 + 128 : 0) |
                       (transportOK && observed.end > observed.start ? 64 : 0);
      for (UInt32 i = 0; i < frames; ++i) {
        if (!bus) {
          while (events &&
                 (events->head.eventSampleTime == AUEventSampleTimeImmediate ||
                  events->head.eventSampleTime <=
                      (AUEventSampleTime)timestamp->mSampleTime + i)) {
            applyEvent(dsp, events, true);
            events = events->head.next;
          }
        }
        const auto values = observed.audio(clockMode, i);
        for (UInt32 c = 0; c < output->mNumberBuffers && c < 2; ++c)
          static_cast<float *>(output->mBuffers[c].mData)[i] =
              values[bus * 2 + c];
      }
      return (AUAudioUnitStatus)noErr;
    }
    if (multipleBuses) {
      if (!pull || frames > 8192 || mainChannels > 2 || keyChannels > 2 ||
          bus > 1)
        return (AUAudioUnitStatus)kAudioUnitErr_FormatNotSupported;
      struct TwoBuffers {
        UInt32 mNumberBuffers;
        AudioBuffer mBuffers[2];
      } sources[2]{};
      for (unsigned index = 0; index < 2; ++index) {
        auto &source = sources[index];
        source.mNumberBuffers = index ? keyChannels : mainChannels;
        for (unsigned c = 0; c < source.mNumberBuffers; ++c) {
          source.mBuffers[c].mNumberChannels = 1;
          source.mBuffers[c].mDataByteSize = frames * sizeof(float);
          source.mBuffers[c].mData = dsp->busInput[index * 2 + c].data();
        }
        const auto status = pull(flags, timestamp, frames, index,
                                 reinterpret_cast<AudioBufferList *>(&source));
        if (status != noErr)
          return status;
      }
      for (UInt32 i = 0; i < frames; ++i) {
        if (audioMidi && !bus) {
          while (events &&
                 (events->head.eventSampleTime == AUEventSampleTimeImmediate ||
                  events->head.eventSampleTime <=
                      (AUEventSampleTime)timestamp->mSampleTime + i)) {
            applyEvent(dsp, events, audioMidi);
            events = events->head.next;
          }
        }
        const double gain =
            audioMidi ? std::pow(10.0, dsp->gain.load() / 20.0) : 1;
        const double frequency = audioMidi ? dsp->frequency.load() : 20000;
        for (UInt32 c = 0; c < output->mNumberBuffers; ++c) {
          auto *target = static_cast<float *>(output->mBuffers[c].mData);
          const auto *source = static_cast<const float *>(
              sources[0].mBuffers[c % mainChannels].mData);
          const auto *keySource =
              static_cast<const float *>(sources[1].mBuffers[0].mData);
          float value =
              busIdentity                           ? (float)getpid()
              : bus                                 ? source[i] * .5f
              : audioMidi && dsp->pass.load() >= .5 ? source[i]
              : audioMidi
                  ? (float)(source[i] *
                            (std::abs(keySource[i]) > .5f ? .25f : 1.f) *
                            (dsp->note ? 1 : 0) * gain * frequency / 20000.0)
                  : source[i] * (std::abs(keySource[i]) > .5f ? .25f : 1.f);
          if (latency && !bus) {
            const float old = dsp->delay[c][dsp->position];
            dsp->delay[c][dsp->position] = value;
            value = old;
          }
          target[i] = value;
        }
        if (audioMidi && !bus)
          dsp->position = (dsp->position + 1) % std::max(20u, latency);
      }
      return (AUAudioUnitStatus)noErr;
    }
    if (!instrument) {
      if (!pull)
        return (AUAudioUnitStatus)kAudioUnitErr_NoConnection;
      const auto status = pull(flags, timestamp, frames, 0, output);
      if (status != noErr)
        return status;
    }
    for (AUAudioFrameCount i = 0; i < frames; ++i) {
      while (events &&
             (events->head.eventSampleTime == AUEventSampleTimeImmediate ||
              events->head.eventSampleTime <=
                  (AUEventSampleTime)timestamp->mSampleTime + i)) {
        applyEvent(dsp, events);
        events = events->head.next;
      }
      const double gain = std::pow(10.0, dsp->gain.load() / 20.0);
      const double frequency = dsp->frequency.load();
      for (UInt32 c = 0; c < output->mNumberBuffers; ++c) {
        auto *samples = static_cast<float *>(output->mBuffers[c].mData);
        if (!samples || c >= 2)
          return (AUAudioUnitStatus)kAudioUnitErr_TooManyFramesToProcess;
        const float input = instrument ? (dsp->note ? 1.0f : 0.0f) : samples[i];
        float result = dsp->pass.load() >= .5
                           ? input
                           : (float)(input * gain * frequency / 20000.0);
        if (latency) {
          const float previous = dsp->delay[c][dsp->position];
          dsp->delay[c][dsp->position] = result;
          result = previous;
        }
        samples[i] = dsp->identity.load() >= .5 ? (float)getpid() : result;
      }
      dsp->position = (dsp->position + 1) % std::max(20u, latency);
    }
    return (AUAudioUnitStatus)noErr;
  };
}
@end

@interface OMAutomationFactory
    : NSObject <AUAudioUnitFactory, NSExtensionRequestHandling>
@end
@implementation OMAutomationFactory
- (AUAudioUnit *)createAudioUnitWithComponentDescription:
                     (AudioComponentDescription)description
                                                   error:(NSError **)error {
  return [[OMAutomationAU alloc] initWithComponentDescription:description
                                                      options:0
                                                        error:error];
}
- (void)beginRequestWithExtensionContext:(NSExtensionContext *)context {
}
@end
