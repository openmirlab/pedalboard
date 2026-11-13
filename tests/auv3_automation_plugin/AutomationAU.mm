/* Original AUv3 diagnostic extension, GPL-3.0-only. */
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
std::atomic<unsigned> nextInstance{1};
struct DSP {
  std::atomic<float> gain{0}, frequency{20}, pass{0}, identity{0}, decoy{0};
  std::atomic<unsigned> renderFrames{0}, renderCapacity{0};
  bool note = false;
  unsigned position = 0;
  std::array<std::array<float, 16385>, 2> delay{};
  void clear() {
    note = false;
    position = 0;
    for (auto &channel : delay)
      channel.fill(0);
  }
};
} // namespace

@interface OMAutomationAU : AUAudioUnit
@end
@implementation OMAutomationAU {
  AUAudioUnitBusArray *_inputs, *_outputs;
  AUParameterTree *_parameters;
  DSP _dsp;
  unsigned _instance;
  BOOL _instrument;
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
  _latency = variant == '1' || variant == '3' ? 20
             : variant == '4'                 ? 4097
             : variant == '5'                 ? 16385
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
  _inputs = [[AUAudioUnitBusArray alloc]
      initWithAudioUnit:self
                busType:AUAudioUnitBusTypeInput
                 busses:_instrument ? @[] : @[ input ]];
  _outputs =
      [[AUAudioUnitBusArray alloc] initWithAudioUnit:self
                                             busType:AUAudioUnitBusTypeOutput
                                              busses:@[ output ]];
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
  identity.value = decoy.value = 0;
  gain.value = 0;
  frequency.value = 20;
  pass.value = 0;
  pid.value = getpid();
  instance.value = _instance;
  _parameters = [AUParameterTree createTreeWithChildren:@[
    gain, frequency, pass, pid, instance, identity, decoy, frames, capacity
  ]];
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
  return _instrument ? @[ @0, @1, @0, @2 ] : @[ @1, @1, @2, @2 ];
}
- (BOOL)canProcessInPlace {
  return YES;
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
  const unsigned latency = _latency;
  const AUAudioFrameCount capacity = self.maximumFramesToRender;
  return ^AUAudioUnitStatus(
      AudioUnitRenderActionFlags *flags, const AudioTimeStamp *timestamp,
      AUAudioFrameCount frames, NSInteger bus, AudioBufferList *output,
      const AURenderEvent *events, AURenderPullInputBlock pull) {
    dsp->renderFrames.store(frames);
    dsp->renderCapacity.store(capacity);
    if (frames > capacity)
      return (AUAudioUnitStatus)kAudioUnitErr_TooManyFramesToProcess;
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
        }
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
