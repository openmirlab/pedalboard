/* Independent VST3 SDK routing oracle. Loads the same fixture binary through
 * its exported factory, without Pedalboard/JUCE processing. GPL-3.0-only. */
#include "host_events.h"
#include "pluginterfaces/base/ipluginbase.h"
#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/ivstprocesscontext.h"
#include "pluginterfaces/vst/vstspeaker.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace Steinberg;
using namespace Steinberg::Vst;

static void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

int main(int argc, char **argv) {
  try {
    require((argc == 9 || argc == 10 || argc == 20),
            "binary output rate block main_channels key_channels "
            "aux_channels frames");
#ifdef _WIN32
    auto module = LoadLibraryA(argv[1]);
    auto getFactory = reinterpret_cast<IPluginFactory *(*)()>(
        GetProcAddress(module, "GetPluginFactory"));
#else
    auto module = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    require(module, "Cannot load native fixture");
    auto getFactory = reinterpret_cast<IPluginFactory *(*)()>(
        dlsym(module, "GetPluginFactory"));
#endif
    require(getFactory, "Missing plugin factory");
    auto *factory = getFactory();
    PClassInfo info{};
    const bool clockMode = argc == 20;
    const std::string clockName =
        clockMode ? "Automation Probe " + std::string(argv[9]) : "";
    const bool midiMode = argc == 10;
    const char *name = clockMode   ? clockName.c_str()
                       : !midiMode ? "Automation Probe Buses"
                       : std::strcmp(argv[9], "delayed") == 0
                           ? "Automation Probe Delayed Audio MIDI"
                           : "Automation Probe Audio MIDI";
    bool found = false;
    for (int i = 0; i < factory->countClasses(); ++i) {
      require(factory->getClassInfo(i, &info) == kResultOk,
              "Class query failed");
      if (std::strcmp(info.name, name) == 0) {
        found = true;
        break;
      }
    }
    require(found, "Missing bus diagnostic class");
    IComponent *component = nullptr;
    require(factory->createInstance(info.cid, IComponent::iid,
                                    (void **)&component) == kResultOk,
            "Cannot create component");
    require(component->initialize(nullptr) == kResultOk, "Cannot initialize");
    IAudioProcessor *processor = nullptr;
    require(component->queryInterface(IAudioProcessor::iid,
                                      (void **)&processor) == kResultOk,
            "Missing audio processor");
    const double rate = std::stod(argv[3]);
    const int block = std::stoi(argv[4]), mainChannels = std::stoi(argv[5]),
              keyChannels = std::stoi(argv[6]),
              auxChannels = std::stoi(argv[7]), frames = std::stoi(argv[8]);
    require(block > 0 && frames > 0 && mainChannels >= 1 && mainChannels <= 2 &&
                keyChannels >= 0 && keyChannels <= 2 && auxChannels >= 0 &&
                auxChannels <= 2,
            "Invalid host arguments");
    SpeakerArrangement ins[] = {
        mainChannels == 1 ? SpeakerArr::kMono : SpeakerArr::kStereo,
        keyChannels == 1 ? SpeakerArr::kMono : SpeakerArr::kStereo};
    SpeakerArrangement outs[] = {
        ins[0], auxChannels == 1 ? SpeakerArr::kMono : SpeakerArr::kStereo};
    require(component->getBusCount(kAudio, kInput) == 2 &&
                component->getBusCount(kAudio, kOutput) == 2,
            "Unexpected native bus count");
    require(processor->setBusArrangements(ins, 2, outs, 2) == kResultOk,
            "Unsupported native layout");
    for (int index = 0; index < 2; ++index) {
      require(component->activateBus(kAudio, kInput, index,
                                     !index || keyChannels > 0) == kResultOk,
              "Input activation failed");
      require(component->activateBus(kAudio, kOutput, index,
                                     !index || auxChannels > 0) == kResultOk,
              "Output activation failed");
    }
    ProcessSetup setup{kOffline, kSample32, block, rate};
    require(processor->setupProcessing(setup) == kResultOk, "Setup failed");
    require(component->setActive(true) == kResultOk &&
                processor->setProcessing(true) == kResultOk,
            "Activation failed");
    std::vector<float> output((mainChannels + auxChannels) * frames);
    std::array<std::vector<float>, 8> storage;
    std::array<float *, 8> pointers;
    for (int c = 0; c < 8; ++c) {
      storage[c].resize(block);
      pointers[c] = storage[c].data();
    }
    std::ofstream midiFile(std::string(argv[2]) + ".midi");
    for (int start = 0; start < frames;) {
      int count = std::min(block, frames - start);
      ProcessContext timing{};
      if (clockMode) {
        const double bpm = std::stod(argv[10]),
                     initialBeat = std::stod(argv[11]);
        const int64 initialSample = std::stoll(argv[12]);
        const bool playing = std::stoi(argv[13]),
                   recording = std::stoi(argv[14]),
                   looping = std::stoi(argv[15]);
        const double firstBeat = std::stod(argv[16]),
                     lastBeat = std::stod(argv[17]);
        const int numerator = std::stoi(argv[18]),
                  denominator = std::stoi(argv[19]);
        const double beatStep = bpm / (60 * rate);
        double beat = initialBeat + (playing ? start * beatStep : 0);
        double exactSample = initialSample + (playing ? start : 0);
        if (playing && looping) {
          if (beat >= lastBeat - 1e-12) {
            double phase = std::fmod(beat - firstBeat, lastBeat - firstBeat);
            if (lastBeat - firstBeat - phase < 1e-12)
              phase = 0;
            beat = firstBeat + phase;
            exactSample = initialSample + (beat - initialBeat) / beatStep;
          }
          // Independent host clips the native block at the next loop boundary.
          const double remaining = (lastBeat - beat) / beatStep;
          if (remaining < count)
            count = std::max(1, (int)std::ceil(remaining - 1e-8));
        }
        timing.sampleRate = rate;
        timing.tempo = bpm;
        timing.projectTimeMusic = beat;
        const double lower = std::floor(exactSample);
        timing.projectTimeSamples =
            (int64)(lower + (exactSample - lower >= .5 - 1e-8 ? 1 : 0));
        timing.timeSigNumerator = numerator;
        timing.timeSigDenominator = denominator;
        timing.barPositionMusic =
            std::floor(beat / (4.0 * numerator / denominator)) *
            (4.0 * numerator / denominator);
        timing.cycleStartMusic = firstBeat;
        timing.cycleEndMusic = lastBeat;
        timing.state = ProcessContext::kTempoValid |
                       ProcessContext::kProjectTimeMusicValid |
                       ProcessContext::kBarPositionValid |
                       ProcessContext::kTimeSigValid;
        if (playing)
          timing.state |= ProcessContext::kPlaying;
        if (recording)
          timing.state |= ProcessContext::kRecording;
        if (looping)
          timing.state |= ProcessContext::kCycleActive;
        if (lastBeat > firstBeat)
          timing.state |= ProcessContext::kCycleValid;
      }
      for (int c = 0; c < mainChannels; ++c)
        for (int i = 0; i < count; ++i)
          storage[c][i] = (c ? -.1f : .2f) + (start + i) * (1.f / 65536.f);
      for (int c = 0; c < keyChannels; ++c)
        for (int i = 0; i < count; ++i)
          storage[2 + c][i] = start + i >= 31 && start + i < 129 ? 1.f : 0.f;
      AudioBusBuffers inputs[2]{}, outputs[2]{};
      inputs[0].numChannels = mainChannels;
      inputs[0].channelBuffers32 = pointers.data();
      inputs[1].numChannels = keyChannels;
      inputs[1].channelBuffers32 = pointers.data() + 2;
      outputs[0].numChannels = mainChannels;
      outputs[0].channelBuffers32 = pointers.data() + 4;
      outputs[1].numChannels = auxChannels;
      outputs[1].channelBuffers32 = pointers.data() + 6;
      ProcessData data{};
      data.processMode = kOffline;
      data.symbolicSampleSize = kSample32;
      data.numSamples = count;
      data.numInputs = 2;
      data.numOutputs = 2;
      data.inputs = inputs;
      data.outputs = outputs;
      if (clockMode)
        data.processContext = &timing;
      HostEvents inputEvents, outputEvents;
      HostChanges changes;
      if (midiMode) {
        const int positions[] = {0, 7, 31, 127, 128, 258};
        for (int index = 0; index < 6; ++index) {
          if (positions[index] < start || positions[index] >= start + count)
            continue;
          Event event{};
          event.sampleOffset = positions[index] - start;
          if (index % 2 == 0) {
            event.type = Event::kNoteOnEvent;
            event.noteOn.pitch = 60;
            event.noteOn.velocity = 100.f / 127.f;
            event.noteOn.noteId = -1;
          } else {
            event.type = Event::kNoteOffEvent;
            event.noteOff.pitch = 60;
            event.noteOff.noteId = -1;
          }
          inputEvents.addEvent(event);
        }
        changes.gain.points = {{0, (float)(60.0 / 72.0)}};
        changes.frequency.points = {{0, 1}};
        data.inputParameterChanges = &changes;
        data.inputEvents = &inputEvents;
        data.outputEvents = &outputEvents;
      }
      require(processor->process(data) == kResultOk, "Native process failed");
      if (midiMode)
        for (const auto &event : outputEvents.events)
          midiFile << start + event.sampleOffset << ' ' << event.type << ' '
                   << (event.type == Event::kNoteOnEvent ? event.noteOn.pitch
                                                         : event.noteOff.pitch)
                   << '\n';
      for (int c = 0; c < mainChannels; ++c)
        std::copy_n(storage[4 + c].data(), count,
                    output.data() + c * frames + start);
      for (int c = 0; c < auxChannels; ++c)
        std::copy_n(storage[6 + c].data(), count,
                    output.data() + (mainChannels + c) * frames + start);
      start += count;
    }
    std::ofstream file(argv[2], std::ios::binary);
    file.write(reinterpret_cast<const char *>(output.data()),
               output.size() * sizeof(float));
    require(file.good(), "Cannot write native output");
    processor->setProcessing(false);
    component->setActive(false);
    processor->release();
    component->terminate();
    component->release();
    factory->release();
    std::cout << "Native VST3 bus processing accepted\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
