/* Explicit hosted audio bus requests. GPL-3.0-only. */
#pragma once

#include "JuceHeader.h"
#include <climits>
#include <cmath>
#include <map>

namespace Pedalboard {

inline int hostedBusInteger(py::handle value, const char *description,
                            int minimum = 0) {
  if (!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value))
    throw py::type_error(std::string(description) + " must be an integer");
  const auto result = py::cast<long long>(value);
  if (result < minimum || result > INT_MAX)
    throw py::value_error(std::string(description) + " is out of range");
  return (int)result;
}

struct HostedAudioRequest {
  std::map<int, py::array_t<float, py::array::c_style | py::array::forcecast>>
      inputs;
  std::map<int, int> outputs;
  int frames = -1;

  HostedAudioRequest(py::dict inputBuses, py::dict outputBuses,
                     py::object numFrames, double rate, int bufferSize) {
    if (!std::isfinite(rate) || rate <= 0 || rate > INT_MAX || bufferSize <= 0)
      throw py::value_error("Unsafe bus sample rate or buffer size");
    if (!numFrames.is_none())
      frames = hostedBusInteger(numFrames, "num_frames");
    for (auto entry : inputBuses) {
      const int index = hostedBusInteger(entry.first, "Input bus index");
      if (!py::isinstance<py::array>(entry.second))
        throw py::type_error("Each input bus must be a NumPy array");
      auto array = py::reinterpret_borrow<py::array>(entry.second);
      const auto kind = py::str(array.dtype().attr("kind")).cast<std::string>();
      if (kind != "f" || (array.itemsize() != 4 && array.itemsize() != 8))
        throw py::type_error("Bus audio must contain float32 or float64");
      if (array.ndim() != 2 || array.shape(0) <= 0 || array.shape(0) > 64 ||
          array.shape(1) > INT_MAX)
        throw py::value_error("Bus arrays must have shape (channels, frames)");
      if (frames < 0)
        frames = (int)array.shape(1);
      if (array.shape(1) != frames)
        throw py::value_error("All bus arrays must share num_frames");
      inputs.emplace(index, decltype(inputs)::mapped_type(array));
    }
    if (frames < 0)
      throw py::value_error("Input-free rendering requires num_frames");
    for (auto entry : outputBuses) {
      const int index = hostedBusInteger(entry.first, "Output bus index");
      const int channels = hostedBusInteger(entry.second, "Output channels", 1);
      if (channels > 64)
        throw py::value_error("Bus audio supports at most 64 channels per bus");
      outputs.emplace(index, channels);
    }
    if (outputs.empty())
      throw py::value_error("Request at least one output bus");
  }

  juce::AudioProcessor::BusesLayout layout(juce::AudioPluginInstance &instance,
                                           bool disableOmitted) const {
    juce::AudioProcessor::BusesLayout result;
    for (bool input : {true, false}) {
      auto &buses = input ? result.inputBuses : result.outputBuses;
      for (int i = 0; i < instance.getBusCount(input); ++i) {
        auto *bus = instance.getBus(input, i);
        buses.add(disableOmitted ? juce::AudioChannelSet::disabled()
                                 : bus->getLastEnabledLayout());
      }
      auto add = [&](int index, int channels) {
        if (index >= buses.size())
          throw py::value_error("Requested audio bus does not exist");
        auto *bus = instance.getBus(input, index);
        // Preserve the plugin's speaker arrangement where possible. Channel
        // counts alone do not identify surround/discrete speaker layouts.
        const auto current = bus->getCurrentLayout();
        const auto preferred = bus->getDefaultLayout();
        buses.set(index,
                  current.size() == channels ? current
                  : preferred.size() == channels
                      ? preferred
                      : juce::AudioChannelSet::canonicalChannelSet(channels));
      };
      if (input)
        for (const auto &[index, array] : inputs)
          add(index, (int)array.shape(0));
      else
        for (const auto &[index, channels] : outputs)
          add(index, channels);
    }
    return result;
  }
};

inline py::dict hostedAudioBuses(juce::AudioPluginInstance &instance) {
  py::dict result;
  for (bool input : {true, false}) {
    py::list buses;
    for (int index = 0; index < instance.getBusCount(input); ++index) {
      auto *bus = instance.getBus(input, index);
      py::dict item;
      item["index"] = index;
      item["name"] = bus->getName().toStdString();
      item["enabled"] = bus->isEnabled();
      item["channels"] = bus->getNumberOfChannels();
      item["default_channels"] = bus->getDefaultLayout().size();
      item["layout"] = bus->getCurrentLayout().getDescription().toStdString();
      buses.append(item);
    }
    result[input ? "inputs" : "outputs"] = buses;
  }
  return result;
}

} // namespace Pedalboard
