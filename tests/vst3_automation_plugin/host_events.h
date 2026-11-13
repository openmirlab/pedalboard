/* Small independent SDK-host event/parameter containers. GPL-3.0-only. */
#pragma once
#include "pluginterfaces/vst/ivstevents.h"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include <vector>

template <typename Interface> struct HostInterface : Interface {
  Steinberg::tresult PLUGIN_API queryInterface(const Steinberg::TUID id,
                                               void **out) override {
    *out = nullptr;
    if (Steinberg::FUnknownPrivate::iidEqual(id, Interface::iid) ||
        Steinberg::FUnknownPrivate::iidEqual(id, Steinberg::FUnknown::iid)) {
      *out = static_cast<Interface *>(this);
      addRef();
      return Steinberg::kResultOk;
    }
    return Steinberg::kNoInterface;
  }
  Steinberg::uint32 PLUGIN_API addRef() override { return ++references; }
  Steinberg::uint32 PLUGIN_API release() override { return --references; }
  Steinberg::uint32 references = 1;
};

struct HostEvents : HostInterface<Steinberg::Vst::IEventList> {
  std::vector<Steinberg::Vst::Event> events;
  Steinberg::int32 PLUGIN_API getEventCount() override {
    return (Steinberg::int32)events.size();
  }
  Steinberg::tresult PLUGIN_API
  getEvent(Steinberg::int32 index, Steinberg::Vst::Event &event) override {
    if (index < 0 || (size_t)index >= events.size())
      return Steinberg::kInvalidArgument;
    event = events[index];
    return Steinberg::kResultOk;
  }
  Steinberg::tresult PLUGIN_API
  addEvent(Steinberg::Vst::Event &event) override {
    events.push_back(event);
    return Steinberg::kResultOk;
  }
};

struct HostQueue : HostInterface<Steinberg::Vst::IParamValueQueue> {
  Steinberg::Vst::ParamID id = 0;
  std::vector<std::pair<Steinberg::int32, Steinberg::Vst::ParamValue>> points;
  Steinberg::Vst::ParamID PLUGIN_API getParameterId() override { return id; }
  Steinberg::int32 PLUGIN_API getPointCount() override {
    return (Steinberg::int32)points.size();
  }
  Steinberg::tresult PLUGIN_API
  getPoint(Steinberg::int32 index, Steinberg::int32 &offset,
           Steinberg::Vst::ParamValue &value) override {
    if (index < 0 || (size_t)index >= points.size())
      return Steinberg::kInvalidArgument;
    offset = points[index].first;
    value = points[index].second;
    return Steinberg::kResultOk;
  }
  Steinberg::tresult PLUGIN_API addPoint(Steinberg::int32 offset,
                                         Steinberg::Vst::ParamValue value,
                                         Steinberg::int32 &index) override {
    index = (Steinberg::int32)points.size();
    points.emplace_back(offset, value);
    return Steinberg::kResultOk;
  }
};

struct HostChanges : HostInterface<Steinberg::Vst::IParameterChanges> {
  HostQueue gain, frequency;
  HostChanges() {
    gain.id = 0x1001;
    frequency.id = 0x314159;
  }
  Steinberg::int32 PLUGIN_API getParameterCount() override { return 2; }
  Steinberg::Vst::IParamValueQueue *PLUGIN_API
  getParameterData(Steinberg::int32 index) override {
    return index == 0 ? &gain : index == 1 ? &frequency : nullptr;
  }
  Steinberg::Vst::IParamValueQueue *PLUGIN_API addParameterData(
      const Steinberg::Vst::ParamID &id, Steinberg::int32 &index) override {
    index = id == gain.id ? 0 : id == frequency.id ? 1 : -1;
    return getParameterData(index);
  }
};
