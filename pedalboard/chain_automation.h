/*
 * Native effect-instance automation for Chain, Pedalboard and Mix.
 * Reuses builtin schemas and schedulers; owns graph validation and curve
 * queues. Licensed under the GNU General Public License, Version 3.0.
 */
#pragma once
#include "parameter_automation.h"
#include "plugins/Chain.h"
#include "plugins/Mix.h"
#include <unordered_set>

namespace Pedalboard {
struct AutomationTopology {
  std::weak_ptr<PluginContainer> container;
  std::vector<std::shared_ptr<Plugin>> children;
};
struct AutomationGraph {
  std::vector<std::shared_ptr<Plugin>> nodes;
  std::vector<AutomationTopology> topology;
};

inline AutomationGraph snapshotAutomationGraph(std::shared_ptr<Plugin> root) {
  AutomationGraph graph;
  std::unordered_set<const Plugin *> seen;
  std::function<void(std::shared_ptr<Plugin>)> visit = [&](auto node) {
    if (!node)
      return;
    if (!seen.insert(node.get()).second)
      throw std::invalid_argument("Automation graphs cannot contain cycles or "
                                  "duplicate plugin instances.");
    graph.nodes.push_back(node);
    if (auto container = std::dynamic_pointer_cast<PluginContainer>(node)) {
      if (!dynamic_cast<Chain *>(node.get()) &&
          !dynamic_cast<Mix *>(node.get()))
        throw std::invalid_argument(
            "Only Chain and Mix containers support graph automation.");
      std::vector<std::shared_ptr<Plugin>> children;
      {
        std::scoped_lock lock(container->mutex);
        children = container->getPlugins();
      }
      graph.topology.push_back({container, children});
      for (const auto &child : children)
        visit(child);
    }
  };
  visit(root);
  return graph;
}

inline bool sameAutomationTopology(const std::vector<AutomationTopology> &a,
                                   const std::vector<AutomationTopology> &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].container.lock() != b[i].container.lock() ||
        a[i].children != b[i].children)
      return false;
  return true;
}

using NodeAutomation =
    std::unordered_map<const Plugin *,
                       std::shared_ptr<DenseParameterProcessor>>;
struct ChainAutomationState {
  NodeAutomation adapters;
  std::vector<AutomationTopology> topology;
  double sampleRate;
  unsigned int channels;
  unsigned int maximumBlockSize;
};

// The host locks this immutable flat snapshot, including the original root.
// All state access, preflight and configuration below occur inside those locks.
class ChainAutomationRender : public PluginContainer {
public:
  ChainAutomationRender(std::shared_ptr<PluginContainer> root,
                        AutomationGraph graph, NodeAutomation incoming,
                        size_t frames, double sampleRate, unsigned int channels,
                        ChannelLayout layout, bool reset, unsigned int capacity)
      : PluginContainer({}), root(std::move(root)), graph(std::move(graph)),
        incoming(std::move(incoming)), frames(frames), sampleRate(sampleRate),
        channels(channels), capacity(capacity), resetting(reset) {
    lastChannelLayout = layout;
    lastSpec.numChannels = channels;
  }
  std::vector<std::shared_ptr<Plugin>> getAllPlugins() override {
    return graph.nodes;
  }
  void reset() override {
    start();
    ScopedProcessingRoutes scope(routes, root.get());
    root->reset();
  }
  void prepare(const juce::dsp::ProcessSpec &spec) override {
    start();
    ScopedProcessingRoutes scope(routes, root.get());
    auto preparedSpec = spec;
    preparedSpec.maximumBlockSize = capacity;
    root->prepare(preparedSpec);
    lastSpec = preparedSpec;
  }
  int getLatencyHint() override {
    ScopedProcessingRoutes scope(routes, root.get());
    return root->getLatencyHint();
  }
  int process(
      const juce::dsp::ProcessContextReplacing<float> &context) override {
    ScopedProcessingRoutes scope(routes, root.get());
    auto block = context.getOutputBlock();
    size_t valid = 0;
    for (size_t start = 0; start < block.getNumSamples(); start += capacity) {
      const size_t count =
          std::min((size_t)capacity, block.getNumSamples() - start);
      auto span = block.getSubBlock(start, count);
      juce::dsp::ProcessContextReplacing<float> part(span);
      part.isBypassed = context.isBypassed;
      const int returned = root->process(part);
      if (returned < 0 || returned > (int)count)
        throw std::runtime_error("Invalid graph output frame count.");
      for (size_t c = 0; c < block.getNumChannels(); ++c)
        std::memmove(block.getChannelPointer(c) + valid,
                     span.getChannelPointer(c) + count - returned,
                     sizeof(float) * returned);
      valid += returned;
    }
    for (size_t c = 0; c < block.getNumChannels(); ++c)
      std::memmove(block.getChannelPointer(c) + block.getNumSamples() - valid,
                   block.getChannelPointer(c), sizeof(float) * valid);
    return (int)valid;
  }

private:
  void start() {
    if (started)
      return;
    for (const auto &entry : graph.topology) {
      auto container = entry.container.lock();
      if (!container || container->getPlugins() != entry.children)
        throw std::invalid_argument("The graph changed while starting this "
                                    "render; retry with reset=True.");
    }
    auto previous = root->automationState;
    if (!resetting) {
      if (!previous && !root->automationResetReady)
        throw std::invalid_argument("Start chain automation with reset=True or "
                                    "explicitly reset the root first.");
      if (previous &&
          (sampleRate != previous->sampleRate ||
           channels != previous->channels ||
           !sameAutomationTopology(graph.topology, previous->topology)))
        throw std::invalid_argument(
            "Changing graph topology or audio format requires reset=True.");
      if (previous) {
        capacity = previous->maximumBlockSize;
        for (const auto &[node, adapter] : incoming)
          if (!previous->adapters.count(node))
            throw std::invalid_argument(
                "Adding automation lanes requires reset=True.");
        for (const auto &[node, old] : previous->adapters) {
          if (!incoming.count(node))
            incoming[node] = old->heldContinuation(frames);
          incoming.at(node)->prependPending(*old);
        }
      }
    }
    // Pure validation must complete for the entire graph before any latch,
    // setter, reset, pending queue or original DSP memory is changed.
    for (const auto &[node, adapter] : incoming)
      adapter->validateOnly();
    auto next = std::make_shared<ChainAutomationState>();
    next->adapters = std::move(incoming);
    next->topology = graph.topology;
    next->sampleRate = sampleRate;
    next->channels = channels;
    next->maximumBlockSize = capacity;
    for (const auto &[node, adapter] : next->adapters) {
      adapter->enableTailHold();
      adapter->commitValidation();
      routes[node] = adapter.get();
    }
    root->automationState = next->adapters.empty() ? nullptr : std::move(next);
    started = true;
  }
  std::shared_ptr<PluginContainer> root;
  AutomationGraph graph;
  NodeAutomation incoming;
  NativeProcessingRoutes routes;
  size_t frames;
  double sampleRate;
  unsigned int channels;
  unsigned int capacity;
  bool resetting, started = false;
};

inline py::array_t<float>
processChainAutomation(std::shared_ptr<PluginContainer> root, py::array input,
                       double sampleRate, unsigned int bufferSize, bool reset,
                       py::object automation) {
  if (!automation.is_none() && !py::isinstance<py::dict>(automation))
    throw py::type_error(
        "automation must map effect instances to parameter-curve dicts.");
  auto mapping =
      automation.is_none() ? py::dict() : automation.cast<py::dict>();
  bool active;
  {
    std::scoped_lock lock(root->mutex);
    active = (bool)root->automationState;
  }
  if (mapping.empty() && !active)
    return process(input, sampleRate, {root}, bufferSize, reset);
  if (!std::isfinite(sampleRate) || sampleRate <= 0 || bufferSize == 0 ||
      bufferSize > (unsigned int)std::numeric_limits<int>::max())
    throw py::value_error("sample_rate and buffer_size must be positive and "
                          "finite; buffer_size must fit a native int.");
  py::array_t<float, py::array::c_style> audio;
  if (input.dtype().char_() == 'f')
    audio = input;
  else if (input.dtype().char_() == 'd')
    audio = input.attr("astype")("float32");
  else
    throw py::type_error("Audio must use float32 or float64.");
  ChannelLayout layout;
  {
    std::scoped_lock lock(root->mutex);
    layout = root->parseAndCacheChannelLayout(audio);
  }
  const size_t frames =
      audio.ndim() == 1
          ? audio.shape(0)
          : audio.shape(layout == ChannelLayout::Interleaved ? 0 : 1);
  const size_t channels =
      audio.ndim() == 1
          ? 1
          : audio.shape(layout == ChannelLayout::Interleaved ? 1 : 0);
  if (frames > (size_t)INT_MAX || channels > (size_t)INT_MAX)
    throw py::value_error("Audio dimensions exceed native buffer capacity.");
  auto graph = snapshotAutomationGraph(root);
  NodeAutomation incoming;
  for (auto entry : mapping) {
    if (!py::isinstance<Plugin>(entry.first))
      throw py::type_error("Automation keys must be effect instances, not "
                           "indices or parameter names.");
    auto node = py::cast<std::shared_ptr<Plugin>>(entry.first);
    if (std::find(graph.nodes.begin(), graph.nodes.end(), node) ==
        graph.nodes.end())
      throw py::value_error("Automation target is not in this effect graph.");
    if (!py::isinstance<py::dict>(entry.second))
      throw py::type_error("Each effect must map to a parameter-curve dict.");
    auto factory =
        denseParameterFactories().find(std::type_index(typeid(*node)));
    if (factory == denseParameterFactories().end())
      throw py::value_error(
          "This effect does not support native parameter automation.");
    auto parameters = py::reinterpret_borrow<py::dict>(entry.second);
    if (!parameters.empty())
      incoming[node.get()] = factory->second(
          node, parameters, frames, sampleRate, channels, layout, reset);
  }
  if (!frames || !channels)
    return process(audio, sampleRate, {}, bufferSize, false);
  if (incoming.empty() && !active)
    return process(audio, sampleRate, {root}, bufferSize, reset);
  auto render = std::make_shared<ChainAutomationRender>(
      root, std::move(graph), std::move(incoming), frames, sampleRate, channels,
      layout, reset, std::min(bufferSize, (unsigned int)DEFAULT_BUFFER_SIZE));
  return process(audio, sampleRate, {render}, bufferSize, reset);
}

// Reset and render use the same lock order; release locks before taking the
// GIL again. In particular, never destroy an active scheduling session while
// another thread is executing its borrowed native routes.
inline void resetChainAutomation(std::shared_ptr<PluginContainer> root) {
  auto graph = snapshotAutomationGraph(root);
  py::gil_scoped_release release;
  std::sort(graph.nodes.begin(), graph.nodes.end(),
            [](const auto &a, const auto &b) { return a.get() < b.get(); });
  std::vector<std::unique_lock<std::mutex>> locks;
  for (const auto &node : graph.nodes)
    locks.emplace_back(node->mutex);
  for (const auto &entry : graph.topology) {
    auto container = entry.container.lock();
    if (!container || container->getPlugins() != entry.children)
      throw std::invalid_argument(
          "The graph changed while resetting; retry reset.");
  }
  for (const auto &entry : graph.topology)
    entry.container.lock()->automationState.reset();
  root->reset();
  root->resetLastChannelLayout();
}

inline void init_chain_automation(py::module &utils) {
  auto bind = [&](auto &binding) {
    binding.def(
        "reset", &resetChainAutomation,
        "Clear DSP and graph scheduling memory, preserving scalar parameters.");
    for (const char *method : {"process", "__call__"})
      binding.def(
          method, &processChainAutomation, py::arg("input_array"),
          py::arg("sample_rate"), py::arg("buffer_size") = DEFAULT_BUFFER_SIZE,
          py::arg("reset") = true, py::kw_only(),
          py::arg("automation") = py::none(),
          R"(Process an effect graph with instance-keyed NumPy parameter curves.

Use automation={effect: {parameter: curve}}. Curves use root input frames and
follow valid audio through serial and parallel latency. Container contents are
unchanged. For offline rendering use reset=True. For streaming explicitly reset
the root first, then slice audio/curves together with
reset=False. Omitted lanes hold their last scheduled target. Changing lanes,
topology, sample rate or channel count requires reset=True. Automatic latency
flush holds the final target. Directly rendering or resetting a participating
child during a session requires restarting the root with reset=True.)");
  };
  auto chain = py::reinterpret_borrow<
      py::class_<Chain, PluginContainer, std::shared_ptr<Chain>>>(
      utils.attr("Chain"));
  auto mix = py::reinterpret_borrow<
      py::class_<Mix, PluginContainer, std::shared_ptr<Mix>>>(
      utils.attr("Mix"));
  bind(chain);
  bind(mix);
}
} // namespace Pedalboard
