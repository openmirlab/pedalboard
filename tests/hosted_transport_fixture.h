/* Audio encoding of actual plugin-received transport. No host clock math.
 * GPL-3.0-only. */
#pragma once
#include <array>
#include <cmath>
#include <cstdint>

struct TransportObservation {
  double tempo = -1, ppq = -1, bar = -1, start = 0, end = 0, rate = 44100;
  int64_t sample = 0;
  int numerator = 0, denominator = 0, flags = 0;
  uint32_t nextBeat = 0;
  bool playing = false, changed = false;

  std::array<float, 4> audio(int mode, int offset) const {
    const double beat = ppq + (playing ? offset * tempo / (60 * rate) : 0);
    if (mode == 0)
      return {(float)(tempo / 1000), (float)(beat / 1000), (float)flags / 256,
              (float)std::sin(beat * 6.283185307179586)};
    if (mode == 1)
      return {(float)numerator / 32, (float)denominator / 32,
              (float)(bar / 1000), (float)nextBeat / 65536};
    if (mode == 2)
      return {(float)(start / 1000), (float)(end / 1000), changed ? 1.f : 0.f,
              0};
    const uint64_t frame = (uint64_t)(sample + (playing ? offset : 0));
    std::array<float, 4> result{};
    for (int word = 0; word < 4; ++word)
      result[word] = (float)((frame >> (16 * word)) & 65535) / 65536;
    return result;
  }
};
