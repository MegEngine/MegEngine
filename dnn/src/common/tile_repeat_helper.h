#pragma once
#include <cstddef>

namespace megdnn {

// Tile (m, n) to (m, n*times) or Repeat (m, n) to (m*times, n)
template <typename T>
void tile_or_repeat_single_axis(
        const T* __restrict src, T* __restrict dst, const size_t m, const size_t n,
        const size_t times);
// forward and backward can share the same init/update functions.
template <typename T>
void init_tile_repeat_state(
        const T* src, T* dst, T* workspace0, T* workspace1, T*& current, T*& next,
        size_t& state, size_t nr_reduces);
template <typename T>
void update_tile_repeat_state(
        const T* src, T* dst, T* workspace0, T* workspace1, T*& current, T*& next,
        size_t& state, size_t nr_reduces);

}  // namespace megdnn

// vim: syntax=cpp.doxygen
