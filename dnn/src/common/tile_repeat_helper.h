/**
 * \file dnn/src/common/tile_repeat_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cstddef>

namespace megdnn {

// Tile (m, n) to (m, n*times) or Repeat (m, n) to (m*times, n)
template <typename T>
void tile_or_repeat_single_axis(const T * __restrict src,
        T * __restrict dst,
        const size_t m, const size_t n, const size_t times);
// forward and backward can share the same init/update functions.
template <typename T>
void init_tile_repeat_state(const T *src, T *dst,
        T *workspace0, T *workspace1,
        T *&current, T *&next, size_t &state,
        size_t nr_reduces);
template <typename T>
void update_tile_repeat_state(const T *src, T *dst,
        T *workspace0, T *workspace1,
        T *&current, T *&next, size_t &state,
        size_t nr_reduces);

} // namespace megdnn

// vim: syntax=cpp.doxygen

