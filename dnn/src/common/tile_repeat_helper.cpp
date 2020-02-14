/**
 * \file dnn/src/common/tile_repeat_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/common/tile_repeat_helper.h"

#include "src/common/utils.h"
#include <cstring>

namespace megdnn {

// Tile (m, n) to (m, n*times) or Repeat (m, n) to (m*times, n)
template <typename T>
void tile_or_repeat_single_axis(const T * __restrict src,
        T * __restrict dst,
        const size_t m, const size_t n, const size_t times)
{
    rep(i, m) {
        // copy Ts of length n to dst
        std::memcpy(dst, src, sizeof(T) * n);
        size_t k = 1u;
        while (k*2 <= times) {
            std::memcpy(dst + k*n, dst, sizeof(T) * (k*n));
            k *= 2;
        }
        if (k < times) {
            std::memcpy(dst + k*n, dst, sizeof(T) * (times-k) * n);
        }
        src += n;
        dst += n*times;
    }
}

template <typename T>
void init_tile_repeat_state(const T *src, T *dst,
        T *workspace0, T * /* workspace1 */,
        T *&current, T *&next, size_t &state,
        size_t nr_reduces)
{
    current = const_cast<T *>(src);
    if (nr_reduces == 1) {
        next = dst;
    } else {
        next = workspace0;
    }
    state = 0;
}

template <typename T>
void update_tile_repeat_state(const T * /* src */, T *dst,
        T *workspace0, T *workspace1,
        T *&current, T *&next, size_t &state,
        size_t nr_reduces)
{
    current = next;
    if (nr_reduces == 1) {
        next = nullptr;
    } else if (nr_reduces == 2) {
        if (state == 0) {
            next = dst;
        } else {
            next = nullptr;
        }
    } else {
        if (state == 0) {
            next = workspace1;
        } else if (state + 1 == nr_reduces) {
            next = nullptr;
        } else if (state + 2 == nr_reduces) {
            next = dst;
        } else {
            megdnn_assert(current == workspace0 || current == workspace1,
                    "Impossible happened; internal bug.");
            next = (current == workspace0 ? workspace1 : workspace0);
        }
    }
    ++state;
}

#define INST(T) \
template void tile_or_repeat_single_axis<T>(const T *, T *, \
        const size_t, const size_t, const size_t); \
template void init_tile_repeat_state<T>(const T *, T *, T *, T *, T *&, T *&, \
        size_t &, size_t); \
template void update_tile_repeat_state<T>(const T *, T *, T *, T *, T *&, T *&, \
        size_t &, size_t);

#define INST_DT(d) INST(DTypeTrait<d>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE(INST_DT)

} // namespace megdnn

// vim: syntax=cpp.doxygen

