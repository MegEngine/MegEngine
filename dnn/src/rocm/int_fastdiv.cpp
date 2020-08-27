/**
 * \file dnn/src/rocm/int_fastdiv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include <cstring>
#include "src/rocm/int_fastdiv.h.hip"

namespace megdnn {
namespace rocm {

Uint32Fastdiv::Uint32Fastdiv() {
    memset(this, 0, sizeof(Uint32Fastdiv));
}

Uint32Fastdiv& Uint32Fastdiv::operator=(uint32_t d) {
    megdnn_assert(d);
    m_divisor = d;
    MEGDNN_CONSTEXPR uint32_t MAX_U32 = ~0u;
    m_inc_dividend = 0;
    m_divisor_is_not_1 = ~0u;
    if (!(d & (d - 1))) {
        // power of 2
        m_mul = 1u << 31;
        int p = 0;
        while ((1u << p) < d)
            ++p;
        megdnn_assert((1u << p) == d);
        m_shift = p ? p - 1 : 0;
        if (d == 1)
            m_divisor_is_not_1 = 0;
        return *this;
    }
    auto n_bound = uint64_t(d / 2 + 1) * MAX_U32;
    uint32_t shift = 32;
    while ((1ull << shift) < n_bound)
        ++shift;
    uint64_t mdst = 1ull << shift;
    int64_t delta = d - mdst % d;
    m_mul = mdst / d + 1;
    if ((uint64_t)delta > d / 2) {
        delta -= d;
        --m_mul;
        m_inc_dividend = 1;
    }
    megdnn_assert((uint64_t)m_mul * d == mdst + delta);
    megdnn_assert((uint64_t)std::abs(delta) * MAX_U32 < mdst);
    m_shift = shift - 32;
    return *this;
}

}  // namespace rocm
}  // namespace megdnn


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
