/**
 * \file dnn/src/fallback/matrix_mul_helper.h
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
#include <cstring>

namespace megdnn {
namespace fallback {

template <typename src_type, typename weight_type, typename dst_type>
void run_matrix_mul_tpl(const src_type * __restrict src,
        const weight_type * __restrict weight,
        dst_type * __restrict dst,
        size_t batch_size, size_t nr_inputs, size_t nr_outputs)
{
    for (size_t b = 0; b < batch_size; ++b) {
        std::memset(dst, 0, sizeof(dst_type) * nr_outputs);
        for (size_t i = 0; i < nr_inputs; ++i)
        for (size_t o = 0; o < nr_outputs; ++o)
        {
            dst[o] += weight[i*nr_outputs + o] * src[i];
        }
        src += nr_inputs;
        dst += nr_outputs;
    }
}

} // namespace fallback
} // namespace megdnn

// vim: syntax=cpp.doxygen

