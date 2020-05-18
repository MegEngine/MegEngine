/**
 * \file dnn/src/x86/add_update/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "src/fallback/add_update/opr_impl.h"

#include "src/common/utils.h"
#include "src/x86/handle.h"
#include "src/x86/utils.h"

#include <immintrin.h>
#ifdef WIN32
#include <avxintrin.h>
#include <fmaintrin.h>
#endif

namespace {

using namespace megdnn;
using namespace x86;

MEGDNN_ATTRIBUTE_TARGET("fma")
void add_update_fp32_fma(_megdnn_tensor_inout dest, _megdnn_tensor_in delta,
                         const AddUpdate::Param& param) {
    dt_float32 alpha(param.alpha), beta(param.beta), bias(param.bias);
    __m256 packed_alpha(_mm256_set1_ps(alpha));
    __m256 packed_beta(_mm256_set1_ps(beta));
    __m256 packed_bias(_mm256_set1_ps(bias));

    dt_float32* dest_ptr = dest.ptr<dt_float32>();
    dt_float32* delta_ptr = delta.ptr<dt_float32>();
    size_t i = 0;
    size_t total_nr_elems = dest.layout.total_nr_elems();
    __m256 x0, b0;
    for (i = 0; i + 7 < total_nr_elems; i += 8) {
        b0 = _mm256_loadu_ps(delta_ptr + i);
        x0 = _mm256_loadu_ps(dest_ptr + i);
        b0 = _mm256_fmadd_ps(packed_beta, b0, packed_bias);
        x0 = _mm256_fmadd_ps(packed_alpha, x0, b0);
        _mm256_storeu_ps(dest_ptr + i, x0);
    }
    for (; i < total_nr_elems; i++) {
        dest_ptr[i] = alpha * dest_ptr[i] + beta * delta_ptr[i] + bias;
    }
}

}  // anonymous namespace

namespace megdnn {
namespace x86 {

void AddUpdateImpl::exec(_megdnn_tensor_inout dest, _megdnn_tensor_in delta) {
    check_exec(dest.layout, delta.layout);
    // eq_shape is the same as eq_layout when both input tensors are contiguous.
    if (is_supported(SIMDType::FMA) && delta.layout.is_contiguous() &&
        dest.layout.is_contiguous() && delta.layout.eq_shape(dest.layout) &&
        dest.layout.dtype == delta.layout.dtype) {
        if (dest.layout.dtype == ::megdnn::dtype::Float32()) {
            MEGDNN_DISPATCH_CPU_KERN_OPR(
                    add_update_fp32_fma(dest, delta, m_param));
            return;
        }
    }
    return megdnn::fallback::AddUpdateImpl::exec(dest, delta);
}

}  // namespace x86
}  // namespace megdnn
// vim: syntax=cpp.doxygen
