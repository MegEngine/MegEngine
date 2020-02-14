/**
 * \file dnn/src/x86/elemwise_helper/kimpl/tanh.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/x86/elemwise_helper/kimpl/op_unary_base.h"
#include "src/x86/utils.h"

namespace megdnn {
namespace x86 {

template <SIMDType simd_type, typename src_ctype,
          typename dst_ctype = src_ctype>
struct TanhOp : UnaryOpBase<simd_type, src_ctype, dst_ctype> {
    using UnaryOpBase<simd_type, src_ctype, dst_ctype>::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 1;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const {
        float tmp = src;
        return tanh(tmp);
    }
};

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
