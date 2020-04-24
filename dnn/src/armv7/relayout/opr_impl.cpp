/**
 * \file dnn/src/armv7/relayout/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.h"
#include "src/common/relayout_helper.h"

#include "src/armv7/handle.h"
#include "src/armv7/relayout/opr_impl.h"

using namespace megdnn;
using namespace relayout;

namespace {
struct TransposeByte {
    uint8_t v;
};

void trans_16x16_u8(const void* src, void* dst, const size_t src_step,
                    const size_t dst_step) {
    // 16x16
    asm volatile(
        "\n"
        "vld1.8 {d0, d1}, [%[src]], %[src_step] \n"
        "vld1.8 {d2, d3}, [%[src]], %[src_step] \n"
        "vld1.8 {d4, d5}, [%[src]], %[src_step] \n"
        "vld1.8 {d6, d7}, [%[src]], %[src_step] \n"
        "vld1.8 {d8, d9}, [%[src]], %[src_step] \n"
        "vld1.8 {d10, d11}, [%[src]], %[src_step] \n"
        "vld1.8 {d12, d13}, [%[src]], %[src_step] \n"
        "vld1.8 {d14, d15}, [%[src]], %[src_step] \n"
        "vld1.8 {d16, d17}, [%[src]], %[src_step] \n"
        "vld1.8 {d18, d19}, [%[src]], %[src_step] \n"
        "vld1.8 {d20, d21}, [%[src]], %[src_step] \n"
        "vld1.8 {d22, d23}, [%[src]], %[src_step] \n"
        "vld1.8 {d24, d25}, [%[src]], %[src_step] \n"
        "vld1.8 {d26, d27}, [%[src]], %[src_step] \n"
        "vld1.8 {d28, d29}, [%[src]], %[src_step] \n"
        "vld1.8 {d30, d31}, [%[src]], %[src_step] \n"
        "vtrn.8 q0, q1 \n"
        "vtrn.8 q2, q3 \n"
        "vtrn.8 q4, q5 \n"
        "vtrn.8 q6, q7 \n"
        "vtrn.8 q8, q9 \n"
        "vtrn.8 q10, q11 \n"
        "vtrn.8 q12, q13 \n"
        "vtrn.8 q14, q15 \n"
        "vtrn.16 q0, q2 \n"
        "vtrn.16 q1, q3 \n"
        "vtrn.16 q4, q6 \n"
        "vtrn.16 q5, q7 \n"
        "vtrn.16 q8, q10 \n"
        "vtrn.16 q9, q11 \n"
        "vtrn.16 q12, q14 \n"
        "vtrn.16 q13, q15 \n"
        "vtrn.32 q0, q4 \n"
        "vtrn.32 q1, q5 \n"
        "vtrn.32 q2, q6 \n"
        "vtrn.32 q3, q7 \n"
        "vtrn.32 q8, q12 \n"
        "vtrn.32 q9, q13 \n"
        "vtrn.32 q10, q14 \n"
        "vtrn.32 q11, q15 \n"
        "vswp d1, d16 \n"
        "vswp d3, d18 \n"
        "vswp d5, d20 \n"
        "vswp d7, d22 \n"
        "vswp d9, d24 \n"
        "vswp d11, d26 \n"
        "vswp d13, d28 \n"
        "vswp d15, d30 \n"
        "vst1.8 {d0, d1}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d2, d3}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d4, d5}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d6, d7}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d8, d9}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d10, d11}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d12, d13}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d14, d15}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d16, d17}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d18, d19}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d20, d21}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d22, d23}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d24, d25}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d26, d27}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d28, d29}, [%[dst]], %[dst_step] \n"
        "vst1.8 {d30, d31}, [%[dst]], %[dst_step] \n"
        :
        [src] "+r" (src),
        [dst] "+r" (dst)
        :
        [src_step] "r" (src_step),
        [dst_step] "r" (dst_step)
        :
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31");
}

} // anonymous namespace

namespace megdnn {
namespace relayout {
namespace transpose_fallback {
template <>
struct transpose_traits<TransposeByte> {
    static constexpr size_t block_size = 16;
};

template <>
void transpose_block<TransposeByte>(const TransposeByte* src,
                                    TransposeByte* dst, const size_t src_stride,
                                    const size_t dst_stride) {
    trans_16x16_u8(src, dst, src_stride, dst_stride);
}

}  // namespace transpose_fallback
}  // namespace relayout
}  // namespace megdnn


void armv7::RelayoutForwardImpl::exec(_megdnn_tensor_in src0,
                                      _megdnn_tensor_out dst0,
                                      Handle* src_handle) {
    check_cpu_handle(src_handle);
    TensorND src = src0, dst = dst0;
    check_layout_and_canonize(src.layout, dst.layout);

    relayout::TransposeParam trans_param;
    bool trans = relayout::is_transpose(src.layout, dst.layout, trans_param);
    if (trans && trans_param.c == 1 && src0.layout.dtype.size() == 1) {
        auto sptr = static_cast<TransposeByte*>(src.raw_ptr),
             dptr = static_cast<TransposeByte*>(dst.raw_ptr);
        MEGDNN_DISPATCH_CPU_KERN_OPR(
                transpose_fallback::transpose<TransposeByte>(
                        trans_param.batch, trans_param.m, trans_param.n, sptr,
                        dptr));
        return;
    }
    exec_after_preprocess(src, dst, trans ? &trans_param : nullptr);
}

// vim: syntax=cpp.doxygen
