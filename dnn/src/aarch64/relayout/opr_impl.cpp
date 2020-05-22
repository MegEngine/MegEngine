/**
 * \file dnn/src/aarch64/relayout/opr_impl.cpp
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

#include "src/aarch64/handle.h"
#include "src/aarch64/relayout/opr_impl.h"

using namespace megdnn;
using namespace relayout;

namespace {

struct TransposeByte {
    uint8_t v;
};

void trans_16x16_u8(const void* src, void* dst, const size_t src_step,
                    const size_t dst_step) {
    asm volatile(
        "\n"
        "ld1 {v0.16b}, [%[src]], %[src_step] \n"
        "ld1 {v1.16b}, [%[src]], %[src_step] \n"
        "ld1 {v2.16b}, [%[src]], %[src_step] \n"
        "ld1 {v3.16b}, [%[src]], %[src_step] \n"
        "ld1 {v4.16b}, [%[src]], %[src_step] \n"
        "ld1 {v5.16b}, [%[src]], %[src_step] \n"
        "ld1 {v6.16b}, [%[src]], %[src_step] \n"
        "ld1 {v7.16b}, [%[src]], %[src_step] \n"
        "ld1 {v8.16b}, [%[src]], %[src_step] \n"
        "ld1 {v9.16b}, [%[src]], %[src_step] \n"
        "ld1 {v10.16b}, [%[src]], %[src_step] \n"
        "ld1 {v11.16b}, [%[src]], %[src_step] \n"
        "ld1 {v12.16b}, [%[src]], %[src_step] \n"
        "ld1 {v13.16b}, [%[src]], %[src_step] \n"
        "ld1 {v14.16b}, [%[src]], %[src_step] \n"
        "ld1 {v15.16b}, [%[src]], %[src_step] \n"
        "trn1 v16.16b, v0.16b, v1.16b \n"
        "trn2 v17.16b, v0.16b, v1.16b \n"
        "trn1 v18.16b, v2.16b, v3.16b \n"
        "trn2 v19.16b, v2.16b, v3.16b \n"
        "trn1 v20.16b, v4.16b, v5.16b \n"
        "trn2 v21.16b, v4.16b, v5.16b \n"
        "trn1 v22.16b, v6.16b, v7.16b \n"
        "trn2 v23.16b, v6.16b, v7.16b \n"
        "trn1 v24.16b, v8.16b, v9.16b \n"
        "trn2 v25.16b, v8.16b, v9.16b \n"
        "trn1 v26.16b, v10.16b, v11.16b \n"
        "trn2 v27.16b, v10.16b, v11.16b \n"
        "trn1 v28.16b, v12.16b, v13.16b \n"
        "trn2 v29.16b, v12.16b, v13.16b \n"
        "trn1 v30.16b, v14.16b, v15.16b \n"
        "trn2 v31.16b, v14.16b, v15.16b \n"
        "trn1 v0.8h, v16.8h, v18.8h \n"
        "trn2 v2.8h, v16.8h, v18.8h \n"
        "trn1 v4.8h, v20.8h, v22.8h \n"
        "trn2 v6.8h, v20.8h, v22.8h \n"
        "trn1 v8.8h, v24.8h, v26.8h \n"
        "trn2 v10.8h, v24.8h, v26.8h \n"
        "trn1 v12.8h, v28.8h, v30.8h \n"
        "trn2 v14.8h, v28.8h, v30.8h \n"
        "trn1 v1.8h, v17.8h, v19.8h \n"
        "trn2 v3.8h, v17.8h, v19.8h \n"
        "trn1 v5.8h, v21.8h, v23.8h \n"
        "trn2 v7.8h, v21.8h, v23.8h \n"
        "trn1 v9.8h, v25.8h, v27.8h \n"
        "trn2 v11.8h, v25.8h, v27.8h \n"
        "trn1 v13.8h, v29.8h, v31.8h \n"
        "trn2 v15.8h, v29.8h, v31.8h \n"
        "trn1 v16.4s, v0.4s, v4.4s \n"
        "trn2 v20.4s, v0.4s, v4.4s \n"
        "trn1 v24.4s, v8.4s, v12.4s \n"
        "trn2 v28.4s, v8.4s, v12.4s \n"
        "trn1 v17.4s, v1.4s, v5.4s \n"
        "trn2 v21.4s, v1.4s, v5.4s \n"
        "trn1 v25.4s, v9.4s, v13.4s \n"
        "trn2 v29.4s, v9.4s, v13.4s \n"
        "trn1 v18.4s, v2.4s, v6.4s \n"
        "trn2 v22.4s, v2.4s, v6.4s \n"
        "trn1 v26.4s, v10.4s, v14.4s \n"
        "trn2 v30.4s, v10.4s, v14.4s \n"
        "trn1 v19.4s, v3.4s, v7.4s \n"
        "trn2 v23.4s, v3.4s, v7.4s \n"
        "trn1 v27.4s, v11.4s, v15.4s \n"
        "trn2 v31.4s, v11.4s, v15.4s \n"
        "trn1 v0.2d, v16.2d, v24.2d \n"
        "trn2 v8.2d, v16.2d, v24.2d \n"
        "trn1 v1.2d, v17.2d, v25.2d \n"
        "trn2 v9.2d, v17.2d, v25.2d \n"
        "trn1 v2.2d, v18.2d, v26.2d \n"
        "trn2 v10.2d, v18.2d, v26.2d \n"
        "trn1 v3.2d, v19.2d, v27.2d \n"
        "trn2 v11.2d, v19.2d, v27.2d \n"
        "trn1 v4.2d, v20.2d, v28.2d \n"
        "trn2 v12.2d, v20.2d, v28.2d \n"
        "trn1 v5.2d, v21.2d, v29.2d \n"
        "trn2 v13.2d, v21.2d, v29.2d \n"
        "trn1 v6.2d, v22.2d, v30.2d \n"
        "trn2 v14.2d, v22.2d, v30.2d \n"
        "trn1 v7.2d, v23.2d, v31.2d \n"
        "trn2 v15.2d, v23.2d, v31.2d \n"
        "st1 {v0.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v1.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v2.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v3.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v4.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v5.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v6.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v7.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v8.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v9.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v10.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v11.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v12.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v13.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v14.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v15.16b}, [%[dst]], %[dst_step] \n"
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

void aarch64::RelayoutForwardImpl::exec(_megdnn_tensor_in src0,
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
