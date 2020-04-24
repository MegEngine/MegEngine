/**
 * \file dnn/src/arm_common/local/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/arm_common/local/opr_impl.h"

#include <cstring>
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace arm_common;

namespace {

void do_one_pixel(float* dst, const float* filter, float sval, int OC) {
    const int width = 4u;
    int oc = 0;
    float32x4_t vs = vdupq_n_f32(sval);
    for (; oc + width <= OC; oc += width, filter += width, dst += width) {
        float32x4_t vf = vld1q_f32(filter);
        float32x4_t vd = vld1q_f32(dst);
        vd = vmlaq_f32(vd, vs, vf);
        vst1q_f32(dst, vd);
    }
    for (; oc < OC; oc++, dst++, filter++) {
        *dst += sval * (*filter);
    }
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void do_one_pixel(dt_float16* dst, const dt_float16* filter, dt_float16 sval,
                  int OC) {
    const __fp16* filter_ptr = reinterpret_cast<const __fp16*>(filter);
    __fp16* dst_ptr = reinterpret_cast<__fp16*>(dst);
    const int width = 8u;
    int oc = 0;
    float16x8_t vs = vdupq_n_f16(sval);
    for (; oc + width <= OC;
         oc += width, filter_ptr += width, dst_ptr += width) {
        float16x8_t vf = vld1q_f16(filter_ptr);
        float16x8_t vd = vld1q_f16(dst_ptr);
        vd = vmlaq_f16(vd, vs, vf);
        vst1q_f16(dst_ptr, vd);
    }
#if MEGDNN_FIX_AARCH32_BUG
    // FIXME: as llvm may cause cannot select error if enable vectorize
    #pragma clang loop vectorize(disable)
#endif
    for (; oc < OC; oc++, dst_ptr++, filter_ptr++) {
        *dst_ptr += sval * (*filter_ptr);
    }
}
#endif

template <bool is_xcorr, typename dtype>
void exec_internal(const LocalImpl::FloatNoncontigBatchKernParam& kparam) {
    UNPACK_LOCAL_FLOAT_NONCONTIG_BATCH_KERN_PARAM(kparam, dtype);
    auto dst2 = workspace;
    // dst2 is (H, W, N, C)
    std::memset(dst2, 0, sizeof(dtype) * OH * OW * N * OC);
    dtype* dst2_hwnc = dst2;
    rep(oh, OH) rep(ow, OW) {
        const dtype* src_bak = src;
        rep(ic, IC) {
            rep(fh, FH) for (int fw = 0; fw < FW; ++fw, filter += OC) {
                int ih = -PH + oh * SH + (is_xcorr ? fh : (FH - fh - 1));
                int iw = -PW + ow * SW + (is_xcorr ? fw : (FW - fw - 1));
                if (ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                    continue;
                dtype* dst2_bak = dst2;
                rep(n, N) {
                    dtype s = src[n * INP_BS + ih * IW + iw];
                    do_one_pixel(dst2, filter, s, OC);
                    dst2 += OC;
                }
                dst2 = dst2_bak;
            }
            src += IH * IW;
        }
        src = src_bak;
        dst2 += N * OC;
    }
    transpose_knc2nsck(dst2_hwnc, dst, OH * OW, N, OC, OUT_BS);
}

}  // anonymous namespace

size_t LocalImpl::get_workspace_in_bytes(const TensorLayout& /* src */,
                                         const TensorLayout& /* filter */,
                                         const TensorLayout& dst) {
    return dst.span().dist_byte();
}

LocalImpl::float_noncontig_batch_kern LocalImpl::dispatch_float_noncontig_batch(
        const TensorLayout& src, const TensorLayout&, const TensorLayout&) {
    megdnn_assert(src.stride[0] > 0 &&
                  static_cast<size_t>(src.stride[0]) >=
                          src.total_nr_elems() / src.shape[0]);
    if (src.dtype == dtype::Float32()) {
        if (param().mode == Mode::CROSS_CORRELATION) {
            return exec_internal<true, float>;
        } else {
            return exec_internal<false, float>;
        }
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    } else {
        megdnn_assert(src.dtype == dtype::Float16());
        if (param().mode == Mode::CROSS_CORRELATION) {
            return exec_internal<true, dt_float16>;
        } else {
            return exec_internal<false, dt_float16>;
        }
#endif
    }
    megdnn_assert_internal(false);
    return nullptr;
}

void LocalImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                     _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    return exec_use_float_noncontig_batch(src, filter, dst, workspace);
}

// vim: syntax=cpp.doxygen
