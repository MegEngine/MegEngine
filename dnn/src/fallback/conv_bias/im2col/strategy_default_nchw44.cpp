/**
 * \file dnn/src/fallback/conv_bias/im2col/strategy_default.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/fallback/conv_bias/im2col/strategy_base.h"
#include "src/fallback/convolution/img2col_helper.h"
#if MEGDNN_X86
#include "src/x86/conv_bias/postprocess_helper.h"
#elif (MEGDNN_ARMV7 || MEGDNN_AARCH64)
#include "src/arm_common/conv_bias/postprocess_helper.h"
#else
#include "src/common/postprocess_helper.h"
#endif

using namespace megdnn;
#if MEGDNN_X86
using namespace x86;
#endif
namespace megdnn {

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
              postprocess_mode, PackMode::DEFAULT, FormatMode::NCHW44>::
        exec_im2col(const WorkspaceBundle& bundle,
                    const WorkspaceBundle& bundle_thread,
                    const StrategyParam& sparam,
                    const fallback::ConvBiasImpl::NCBKernParam& param,
                    fallback::MatrixMulImpl::KernParam matmul_param,
                    const fallback::MatrixMulImpl::AlgoBase* matmul_algo) {
    size_t sh = param.filter_meta.stride[0];
    size_t sw = param.filter_meta.stride[1];
    size_t oc = param.filter_meta.ocpg;
    size_t oh = param.osz[0];
    size_t ow = param.osz[1];
    size_t ic = param.filter_meta.icpg;
    size_t ih = param.isz[0] + param.filter_meta.padding[0] * 2;
    size_t iw = param.isz[1] + param.filter_meta.padding[1] * 2;
    size_t fh = param.filter_meta.spatial[0];
    size_t fw = param.filter_meta.spatial[1];
    size_t is_xcorr = !param.filter_meta.should_flip;
    constexpr static size_t pack_size = 4;
    size_t input_offset =
            ih * iw * ic *
            (sparam.group_id + param.filter_meta.group * sparam.batch_id) *
            sizeof(src_ctype);

    src_ctype* src2 = reinterpret_cast<src_ctype*>(
            reinterpret_cast<uintptr_t>(bundle.get(BUNDLE_PADDING_INDEX)) +
            input_offset);
    bool is_phpwzero = param.filter_meta.padding[0] == 0 &&
                       param.filter_meta.padding[1] == 0;
    if (is_phpwzero) {
        src2 = const_cast<src_ctype*>(
                param.src<src_ctype>(sparam.batch_id, sparam.group_id));
    }
    src_ctype* im2col_dst = static_cast<src_ctype*>(
            bundle_thread.get(THREAD_BUNDLE_IM2COL_INDEX));
    if (is_xcorr) {
        if (sh == sw && sh == 1) {
            img2col_nchw4<true>(src2, im2col_dst, oc, oh, ow, ic, ih, iw, fh,
                                fw, sh, sw, sparam.ohw_cur_index,
                                sparam.output_block_size);
        } else {
          img2col_stride_nchw4<true>(src2, im2col_dst, oc, oh, ow, ic, ih, iw,
                                     fh, fw, sh, sw, sparam.ohw_cur_index,
                                     sparam.output_block_size);
        }
    } else {
        if (sh == sw && sh == 1) {
            img2col_nchw4<false>(src2, im2col_dst, oc, oh, ow, ic, ih, iw, fh,
                                 fw, sh, sw, sparam.ohw_cur_index,
                                 sparam.output_block_size);
        } else {
          img2col_stride_nchw4<false>(
                  src2, im2col_dst, oc, oh, ow, ic, ih, iw, fh, fw, sh, sw,
                  sparam.ohw_cur_index, sparam.output_block_size);
        }
    }
    matmul_param.M = sparam.output_block_oc_size;
    matmul_param.N = sparam.output_block_size;
    matmul_param.LDB = pack_size * sparam.output_block_size;
    matmul_param.LDC = pack_size * sparam.output_block_size;
    matmul_param.B_ptr = im2col_dst;

    src_ctype* b_panel =
            reinterpret_cast<src_ctype*>(reinterpret_cast<uintptr_t>(
                    bundle_thread.get(THREAD_BUNDLE_PACKB_INDEX)));
    matmul_algo->pack_B(matmul_param, b_panel, 0, matmul_param.N);
}

#define INSTANTIAL_CLASS(_src_ctype, _bias_ctype, _dst_ctype, _op_ctype,     \
                         _op_dtype, _postprocess_mode)                       \
    template class Strategy<_src_ctype, _bias_ctype, _dst_ctype, _op_ctype,  \
                            _op_dtype, _postprocess_mode, PackMode::DEFAULT, \
                            FormatMode::NCHW44>;

INSTANTIAL_CLASS(dt_float32, dt_float32, dt_float32, dt_float32, dt_float32,
                 megdnn::PostprocessMode::FLOAT)

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
INSTANTIAL_CLASS(dt_float16, dt_float16, dt_float16, __fp16, __fp16,
                 megdnn::PostprocessMode::FLOAT)
#endif
#if !MEGDNN_DISABLE_FLOAT16
INSTANTIAL_CLASS(dt_float16, dt_float16, dt_float16, dt_float16, dt_float16,
                 megdnn::PostprocessMode::NO_PROCESS)
#endif

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
//! x86 do not have uint8 matmul so only armv7 armv8 support uint8
INSTANTIAL_CLASS(dt_uint8, dt_int32, dt_uint8, dt_qint32, dt_quint8,
                 megdnn::PostprocessMode::QUANTIZED)
INSTANTIAL_CLASS(dt_uint8, dt_int32, dt_int32, dt_int32, dt_int32,
                 megdnn::PostprocessMode::ADD_BIAS)
#endif

INSTANTIAL_CLASS(dt_int8, dt_int32, dt_int8, dt_qint32, dt_qint8,
                 megdnn::PostprocessMode::QUANTIZED)
INSTANTIAL_CLASS(dt_int8, dt_int32, dt_int32, dt_int32, dt_int32,
                 megdnn::PostprocessMode::ADD_BIAS)
INSTANTIAL_CLASS(dt_int8, dt_int16, dt_int16, dt_int16, dt_int16,
                 megdnn::PostprocessMode::ADD_BIAS)
#undef INSTANTIAL_CLASS
}  // namespace megdnn

// vim: syntax=cpp.doxygen
