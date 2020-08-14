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

namespace megdnn {

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
              postprocess_mode, PackMode::DEFAULT>::
        packA_kern(const WorkspaceBundle& bundle,
                   const fallback::ConvBiasImpl::NCBKernParam& param,
                   fallback::MatrixMulImpl::KernSizeParam matmulparam,
                   const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                   const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
                   const fallback::MatrixMulImpl::AlgoBase::MatmulDescription&
                           matmul_desc,
                   const StrategyParam& sparam) {
    fallback::MatrixMulImpl::KernParam matmul_param;
    size_t group_id = ncb_index.ndrange_id[0];
    static_cast<fallback::MatrixMulImpl::KernSizeParam&>(matmul_param) =
            matmulparam;
    size_t packed_per_oc_block_size =
            round_up(matmul_param.K, matmul_desc.innerblocksize.k) *
            matmul_desc.innerblocksize.m * matmul_desc.packa_type_size;

    size_t a_panel_offset = ncb_index.ndrange_id[1] * packed_per_oc_block_size;
    int8_t* tmp_ptr =
            sparam.enable_filter_preprocess
                    ? static_cast<int8_t*>(
                              param.preprocessed_filter->tensors[0].raw_ptr)
                    : static_cast<int8_t*>(bundle.get(BUNDLE_PACKA_INDEX));
    int8_t* a_panel =
            tmp_ptr + group_id * sparam.packA_group_size + a_panel_offset;
    matmul_param.A_ptr =
            const_cast<src_ctype*>(param.filter<src_ctype>(group_id));
    matmul_algo->pack_A(matmul_param, a_panel, ncb_index.ndrange_id[1],
                        matmul_desc.innerblocksize.m);
}

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
              postprocess_mode, PackMode::DEFAULT>::
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
    if (sh == 1 && sw == 1) {
        if (is_xcorr) {
            img2col<true>(src2, im2col_dst, oc, oh, ow, ic, ih, iw, fh, fw,
                          sparam.ohw_cur_index, sparam.output_block_size);
        } else {
            img2col<false>(src2, im2col_dst, oc, oh, ow, ic, ih, iw, fh, fw,
                           sparam.ohw_cur_index, sparam.output_block_size);
        }
    } else {
        if (is_xcorr) {
            img2col_stride<true>(src2, im2col_dst, oc, oh, ow, ic, ih, iw, fh,
                                 fw, sh, sw, sparam.ohw_cur_index,
                                 sparam.output_block_size);
        } else {
            img2col_stride<false>(src2, im2col_dst, oc, oh, ow, ic, ih, iw, fh,
                                  fw, sh, sw, sparam.ohw_cur_index,
                                  sparam.output_block_size);
        }
    }
    matmul_param.M = sparam.output_block_oc_size;
    matmul_param.N = sparam.output_block_size;
    matmul_param.LDB = sparam.output_block_size;
    matmul_param.LDC = sparam.output_block_size;
    matmul_param.B_ptr = im2col_dst;

    src_ctype* b_panel =
            reinterpret_cast<src_ctype*>(reinterpret_cast<uintptr_t>(
                    bundle_thread.get(THREAD_BUNDLE_PACKB_INDEX)));
    matmul_algo->pack_B(matmul_param, b_panel, 0, matmul_param.N);
}

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void* Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
               postprocess_mode, PackMode::DEFAULT>::
        get_matmul_dst_ptr(const fallback::ConvBiasImpl::NCBKernParam& param,
                           const WorkspaceBundle& bundle_thread,
                           const StrategyParam& sparam) {
    if (sparam.is_dst_8bit || !sparam.is_ohw_size_bigger) {
        return static_cast<void*>(
                bundle_thread.get(THREAD_BUNDLE_IM2COL_INDEX));
    } else {
        bias_ctype* dst =
                param.dst<bias_ctype>(sparam.batch_id, sparam.group_id) +
                sparam.oc_cur_index * sparam.ohw;
        return static_cast<void*>(dst);
    }
}

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
              postprocess_mode, PackMode::DEFAULT>::
        exec_matmul(const fallback::ConvBiasImpl::NCBKernParam& param,
                    const StrategyParam& sparam, const WorkspaceBundle& bundle,
                    const WorkspaceBundle& bundle_thread,
                    fallback::MatrixMulImpl::KernParam matmul_param,
                    const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                    const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
                    const fallback::MatrixMulImpl::AlgoBase::MatmulDescription&
                            matmul_desc) {
    size_t packA_per_oc_block_size =
            round_up(matmul_param.K, matmul_desc.innerblocksize.k) *
            sparam.oc_tile_size * matmul_desc.packa_type_size;
    size_t packA_group_size = sparam.packA_group_size;
    size_t a_panel_offset = ncb_index.ndrange_id[1] * packA_group_size +
                            ncb_index.ndrange_id[3] * packA_per_oc_block_size;

    void* matmul_dst = get_matmul_dst_ptr(param, bundle_thread, sparam);

    int8_t* tmp_ptr =
           sparam.enable_filter_preprocess
                    ? static_cast<int8_t*>(
                              param.preprocessed_filter->tensors[0].raw_ptr)
                    : static_cast<int8_t*>(bundle.get(BUNDLE_PACKA_INDEX));

    src_ctype* a_panel =
                    reinterpret_cast<src_ctype*>(tmp_ptr + a_panel_offset);
    src_ctype* b_panel =
            reinterpret_cast<src_ctype*>(reinterpret_cast<uintptr_t>(
                    bundle_thread.get(THREAD_BUNDLE_PACKB_INDEX)));
    size_t pack_oc_size = sparam.pack_oc_size;
    matmul_param.M = sparam.output_block_oc_size;
    matmul_param.N = sparam.output_block_size;
    matmul_param.LDB = pack_oc_size * sparam.output_block_size;
    matmul_param.LDC = pack_oc_size * sparam.output_block_size;
    matmul_param.C_ptr = matmul_dst;

    auto matmul_kern_naked = matmul_algo->get_kern_naked(matmul_param);
    matmul_kern_naked(matmul_param, a_panel, b_panel);
}

#define INSTANTIAL_CLASS(_src_ctype, _bias_ctype, _dst_ctype, _op_ctype,    \
                         _op_dtype, _postprocess_mode)                      \
    template class Strategy<_src_ctype, _bias_ctype, _dst_ctype, _op_ctype, \
                            _op_dtype, _postprocess_mode, PackMode::DEFAULT>;

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
