/**
 * \file dnn/src/fallback/conv_bias/im2col/strategy_nopack.cpp
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
              postprocess_mode, PackMode::NO_PACK>::
        packA_kern(const WorkspaceBundle& bundle,
                   const fallback::ConvBiasImpl::NCBKernParam& param,
                   fallback::MatrixMulImpl::KernSizeParam matmulparam,
                   const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                   const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
                   const fallback::MatrixMulImpl::AlgoBase::
                           MatmulDescription& /*matmul_dsec*/,
                   const StrategyParam&) {
    MEGDNN_MARK_USED_VAR(bundle);
    MEGDNN_MARK_USED_VAR(param);
    MEGDNN_MARK_USED_VAR(matmulparam);
    MEGDNN_MARK_USED_VAR(matmul_algo);
    MEGDNN_MARK_USED_VAR(ncb_index);
    megdnn_throw(
            "nopack mode should not call packA_kern please check your code");
}

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void* Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
               postprocess_mode, PackMode::NO_PACK>::
        get_matmul_dst_ptr(const fallback::ConvBiasImpl::NCBKernParam& param,
                           const WorkspaceBundle& bundle_thread,
                           const StrategyParam& sparam) {
    if (sparam.is_dst_8bit || !sparam.is_ohw_size_bigger) {
        return static_cast<bias_ctype*>(
                bundle_thread.get(THREAD_BUNDLE_MATMULDST_INDEX));
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
              postprocess_mode, PackMode::NO_PACK>::
        exec_matmul(const fallback::ConvBiasImpl::NCBKernParam& param,
                    const StrategyParam& sparam, const WorkspaceBundle& bundle,
                    const WorkspaceBundle& bundle_thread,
                    fallback::MatrixMulImpl::KernParam matmul_param,
                    const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                    const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
                    const fallback::MatrixMulImpl::AlgoBase::
                            MatmulDescription& /*matmul_desc*/
        ) {
    MEGDNN_MARK_USED_VAR(bundle);
    MEGDNN_MARK_USED_VAR(ncb_index);
    matmul_param.workspace_ptr = bundle_thread.get(THREAD_BUNDLE_MATCOMP_INDEX);
    void* matmul_dst = get_matmul_dst_ptr(param, bundle_thread, sparam);

    src_ctype* im2col_dst = static_cast<src_ctype*>(
            bundle_thread.get(THREAD_BUNDLE_IM2COL_INDEX));
    const void* filter = param.filter<src_ctype>(sparam.group_id) +
                         sparam.oc_cur_index * param.filter_meta.icpg *
                                 param.filter_meta.spatial[0] *
                                 param.filter_meta.spatial[1];
    matmul_param.M = sparam.output_block_oc_size;
    matmul_param.N = sparam.output_block_size;
    matmul_param.LDB = sparam.output_block_size;
    matmul_param.LDC = sparam.output_block_size;
    matmul_param.A_ptr = filter;
    matmul_param.B_ptr = im2col_dst;
    matmul_param.C_ptr = matmul_dst;
    auto matmul_kern = matmul_algo->get_kern(matmul_param);
    matmul_kern(matmul_param);
}

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
              postprocess_mode, PackMode::NO_PACK>::
        exec_im2col(const WorkspaceBundle& bundle,
                    const WorkspaceBundle& bundle_thread,
                    const StrategyParam& sparam,
                    const fallback::ConvBiasImpl::NCBKernParam& param,
                    fallback::MatrixMulImpl::KernParam matmul_param,
                    const fallback::MatrixMulImpl::AlgoBase* matmul_algo) {
    MEGDNN_MARK_USED_VAR(matmul_param);
    MEGDNN_MARK_USED_VAR(matmul_algo);
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
}

#define INSTANTIAL_CLASS(_src_ctype, _bias_ctype, _dst_ctype, _op_ctype,    \
                         _op_dtype, _postprocess_mode)                      \
    template class Strategy<_src_ctype, _bias_ctype, _dst_ctype, _op_ctype, \
                            _op_dtype, _postprocess_mode, PackMode::NO_PACK>;

INSTANTIAL_CLASS(dt_float32, dt_float32, dt_float32, dt_float32, dt_float32,
                 megdnn::PostprocessMode::FLOAT)
INSTANTIAL_CLASS(dt_int8, dt_int16, dt_int16, dt_int16, dt_int16,
                 megdnn::PostprocessMode::ADD_BIAS)
INSTANTIAL_CLASS(dt_int8, dt_int32, dt_int32, dt_int32, dt_int32,
                 megdnn::PostprocessMode::ADD_BIAS)
#if !MEGDNN_DISABLE_FLOAT16
INSTANTIAL_CLASS(dt_float16, dt_float16, dt_float16, dt_float16, dt_float16,
                 megdnn::PostprocessMode::NO_PROCESS)
#endif
#undef INSTANTIAL_CLASS
}  // namespace megdnn

// vim: syntax=cpp.doxygen
