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
              postprocess_mode, PackMode::DEFAULT>::
        copy_padding_kern(WorkspaceBundle bundle,
                          const fallback::ConvBiasImpl::NCBKernParam& param,
                          const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
                          size_t pack_oc_size) {
    UNPACK_CONV_F32_NCB_KERN_SIZES(param);
    MEGDNN_MARK_USED_VAR(N);
    MEGDNN_MARK_USED_VAR(OC);
    MEGDNN_MARK_USED_VAR(OH);
    MEGDNN_MARK_USED_VAR(OW);
    MEGDNN_MARK_USED_VAR(FH);
    MEGDNN_MARK_USED_VAR(FW);
    MEGDNN_MARK_USED_VAR(SH);
    MEGDNN_MARK_USED_VAR(SW);

    size_t IW2 = IW + 2 * PW;
    size_t IH2 = IH + 2 * PH;
    size_t batch_id = ncb_index.ndrange_id[0];
    size_t group_id = ncb_index.ndrange_id[1];
    size_t channel_id = ncb_index.ndrange_id[2];
    size_t PH_SIZE = PH * IW2 * pack_oc_size;

    PW = PW * pack_oc_size;
    IW = IW * pack_oc_size;

    size_t padding_group_size = IH2 * IW2 * IC;
    size_t workspace_channel_offset = pack_oc_size * IH2 * IW2 * channel_id;
    size_t workspace_group_offset = group_id * padding_group_size;
    size_t workspace_batch_offset =
            param.filter_meta.group * batch_id * padding_group_size;
    bundle.set(param.workspace_ptr);

    src_ctype src_zp = static_cast<src_ctype>(0);
    if (param.src_type.enumv() == DTypeEnum::Quantized8Asymm) {
        src_zp = param.src_type.param<dtype::Quantized8Asymm>().zero_point;
    }
    src_ctype* src = const_cast<src_ctype*>(param.src<src_ctype>(
            batch_id, group_id, channel_id, 1, pack_oc_size));
    src_ctype* src2;
    src2 = static_cast<src_ctype*>(bundle.get(BUNDLE_PADDING_INDEX)) +
           workspace_group_offset + workspace_batch_offset +
           workspace_channel_offset;
    src_ctype* src2_ptr = src2;
    const src_ctype* src_ptr = src;
    if (PH != 0) {
        std::memset(src2_ptr, src_zp, sizeof(src_ctype) * PH_SIZE);
        src2_ptr += PH_SIZE;
    }
    rep(ih, IH) {
        if (PW != 0)
            rep(pw, PW) * (src2_ptr++) = src_zp;
        std::memcpy(src2_ptr, src_ptr, sizeof(src_ctype) * IW);
        src2_ptr += IW;
        src_ptr += IW;
        if (PW != 0)
            rep(pw, PW) * (src2_ptr++) = src_zp;
    }
    if (PH != 0) {
        std::memset(src2_ptr, src_zp, sizeof(src_ctype) * PH_SIZE);
        src2_ptr += PH_SIZE;
    }
}

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
              postprocess_mode, PackMode::DEFAULT>::
        packA_kern(WorkspaceBundle bundle,
                   const fallback::ConvBiasImpl::NCBKernParam& param,
                   fallback::MatrixMulImpl::KernSizeParam matmulparam,
                   fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                   const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
                   size_t) {
    bundle.set(param.workspace_ptr);
    fallback::MatrixMulImpl::KernParam matmul_param;
    size_t group_id = ncb_index.ndrange_id[0];
    static_cast<fallback::MatrixMulImpl::KernSizeParam&>(matmul_param) =
            matmulparam;
    size_t packA_group_size = matmul_algo->get_bundle(matmul_param).get_size(0);
    size_t packed_per_oc_block_size =
            round_up(matmul_param.K, matmul_algo->get_inner_block_size().k) *
            matmul_algo->get_inner_block_size().m *
            matmul_algo->get_packA_type_size();
    size_t a_panel_offset = ncb_index.ndrange_id[1] * packed_per_oc_block_size;
    int8_t* a_panel = static_cast<int8_t*>(bundle.get(BUNDLE_PACKA_INDEX)) +
                      group_id * packA_group_size + a_panel_offset;
    matmul_param.A_ptr =
            const_cast<src_ctype*>(param.filter<src_ctype>(group_id));
    matmul_algo->pack_A(matmul_param, a_panel, ncb_index.ndrange_id[1],
                        matmul_algo->get_inner_block_size().m);
}

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
              postprocess_mode, PackMode::DEFAULT>::
        exec_im2col(WorkspaceBundle bundle, WorkspaceBundle bundle_thread,
                    const StrategyParam& sparam,
                    const fallback::ConvBiasImpl::NCBKernParam& param,
                    fallback::MatrixMulImpl::KernParam matmul_param,
                    fallback::MatrixMulImpl::AlgoBase* matmul_algo) {
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
                    const StrategyParam& sparam, WorkspaceBundle bundle,
                    WorkspaceBundle bundle_thread,
                    fallback::MatrixMulImpl::KernParam matmul_param,
                    fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                    const fallback::ConvBiasImpl::NCBKernIndex& ncb_index) {
    size_t packA_per_oc_block_size =
            round_up(matmul_param.K, matmul_algo->get_inner_block_size().k) *
            sparam.oc_tile_size * matmul_algo->get_packA_type_size();
    size_t packA_group_size = matmul_algo->get_bundle(matmul_param).get_size(0);
    size_t a_panel_offset = ncb_index.ndrange_id[1] * packA_group_size +
                            ncb_index.ndrange_id[3] * packA_per_oc_block_size;

    void* matmul_dst = get_matmul_dst_ptr(param, bundle_thread, sparam);

    src_ctype* a_panel = reinterpret_cast<src_ctype*>(
            reinterpret_cast<uintptr_t>(bundle.get(BUNDLE_PACKA_INDEX)) +
            a_panel_offset);
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

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
              postprocess_mode, PackMode::DEFAULT>::
        exec_postprocess(const fallback::ConvBiasImpl::NCBKernParam& param,
                         const StrategyParam& sparam,
                         WorkspaceBundle bundle_thread) {
    copy_bias(param, bundle_thread, sparam);
    void* matmul_dst = get_matmul_dst_ptr(param, bundle_thread, sparam);

    const bias_ctype* bias_ptr = static_cast<const bias_ctype*>(
            param.bias<bias_ctype>(sparam.batch_id, sparam.group_id));
    void* bias_temp_ptr = get_bias_temp_ptr(param, bundle_thread);
    void* bias_preprocess_ptr = const_cast<void*>(
            param.bias_mode == megdnn::BiasMode::BIAS
                    ? bias_temp_ptr
                    : static_cast<void*>(const_cast<bias_ctype*>(
                              bias_ptr + sparam.oc_cur_index)));
    size_t pack_oc_size = sparam.pack_oc_size;
    PostProcess<op_ctype, op_dtype, postprocess_mode>::run(
            matmul_dst, bias_preprocess_ptr, matmul_dst, param.bias_mode,
            param.nonlineMode, param.bias_type, param.dst_type, 1_z,
            sparam.output_block_oc_size / pack_oc_size, 1_z,
            sparam.output_block_size, pack_oc_size);
    copy_dst(param, matmul_dst, sparam);
}

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
              postprocess_mode, PackMode::DEFAULT>::
        copy_dst(const fallback::ConvBiasImpl::NCBKernParam& param,
                 const void* matmul_dst, const StrategyParam& sparam) {
    if (!sparam.skip_copy_dst) {
        size_t pack_oc_size = sparam.pack_oc_size;
        dst_ctype* dst_tmp_ptr =
                reinterpret_cast<dst_ctype*>(const_cast<void*>(matmul_dst));
        dst_ctype* dst =
                param.dst<dst_ctype>(sparam.batch_id, sparam.group_id) +
                sparam.oc_cur_index * sparam.ohw +
                sparam.ohw_cur_index * pack_oc_size;
        size_t oc_loop = sparam.output_block_oc_size / pack_oc_size;
        for (size_t oc = 0; oc < oc_loop; oc++) {
            std::memcpy(dst, dst_tmp_ptr,
                        sizeof(dst_ctype) * sparam.output_block_size *
                                pack_oc_size);
            dst_tmp_ptr += sparam.output_block_size * pack_oc_size;
            dst += sparam.ohw * pack_oc_size;
        }
    }
}

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void* Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
               postprocess_mode, PackMode::DEFAULT>::
        get_bias_temp_ptr(const fallback::ConvBiasImpl::NCBKernParam& param,
                          const WorkspaceBundle& bundle_thread) {
    bias_ctype* bias_tmp_ptr =
            param.bias_mode == megdnn::BiasMode::BIAS
                    ? static_cast<bias_ctype*>(
                              bundle_thread.get(THREAD_BUNDLE_BIAS_INDEX))
                    : nullptr;
    return bias_tmp_ptr;
}

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
              postprocess_mode, PackMode::DEFAULT>::
        copy_bias(const fallback::ConvBiasImpl::NCBKernParam& param,
                  WorkspaceBundle bundle_thread, const StrategyParam& sparam) {
    const bias_ctype* bias_ptr = static_cast<const bias_ctype*>(
            param.bias<bias_ctype>(sparam.batch_id, sparam.group_id));
    bias_ctype* bias_temp_ptr =
            static_cast<bias_ctype*>(get_bias_temp_ptr(param, bundle_thread));
    if (param.bias_mode == megdnn::BiasMode::BIAS) {
        bias_ctype* copy_dst = bias_temp_ptr;
        const bias_ctype* copy_src = bias_ptr +
                                     sparam.oc_cur_index * sparam.ohw +
                                     sparam.ohw_cur_index;
        for (size_t oc = sparam.oc_cur_index; oc < sparam.oc_end_index; oc++) {
            std::memcpy(copy_dst, copy_src,
                        sizeof(bias_ctype) * sparam.output_block_size);
            copy_dst += sparam.output_block_size;
            copy_src += sparam.ohw;
        }
    }
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
#else
#if !MEGDNN_DISABLE_FLOAT16
INSTANTIAL_CLASS(dt_float16, dt_float16, dt_float16, dt_float16, dt_float16,
                 megdnn::PostprocessMode::NO_PROCESS)
#endif
#endif

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
//! x86 do not have uint8 matmul so only armv7 armv8 support uint8
INSTANTIAL_CLASS(dt_uint8, dt_int32, dt_uint8, dt_qint32, dt_quint8,
                 megdnn::PostprocessMode::QUANTIZED)
INSTANTIAL_CLASS(dt_uint8, dt_int32, dt_int32, dt_qint32, dt_qint32,
                 megdnn::PostprocessMode::NO_PROCESS)
#endif

INSTANTIAL_CLASS(dt_int8, dt_int32, dt_int8, dt_qint32, dt_qint8,
                 megdnn::PostprocessMode::QUANTIZED)
INSTANTIAL_CLASS(dt_int8, dt_int32, dt_int32, dt_int32, dt_int32,
                 megdnn::PostprocessMode::NO_PROCESS)
INSTANTIAL_CLASS(dt_int8, dt_int16, dt_int16, dt_int16, dt_int16,
                 megdnn::PostprocessMode::NO_PROCESS)
INSTANTIAL_CLASS(dt_int8, dt_int32, dt_int32, dt_qint32, dt_qint32,
                 megdnn::PostprocessMode::NO_PROCESS)

#undef INSTANTIAL_CLASS
}  // namespace megdnn

// vim: syntax=cpp.doxygen
