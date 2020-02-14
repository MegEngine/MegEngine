/**
 * \file dnn/src/fallback/conv_bias/im2col/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/conv_bias/im2col/algos.h"
#include "megdnn/opr_param_defs.h"
#include "src/common/opr_delegate.h"
#include "src/fallback/conv_bias/common.h"
#include "src/fallback/conv_bias/opr_impl.h"
#include "src/fallback/conv_bias/winograd/strategy.h"
#include "src/fallback/convolution/img2col_helper.h"
#include "src/naive/convolution/helper.h"
#if MEGDNN_X86
#include "src/x86/conv_bias/postprocess_helper.h"
#endif
#include "midout.h"
MIDOUT_DECL(megdnn_fallback_im2col)

using namespace megdnn;
using namespace fallback;

#if MEGDNN_X86
using namespace x86;
#endif

/*======================== AlgoIm2col=======================*/
/*!
 *  *\brief The index of all parts workspace in im2col workspace bundel
 *  *Through witch can convenient get the needed ptr
 */
struct Im2colBundelIndex {
    static constexpr size_t BUNDLE_PADDING_INDEX = 0_z;
    static constexpr size_t BUNDLE_PACKA_INDEX = 1_z;
    static constexpr size_t BUNDLE_THREAD_INDEX = 2_z;
    static constexpr size_t THREAD_BUNDLE_PACKB_INDEX = 0_z;
    static constexpr size_t THREAD_BUNDLE_IM2COL_INDEX = 1_z;
    static constexpr size_t THREAD_BUNDLE_MATMUL_DST_INDEX = 2_z;
    static constexpr size_t THREAD_BUNDLE_BIAS_INDEX = 3_z;
    static constexpr size_t THREAD_BUNDLE_COMPUTE_INDEX = 4_z;
};

/*!
 *  *\brief PtrGetter is get the im2col needed ptr according to the provided
 *  *conditions
 */
class PtrGetter {
public:
    template <typename dtype>
    static inline dtype* get_matmul_dst_ptr(
            const ConvBiasImpl::NCBKernParam& param,
            const WorkspaceBundle& bundle_thread, size_t bundle_id,
            size_t oc_cur_index, size_t OHW, bool is_dst_8bit,
            bool ohw_bigger_ohwblock) {
        if (is_dst_8bit || !ohw_bigger_ohwblock) {
            return static_cast<dtype*>(bundle_thread.get(bundle_id));
        } else {
            dtype* dst = param.dst<dtype>() + oc_cur_index * OHW;
            return static_cast<dtype*>(dst);
        }
    }

    template <typename bias_ctype>
    static inline bias_ctype* get_bias_temp_ptr(
            const ConvBiasImpl::NCBKernParam& param,
            const WorkspaceBundle& bundle_thread) {
        bias_ctype* bias_tmp_ptr =
                param.bias_mode == megdnn::BiasMode::BIAS
                        ? static_cast<bias_ctype*>(bundle_thread.get(
                                  Im2colBundelIndex::THREAD_BUNDLE_BIAS_INDEX))
                        : nullptr;
        return bias_tmp_ptr;
    }

    template <typename dtype>
    static inline dtype* get_bundle_offset_byte_ptr(
            const WorkspaceBundle& bundle, size_t bundle_id, size_t offset) {
        return reinterpret_cast<dtype*>(
                reinterpret_cast<uintptr_t>(bundle.get(bundle_id)) + offset);
    }
};

using Pack_Mode=fallback::MatrixMulImpl::AlgoBase::PackMode;

//! Process one input channel copy padding
template <typename src_ctype>
static void copy_padding_kern(WorkspaceBundle bundle,
                              const ConvBiasImpl::NCBKernParam& param,
                              ConvBiasImpl::NCBKernIndex ncb_index) {
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

    size_t padding_group_size = IH2 * IW2 * IC;
    size_t input_channel_offset = IH * IW * ncb_index.ndrange_id[2];
    size_t workspace_channel_offset = IH2 * IW2 * ncb_index.ndrange_id[2];
    size_t workspace_group_offset =
            ncb_index.ndrange_id[0] * padding_group_size;
    size_t workspace_batch_offset = param.filter_meta.group *
                                    ncb_index.ndrange_id[1] *
                                    padding_group_size;
    bundle.set(param.workspace_ptr);

    src_ctype src_zp = static_cast<src_ctype>(0);
    if (param.src_type.enumv() == DTypeEnum::Quantized8Asymm) {
        src_zp = param.src_type.param<dtype::Quantized8Asymm>().zero_point;
    }
    src_ctype* src = const_cast<src_ctype*>(param.src<src_ctype>() +
                                            input_channel_offset);
    src_ctype* src2;
    src2 = static_cast<src_ctype*>(
                   bundle.get(Im2colBundelIndex::BUNDLE_PADDING_INDEX)) +
           workspace_group_offset + workspace_batch_offset +
           workspace_channel_offset;
    src_ctype* src2_ptr = src2;
    const src_ctype* src_ptr = src;
    if (PH != 0) {
        std::memset(src2_ptr, src_zp, sizeof(src_ctype) * PH * IW2);
        src2_ptr += PH * IW2;
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
        std::memset(src2_ptr, src_zp, sizeof(src_ctype) * PH * IW2);
        src2_ptr += PH * IW2;
    }
};

/*!
 * *\brief Im2colKerns collects all the im2col kerns in it
 */

#define COPY_BIAS()                                                         \
    const bias_ctype* bias_ptr =                                            \
            static_cast<const bias_ctype*>(param.bias_ptr);                 \
    bias_ctype* bias_temp_ptr =                                             \
            PtrGetter::get_bias_temp_ptr<bias_ctype>(param, bundle_thread); \
    if (param.bias_mode == megdnn::BiasMode::BIAS) {                        \
        bias_ctype* copy_dst = bias_temp_ptr;                               \
        const bias_ctype* copy_src =                                        \
                bias_ptr + oc_cur_index * OH * OW + ohw_cur_index;          \
        for (size_t oc = oc_cur_index; oc < oc_end_index; oc++) {           \
            std::memcpy(copy_dst, copy_src,                                 \
                        sizeof(bias_ctype) * output_block_size);            \
            copy_dst += output_block_size;                                  \
            copy_src += OH * OW;                                            \
        }                                                                   \
    }

#define IM2COL()                                                               \
    src_ctype* im2col_dst = nullptr;                                           \
    src_ctype* no_padding_src =                                                \
            const_cast<src_ctype*>(param.src<src_ctype>()) + ohw_cur_index;    \
    if (!special_1x1) {                                                        \
        size_t padding_group_size = IH2 * IW2 * IC * sizeof(src_ctype);        \
        src_ctype* src2 = PtrGetter::get_bundle_offset_byte_ptr<src_ctype>(    \
                bundle, Im2colBundelIndex::BUNDLE_PADDING_INDEX,               \
                (ncb_index.ndrange_id[0] +                                     \
                 param.filter_meta.group * ncb_index.ndrange_id[1]) *          \
                        padding_group_size);                                   \
        if (PH == 0 && PW == 0) {                                              \
            src2 = const_cast<src_ctype*>(param.src<src_ctype>());             \
        }                                                                      \
        im2col_dst = static_cast<src_ctype*>(bundle_thread.get(                \
                Im2colBundelIndex::THREAD_BUNDLE_IM2COL_INDEX));               \
        if (SH == 1 && SW == 1) {                                              \
            if (is_xcorr) {                                                    \
                img2col<true>(src2, im2col_dst, OC, OH, OW, IC, IH2, IW2, FH,  \
                              FW, ohw_cur_index, output_block_size);           \
            } else {                                                           \
                img2col<false>(src2, im2col_dst, OC, OH, OW, IC, IH2, IW2, FH, \
                               FW, ohw_cur_index, output_block_size);          \
            }                                                                  \
        } else {                                                               \
            if (is_xcorr) {                                                    \
                img2col_stride<true>(src2, im2col_dst, OC, OH, OW, IC, IH2,    \
                                     IW2, FH, FW, SH, SW, ohw_cur_index,       \
                                     output_block_size);                       \
            } else {                                                           \
                img2col_stride<false>(src2, im2col_dst, OC, OH, OW, IC, IH2,   \
                                      IW2, FH, FW, SH, SW, ohw_cur_index,      \
                                      output_block_size);                      \
            }                                                                  \
        }                                                                      \
    }

#define POSTPROCESS_AND_COPYDST()                                            \
    PostProcess<op_ctype, op_dtype, postprocess_mode>::run(                  \
            matmul_dst,                                                      \
            param.bias_mode == megdnn::BiasMode::BIAS                        \
                    ? bias_temp_ptr                                          \
                    : const_cast<bias_ctype*>(bias_ptr + oc_cur_index),      \
            matmul_dst, param.bias_mode, param.nonlineMode, param.bias_type, \
            param.dst_type, 1_z, output_block_oc_size, 1_z,                  \
            output_block_size);                                              \
    if (!skip_copy_dst) {                                                    \
        dst_ctype* dst_tmp_ptr = reinterpret_cast<dst_ctype*>(matmul_dst);   \
        dst_ctype* dst =                                                     \
                param.dst<dst_ctype>() + oc_cur_index * OHW + ohw_cur_index; \
        for (size_t oc = 0; oc < output_block_oc_size; oc++) {               \
            std::memcpy(dst, dst_tmp_ptr,                                    \
                        sizeof(dst_ctype) * output_block_size);              \
            dst_tmp_ptr += output_block_size;                                \
            dst += OHW;                                                      \
        }                                                                    \
    }

#define PREPAR_MATMUL_DATA()                                                  \
    size_t packA_per_oc_block_size =                                          \
            round_up(matmul_param.K, matmul_algo->get_inner_block_size().k) * \
            oc_tile_size * matmul_algo->get_packA_type_size();                \
    size_t packA_group_size =                                                 \
            matmul_algo->get_bundle(matmul_param).get_size(0);                \
    src_ctype* a_panel = PtrGetter::get_bundle_offset_byte_ptr<src_ctype>(    \
            bundle, Im2colBundelIndex::BUNDLE_PACKA_INDEX,                    \
            ncb_index.ndrange_id[0] * packA_group_size +                      \
                    ncb_index.ndrange_id[3] * packA_per_oc_block_size);       \
    src_ctype* b_panel = PtrGetter::get_bundle_offset_byte_ptr<src_ctype>(    \
            bundle_thread, Im2colBundelIndex::THREAD_BUNDLE_PACKB_INDEX, 0);  \
    /*In pack mode, the matmul dst and im2col dst is the same workspace*/     \
    bias_ctype* matmul_dst = PtrGetter::get_matmul_dst_ptr<bias_ctype>(       \
            param, bundle_thread,                                             \
            Im2colBundelIndex::THREAD_BUNDLE_IM2COL_INDEX, oc_cur_index, OHW, \
            is_dst_8bit, is_ohw_size_bigger);

#define MATMUL_COMPUTE()                                                      \
    auto matmul_kern_naked = matmul_algo->get_kern_naked(matmul_param);       \
    matmul_param.M = output_block_oc_size;                                    \
    matmul_param.N = output_block_size;                                       \
    matmul_param.LDB = special_1x1 ? OH * OW : output_block_size;             \
    matmul_param.LDC = output_block_size;                                     \
    matmul_param.A_ptr = a_panel;                                             \
    matmul_param.B_ptr = im2col_dst ? im2col_dst : no_padding_src;            \
    matmul_param.C_ptr = matmul_dst;                                          \
    matmul_algo->pack_B(matmul_param, b_panel, 0, output_block_size);         \
    matmul_kern_naked(matmul_param, a_panel, b_panel);

template <Pack_Mode packmode>
class Im2colKerns;

template <>
class Im2colKerns<Pack_Mode::DEFAULT> {
public:
    //! packA kern
    template <typename src_ctype>
    static void packA_kern(WorkspaceBundle bundle,
                           const ConvBiasImpl::NCBKernParam& param,
                           fallback::MatrixMulImpl::KernSizeParam matmulparam,
                           fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                           ConvBiasImpl::NCBKernIndex ncb_index) {
        bundle.set(param.workspace_ptr);
        fallback::MatrixMulImpl::KernParam matmul_param;
        static_cast<fallback::MatrixMulImpl::KernSizeParam&>(matmul_param) =
                matmulparam;
        size_t packA_group_size =
                matmul_algo->get_bundle(matmul_param).get_size(0);
        size_t packed_per_oc_block_size =
                round_up(matmul_param.K,
                         matmul_algo->get_inner_block_size().k) *
                matmul_algo->get_inner_block_size().m *
                matmul_algo->get_packA_type_size();
        size_t a_panel_offset =
                ncb_index.ndrange_id[2] * packed_per_oc_block_size;
        int8_t* a_panel =
                static_cast<int8_t*>(
                        bundle.get(Im2colBundelIndex::BUNDLE_PACKA_INDEX)) +
                ncb_index.ndrange_id[0] * packA_group_size + a_panel_offset;
        matmul_param.A_ptr = const_cast<src_ctype*>(param.filter<src_ctype>());
        matmul_algo->pack_A(matmul_param, a_panel, ncb_index.ndrange_id[2],
                            matmul_algo->get_inner_block_size().m);
    };

    //! conv kernel
    template <typename src_ctype, typename bias_ctype, typename dst_ctype,
              typename op_ctype, typename op_dtype,
              PostprocessMode postprocess_mode>
    static void kerns(
            WorkspaceBundle bundle, WorkspaceBundle bundle_thread,
            const ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernSizeParam matmul_kernsize_param,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            fallback::ConvBiasImpl::NCBKernIndex ncb_index,
            size_t ohw_tile_size, size_t oc_tile_size) {
        auto is_xcorr = !param.filter_meta.should_flip;
        UNPACK_CONV_F32_NCB_KERN_SIZES(param);
        MEGDNN_MARK_USED_VAR(N);
        auto IH2 = IH + 2 * PH;
        auto IW2 = IW + 2 * PW;
        size_t OHW = OH * OW;
        size_t output_block_size = std::min(
                ohw_tile_size, OHW - ncb_index.ndrange_id[2] * ohw_tile_size);
        size_t output_block_oc_size = std::min(
                oc_tile_size, OC - ncb_index.ndrange_id[3] * oc_tile_size);

        //! misc flags
        bool special_1x1 = (FH == 1 && FW == 1 && SH == 1 && SW == 1 &&
                            PH == 0 && PW == 0);
        bool is_dst_8bit =
                (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                 param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                (param.src_type.enumv() == DTypeEnum::Quantized8Asymm &&
                 param.dst_type.enumv() == DTypeEnum::Quantized8Asymm);
        bool is_ohw_size_bigger = (ohw_tile_size >= OHW);
        bool skip_copy_dst = is_ohw_size_bigger && !is_dst_8bit;

        //! misc index
        size_t ohw_cur_index = ncb_index.ndrange_id[2] * ohw_tile_size;
        size_t oc_cur_index = ncb_index.ndrange_id[3] * oc_tile_size;
        size_t oc_end_index = oc_cur_index + output_block_oc_size;

        bundle.set(param.workspace_ptr);
        bundle_thread.set(PtrGetter::get_bundle_offset_byte_ptr<int8_t>(
                bundle, Im2colBundelIndex::BUNDLE_THREAD_INDEX,
                bundle_thread.total_size_in_bytes() * ncb_index.thread_id));

        fallback::MatrixMulImpl::KernParam matmul_param;
        static_cast<fallback::MatrixMulImpl::KernSizeParam&>(matmul_param) =
                matmul_kernsize_param;
        matmul_param.workspace_ptr = bundle_thread.get(
                Im2colBundelIndex::THREAD_BUNDLE_COMPUTE_INDEX);

        //! 1.Copy bias if need
        COPY_BIAS();

        //! 2.Im2col
        IM2COL();

        //! 3.packb and matmul compute
        PREPAR_MATMUL_DATA();
        MATMUL_COMPUTE();

        //! 4.postprocess and copy dst if need
        POSTPROCESS_AND_COPYDST();
#undef PREPAR_MATMUL_DATA
#undef MATMUL_COMPUTE
    }
};

#define PREPAR_MATMUL_DATA()                                                   \
    bias_ctype* matmul_dst = nullptr;                                          \
    src_ctype* b_panel = nullptr;                                              \
    size_t packA_group_size =                                                  \
            bundle.get_size(Im2colBundelIndex::BUNDLE_PACKA_INDEX) /           \
            param.filter_meta.group;                                           \
    size_t a_panel_offset = ncb_index.ndrange_id[3] *                          \
                            matmul_algo->get_bundle(matmul_param).get_size(0); \
                                                                               \
    src_ctype* a_panel = PtrGetter::get_bundle_offset_byte_ptr<src_ctype>(     \
            bundle, Im2colBundelIndex::BUNDLE_PACKA_INDEX,                     \
            ncb_index.ndrange_id[0] * packA_group_size + a_panel_offset);      \
    matmul_dst = PtrGetter::get_matmul_dst_ptr<bias_ctype>(                    \
            param, bundle_thread,                                              \
            Im2colBundelIndex::THREAD_BUNDLE_MATMUL_DST_INDEX, oc_cur_index,   \
            OHW, is_dst_8bit, is_ohw_size_bigger);

#define MATMUL_COMPUTE()                                                      \
    auto matmul_kern_naked = matmul_algo->get_kern_naked(matmul_param);       \
    matmul_param.M = output_block_oc_size;                                    \
    matmul_param.N = output_block_size;                                       \
    matmul_param.LDB = special_1x1 ? OH * OW : output_block_size;             \
    matmul_param.LDC = output_block_size;                                     \
    matmul_param.A_ptr = a_panel;                                             \
    matmul_param.B_ptr = im2col_dst ? im2col_dst : no_padding_src;            \
    matmul_param.C_ptr = matmul_dst;                                          \
    matmul_kern_naked(matmul_param, a_panel, b_panel);

template <>
class Im2colKerns<Pack_Mode::ONLY_PACKA> {
public:
    //! packA kern
    template <typename src_ctype>
    static void packA_kern(WorkspaceBundle bundle,
                           const ConvBiasImpl::NCBKernParam& param,
                           fallback::MatrixMulImpl::KernSizeParam matmulparam,
                           fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                           ConvBiasImpl::NCBKernIndex ncb_index) {
        bundle.set(param.workspace_ptr);
        fallback::MatrixMulImpl::KernParam matmul_param;
        static_cast<fallback::MatrixMulImpl::KernSizeParam&>(matmul_param) =
                matmulparam;
        size_t OC = param.filter_meta.ocpg;
        size_t oc_tile_size = matmul_param.M;
        size_t output_block_oc_size = std::min(
                oc_tile_size, OC - ncb_index.ndrange_id[2] * oc_tile_size);
        size_t oc_cur_index = ncb_index.ndrange_id[2] * oc_tile_size;
        size_t packA_group_size =
                bundle.get_size(Im2colBundelIndex::BUNDLE_PACKA_INDEX) /
                param.filter_meta.group;
        size_t a_panel_offset =
                ncb_index.ndrange_id[2] *
                matmul_algo->get_bundle(matmul_param).get_size(0);
        int8_t* a_panel =
                static_cast<int8_t*>(
                        bundle.get(Im2colBundelIndex::BUNDLE_PACKA_INDEX)) +
                ncb_index.ndrange_id[0] * packA_group_size + a_panel_offset;
        matmul_param.A_ptr = const_cast<src_ctype*>(param.filter<src_ctype>()) +
                             oc_cur_index * matmul_param.K;
        matmul_param.M = output_block_oc_size;
        matmul_algo->pack_A(matmul_param, a_panel, 0_z, 0_z);
    };

    //! conv kernel
    template <typename src_ctype, typename bias_ctype, typename dst_ctype,
              typename op_ctype, typename op_dtype,
              PostprocessMode postprocess_mode>
    static void kerns(
            WorkspaceBundle bundle, WorkspaceBundle bundle_thread,
            const ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernSizeParam matmul_kernsize_param,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            fallback::ConvBiasImpl::NCBKernIndex ncb_index,
            size_t ohw_tile_size, size_t oc_tile_size) {
        auto is_xcorr = !param.filter_meta.should_flip;
        UNPACK_CONV_F32_NCB_KERN_SIZES(param);
        MEGDNN_MARK_USED_VAR(N);
        auto IH2 = IH + 2 * PH;
        auto IW2 = IW + 2 * PW;
        size_t OHW = OH * OW;
        size_t output_block_size = std::min(
                ohw_tile_size, OHW - ncb_index.ndrange_id[2] * ohw_tile_size);
        size_t output_block_oc_size = std::min(
                oc_tile_size, OC - ncb_index.ndrange_id[3] * oc_tile_size);

        //! misc flags
        bool special_1x1 = (FH == 1 && FW == 1 && SH == 1 && SW == 1 &&
                            PH == 0 && PW == 0);
        bool is_dst_8bit =
                (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                 param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                (param.src_type.enumv() == DTypeEnum::Quantized8Asymm &&
                 param.dst_type.enumv() == DTypeEnum::Quantized8Asymm);
        bool is_ohw_size_bigger = (ohw_tile_size >= OHW);
        bool skip_copy_dst = is_ohw_size_bigger && !is_dst_8bit;

        //! misc index
        size_t ohw_cur_index = ncb_index.ndrange_id[2] * ohw_tile_size;
        size_t oc_cur_index = ncb_index.ndrange_id[3] * oc_tile_size;
        size_t oc_end_index = oc_cur_index + output_block_oc_size;

        bundle.set(param.workspace_ptr);
        bundle_thread.set(PtrGetter::get_bundle_offset_byte_ptr<int8_t>(
                bundle, Im2colBundelIndex::BUNDLE_THREAD_INDEX,
                bundle_thread.total_size_in_bytes() * ncb_index.thread_id));

        fallback::MatrixMulImpl::KernParam matmul_param;
        static_cast<fallback::MatrixMulImpl::KernSizeParam&>(matmul_param) =
                matmul_kernsize_param;
        matmul_param.workspace_ptr = bundle_thread.get(
                Im2colBundelIndex::THREAD_BUNDLE_COMPUTE_INDEX);

        //! 1.Copy bias if need
        COPY_BIAS();

        //! 2.Im2col
        IM2COL();

        //! 3.packb and matmul compute
        PREPAR_MATMUL_DATA();
        MATMUL_COMPUTE();

        //! 4.postprocess and copy dst if need
        POSTPROCESS_AND_COPYDST();
#undef PREPAR_MATMUL_DATA
#undef MATMUL_COMPUTE
    }
};

#define PREPAR_MATMUL_DATA()                                                 \
    bias_ctype* matmul_dst = nullptr;                                        \
    const src_ctype* filter =                                                \
            param.filter<src_ctype>() + oc_cur_index * IC * FH * FW;         \
    matmul_dst = PtrGetter::get_matmul_dst_ptr<bias_ctype>(                  \
            param, bundle_thread,                                            \
            Im2colBundelIndex::THREAD_BUNDLE_MATMUL_DST_INDEX, oc_cur_index, \
            OHW, is_dst_8bit, is_ohw_size_bigger);

#define MATMUL_COMPUTE()                                           \
    matmul_param.M = output_block_oc_size;                         \
    matmul_param.N = output_block_size;                            \
    matmul_param.LDB = special_1x1 ? OH * OW : output_block_size;  \
    matmul_param.LDC = output_block_size;                          \
    matmul_param.A_ptr = filter;                                   \
    matmul_param.B_ptr = im2col_dst ? im2col_dst : no_padding_src; \
    matmul_param.C_ptr = matmul_dst;                               \
    auto matmul_kern_t = matmul_algo->get_kern(matmul_param);      \
    matmul_kern_t(matmul_param);

template <>
class Im2colKerns<Pack_Mode::NO_PACK> {
public:
    //! conv kernel
    template <typename src_ctype, typename bias_ctype, typename dst_ctype,
              typename op_ctype, typename op_dtype,
              PostprocessMode postprocess_mode>
    static void kerns(
            WorkspaceBundle bundle, WorkspaceBundle bundle_thread,
            const ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernSizeParam matmul_kernsize_param,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            fallback::ConvBiasImpl::NCBKernIndex ncb_index,
            size_t ohw_tile_size, size_t oc_tile_size) {
        auto is_xcorr = !param.filter_meta.should_flip;
        UNPACK_CONV_F32_NCB_KERN_SIZES(param);
        MEGDNN_MARK_USED_VAR(N);
        auto IH2 = IH + 2 * PH;
        auto IW2 = IW + 2 * PW;
        size_t OHW = OH * OW;
        size_t output_block_size = std::min(
                ohw_tile_size, OHW - ncb_index.ndrange_id[2] * ohw_tile_size);
        size_t output_block_oc_size = std::min(
                oc_tile_size, OC - ncb_index.ndrange_id[3] * oc_tile_size);
        //! misc flags
        bool special_1x1 = (FH == 1 && FW == 1 && SH == 1 && SW == 1 &&
                            PH == 0 && PW == 0);
        bool is_dst_8bit =
                (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                 param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                (param.src_type.enumv() == DTypeEnum::Quantized8Asymm &&
                 param.dst_type.enumv() == DTypeEnum::Quantized8Asymm);
        bool is_ohw_size_bigger = (ohw_tile_size >= OHW);
        bool skip_copy_dst = is_ohw_size_bigger && !is_dst_8bit;

        //! misc index
        size_t ohw_cur_index = ncb_index.ndrange_id[2] * ohw_tile_size;
        size_t oc_cur_index = ncb_index.ndrange_id[3] * oc_tile_size;
        size_t oc_end_index = oc_cur_index + output_block_oc_size;

        bundle.set(param.workspace_ptr);
        bundle_thread.set(PtrGetter::get_bundle_offset_byte_ptr<int8_t>(
                bundle, Im2colBundelIndex::BUNDLE_THREAD_INDEX,
                bundle_thread.total_size_in_bytes() * ncb_index.thread_id));

        fallback::MatrixMulImpl::KernParam matmul_param;
        static_cast<fallback::MatrixMulImpl::KernSizeParam&>(matmul_param) =
                matmul_kernsize_param;
        matmul_param.workspace_ptr = bundle_thread.get(
                Im2colBundelIndex::THREAD_BUNDLE_COMPUTE_INDEX);

        //! 1.Copy bias if need
        COPY_BIAS();

        //! 2.Im2col
        IM2COL();

        //! 3.packb and matmul compute
        PREPAR_MATMUL_DATA();
        MATMUL_COMPUTE();

        //! 4.postprocess and copy dst if need
        POSTPROCESS_AND_COPYDST();

#undef PREPAR_MATMUL_DATA
#undef MATMUL_COMPUTE
    }
};

#undef COPY_BIAS
#undef IM2COL
#undef POSTPROCESS_AND_COPYDST
fallback::MatrixMulImpl::KernSizeParam
ConvBiasImpl::AlgoIm2col ::get_matmul_kern_param(const NCBKernSizeParam& param,
                                                 size_t ohw_tile_size,
                                                 size_t oc_tile_size) const {
    size_t M = oc_tile_size;
    size_t N = ohw_tile_size;
    size_t K = param.filter_meta.icpg * param.filter_meta.spatial[0] *
               param.filter_meta.spatial[1];
    size_t LDA = K, LDB = N, LDC = N;
    bool is_dst_8bit = (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                        param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                       (param.src_type.enumv() == DTypeEnum::Quantized8Asymm &&
                        param.dst_type.enumv() == DTypeEnum::Quantized8Asymm);
    return {param.filter_type,
            param.src_type,
            is_dst_8bit ? param.bias_type : param.dst_type,
            M,
            N,
            K,
            LDA,
            LDB,
            LDC,
            false,
            false,
            param::MatrixMul::ComputeMode::DEFAULT,
            param::MatrixMul::Format::DEFAULT};
}

void ConvBiasImpl::AlgoIm2col::choice_ohw_oc_block(
        const NCBKernSizeParam& param, size_t block_m, size_t block_n,
        bool need_pack) const {
    size_t nr_threads = param.nr_threads;
    size_t OC = param.filter_meta.ocpg;
    size_t ohw = param.osz[0] * param.osz[1];
    //! pay attention please, should not change the 2 line code,
    //! the opr use the same im2col algo, via choice_ohw_oc_block may change the
    //! m_ohw_tile_size and m_oc_tile_size， if the two value changed, the
    //! workspace size may change, will ocur workspace not match problem, so
    //! should use the original data init them to avoid the problem
    m_oc_tile_size = DEFAULT_OC_TILE_SIZE;
    m_ohw_tile_size = m_ohw_tile_origin;

    m_oc_tile_size = std::min(m_oc_tile_size, OC);
    m_ohw_tile_size = std::min(m_ohw_tile_size, ohw);

    if (nr_threads > 1) {
        if (ohw / m_ohw_tile_size < nr_threads) {
            m_ohw_tile_size = round_up(div_ceil(ohw, nr_threads), block_n);
            if (m_ohw_tile_size < DEFAULT_OHW_MIN_TILE_SIZE) {
                m_ohw_tile_size = ohw;
                m_oc_tile_size = round_up(div_ceil(OC, nr_threads), block_m);
                if (m_oc_tile_size > DEFAULT_OC_MAX_TILE_SIZE) {
                    m_oc_tile_size = DEFAULT_OC_MAX_TILE_SIZE;
                } else if (m_oc_tile_size < DEFAULT_OC_MIN_TILE_SIZE) {
                    m_oc_tile_size = DEFAULT_OC_MIN_TILE_SIZE;
                }
            }
        }
    } else {
        if (!need_pack) {  //! no pack ,usually in x86 save memroy
            m_ohw_tile_size = ohw;
            m_oc_tile_size = OC;
        }
    }
}

WorkspaceBundle ConvBiasImpl::AlgoIm2col::get_bundle(
        const NCBKernSizeParam& param) const {
    UNPACK_CONV_F32_NCB_KERN_SIZES(param);
    MEGDNN_MARK_USED_VAR(OC);
    MEGDNN_MARK_USED_VAR(OH);
    MEGDNN_MARK_USED_VAR(OW);
    MEGDNN_MARK_USED_VAR(FH);
    MEGDNN_MARK_USED_VAR(FW);
    MEGDNN_MARK_USED_VAR(SW);
    MEGDNN_MARK_USED_VAR(SH);

    auto IW2 = IH + 2 * PH;
    auto IH2 = IW + 2 * PW;
    bool no_need_pading = (PH == 0 && PW == 0);
    size_t padding = 0, packa_size = 0, packa_group_size = 0;
    size_t nr_threads = param.nr_threads;
    size_t GROUP = param.filter_meta.group;
    bool need_pack = m_matmul_algo->packmode() == Pack_Mode::DEFAULT;
    bool only_packA = m_matmul_algo->packmode() == Pack_Mode::ONLY_PACKA;
    if (need_pack || only_packA) {
        auto inner_block = m_matmul_algo->get_inner_block_size();
        choice_ohw_oc_block(param, inner_block.m, inner_block.n, need_pack);
        auto im2col_kern_param = get_matmul_kern_param(
                param, m_ohw_tile_size, only_packA ? m_oc_tile_size : OC);
        size_t oc_parallel_times = div_ceil<size_t>(OC, m_oc_tile_size);
        WorkspaceBundle wb = m_matmul_algo->get_bundle(im2col_kern_param);
        packa_group_size = only_packA ? oc_parallel_times * wb.get_size(0)
                                      : wb.get_size(0);
    } else {  //! not support pack,not need pack
        size_t nopack_default_blockm = 8;
        size_t nopack_default_blockn = 16;
        choice_ohw_oc_block(param, nopack_default_blockm, nopack_default_blockn,
                            need_pack);
        packa_group_size = 0;
    }
    if (no_need_pading) {
        padding = 0;  //! not need  padding
    } else {
        padding = (GROUP * N * IC * IH2 * IW2) *
                  sizeof(param.src_type);  //! for padding
    }
    packa_size = GROUP * packa_group_size;  //! for packA  size = GROUP * a_size
    WorkspaceBundle ws = get_thread_bundle(param);
    return {nullptr,
            {padding, packa_size, ws.total_size_in_bytes() * nr_threads}};
}

WorkspaceBundle ConvBiasImpl::AlgoIm2col::get_thread_bundle(
        const NCBKernSizeParam& param) const {
    size_t IC = param.filter_meta.icpg, FH = param.filter_meta.spatial[0],
           FW = param.filter_meta.spatial[1];
    size_t ohw = param.osz[0] * param.osz[1];

    size_t im2col = 0, packb = 0, matmul_dst = 0, bias_temp = 0,
           matmul_compute = 0;
    auto im2col_kern_param =
            get_matmul_kern_param(param, m_ohw_tile_size, m_oc_tile_size);
    bool default_pack = m_matmul_algo->packmode() == Pack_Mode::DEFAULT;
    bool only_packA = m_matmul_algo->packmode() == Pack_Mode::ONLY_PACKA;
    bool is_dst_8bit = (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                        param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                       (param.src_type.enumv() == DTypeEnum::Quantized8Asymm &&
                        param.dst_type.enumv() == DTypeEnum::Quantized8Asymm);
    size_t im2col_dst_size =
            IC * FH * FW * m_ohw_tile_size * sizeof(param.src_type);
    size_t matmul_dst_size =
            m_oc_tile_size * m_ohw_tile_size * sizeof(param.bias_type);
    if (default_pack || only_packA) {
        //! matmul_dst and im2col_dst use the same memory
        WorkspaceBundle wb = m_matmul_algo->get_bundle(im2col_kern_param);
        packb = wb.get_size(1);
        im2col = only_packA ? im2col_dst_size
                            : std::max(im2col_dst_size, matmul_dst_size);
        matmul_dst = only_packA ? matmul_dst_size : 0;
    } else {
        im2col = im2col_dst_size;
        if (is_dst_8bit) {
            matmul_dst = matmul_dst_size;
        } else {
            matmul_dst = m_ohw_tile_size >= ohw ? 0 : matmul_dst_size;
        }
        matmul_compute = m_matmul_algo->get_workspace(im2col_kern_param);
    }
    if (param.bias_mode == megdnn::BiasMode::BIAS) {
        bias_temp = m_oc_tile_size * m_ohw_tile_size * sizeof(param.bias_type);
    }
    return {nullptr, {packb, im2col, matmul_dst, bias_temp, matmul_compute}};
}

size_t ConvBiasImpl::AlgoIm2col::get_workspace(
        ConvBiasImpl*, const NCBKernSizeParam& p) const {
    MIDOUT_BEGIN(megdnn_fallback_im2col, 0, 0) {
        return get_bundle(p).total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoIm2col::dispatch_kerns(
        ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_fallback_im2col, 0, 1) {
        size_t ohw = param.osz[0] * param.osz[1];
        size_t ohw_parallel_times = div_ceil(ohw, m_ohw_tile_size);
        size_t GROUP = param.filter_meta.group;
        size_t IC = param.filter_meta.icpg;
        size_t OC = param.filter_meta.ocpg;
        size_t PH = param.filter_meta.padding[0];
        size_t PW = param.filter_meta.padding[1];

        WorkspaceBundle bundle = get_bundle(param);
        WorkspaceBundle bundle_thread = get_thread_bundle(param);

        size_t oc_parallel_times = div_ceil(OC, m_oc_tile_size);
        bool need_padding = (PH != 0 || PW != 0);
        bool default_pack = m_matmul_algo->packmode() == Pack_Mode::DEFAULT;
        bool no_pack = m_matmul_algo->packmode() == Pack_Mode::NO_PACK;
        bool only_packA = m_matmul_algo->packmode() == Pack_Mode::ONLY_PACKA;
        size_t packa_parallel_times = 0;
        if (only_packA) {
            packa_parallel_times = div_ceil(OC, m_oc_tile_size);
        } else if (default_pack) {
            packa_parallel_times =
                    div_ceil(OC, m_matmul_algo->get_inner_block_size().m);
        }

        auto matmul_param = get_matmul_kern_param(
                param, m_ohw_tile_size, only_packA ? m_oc_tile_size : OC);

        SmallVector<ConvBiasImpl::NCBKern> ret_kern;

#define RETURN_KERNS()                                                      \
    if (default_pack) {                                                     \
        ret_kern.push_back(                                                 \
                {kern_default_packA, {GROUP, 1_z, packa_parallel_times}});  \
    }                                                                       \
    if (only_packA) {                                                       \
        ret_kern.push_back(                                                 \
                {kern_only_packA, {GROUP, 1_z, packa_parallel_times}});     \
    }                                                                       \
    if (need_padding) {                                                     \
        ret_kern.push_back({kern_padding, {GROUP, param.n, IC}});           \
    }                                                                       \
    if (default_pack) {                                                     \
        ret_kern.push_back(                                                 \
                {kern_compute_default,                                      \
                 {GROUP, param.n, ohw_parallel_times, oc_parallel_times}}); \
    }                                                                       \
    if (no_pack) {                                                          \
        ret_kern.push_back(                                                 \
                {kern_compute_nopack,                                       \
                 {GROUP, param.n, ohw_parallel_times, oc_parallel_times}}); \
    }                                                                       \
    if (only_packA) {                                                       \
        ret_kern.push_back(                                                 \
                {kern_compute_onlypackA,                                    \
                 {GROUP, param.n, ohw_parallel_times, oc_parallel_times}}); \
    }                                                                       \
    return ret_kern;

#define COMPUTE_KERN(_name, _pack_mode, _dt, _post_ctype, _postprocess_mode) \
    auto kern_compute_##_name = [bundle, bundle_thread, matmul_param,        \
                                 matmul_algo = m_matmul_algo,                \
                                 ohw_tile_size = m_ohw_tile_size,            \
                                 oc_tile_size = m_oc_tile_size](             \
                                        const NCBKernParam& param,           \
                                        const NCBKernIndex& ncb_index) {     \
        Im2colKerns<_pack_mode>::kerns<_dt, _dt, _dt, _post_ctype,           \
                                       _post_ctype, _postprocess_mode>(      \
                bundle, bundle_thread, param, matmul_param, matmul_algo,     \
                ncb_index, ohw_tile_size, oc_tile_size);                     \
    };

#define cb(_dt, _post_ctype, _postprocess_mode, _midout_tags)                 \
    do {                                                                      \
        if (param.filter_type.enumv() == DTypeTrait<_dt>::enumv) {            \
            MIDOUT_BEGIN(megdnn_fallback_im2col, 0, 1, _midout_tags) {        \
                auto kern_padding = [bundle](const NCBKernParam& param,       \
                                             const NCBKernIndex& ncb_index) { \
                    copy_padding_kern<_dt>(bundle, param, ncb_index);         \
                };                                                            \
                auto kern_default_packA =                                     \
                        [bundle, matmul_algo = m_matmul_algo, matmul_param](  \
                                const NCBKernParam& param,                    \
                                const NCBKernIndex& ncb_index) {              \
                            Im2colKerns<Pack_Mode::DEFAULT>::packA_kern<_dt>( \
                                    bundle, param, matmul_param, matmul_algo, \
                                    ncb_index);                               \
                        };                                                    \
                auto kern_only_packA = [bundle, matmul_algo = m_matmul_algo,  \
                                        matmul_param](                        \
                                               const NCBKernParam& param,     \
                                               const NCBKernIndex&            \
                                                       ncb_index) {           \
                    Im2colKerns<Pack_Mode::ONLY_PACKA>::packA_kern<_dt>(      \
                            bundle, param, matmul_param, matmul_algo,         \
                            ncb_index);                                       \
                };                                                            \
                COMPUTE_KERN(default, Pack_Mode::DEFAULT, _dt, _post_ctype,   \
                             _postprocess_mode);                              \
                COMPUTE_KERN(nopack, Pack_Mode::NO_PACK, _dt, _post_ctype,    \
                             _postprocess_mode);                              \
                COMPUTE_KERN(onlypackA, Pack_Mode::ONLY_PACKA, _dt,           \
                             _post_ctype, _postprocess_mode);                 \
                RETURN_KERNS();                                               \
            }                                                                 \
            MIDOUT_END();                                                     \
            return {};                                                        \
        }                                                                     \
    } while (0);

        cb(dt_float32, dt_float32, PostprocessMode::FLOAT, 0);
#if !MEGDNN_DISABLE_FLOAT16
        cb(dt_float16, dt_float16, PostprocessMode::NO_PROCESS, 2);
#endif
#undef cb
#undef COMPUTE_KERN

#define COMPUTE_KERN(_name, _pack_mode, _src_ctype, _bias_ctype, _dst_ctype, \
                     _i_bias_type, _i_dst_type, _postprocess_mode)           \
    auto kern_compute_##_name = [bundle, bundle_thread, matmul_param,        \
                                 matmul_algo = m_matmul_algo,                \
                                 ohw_tile_size = m_ohw_tile_size,            \
                                 oc_tile_size = m_oc_tile_size](             \
                                        const NCBKernParam& param,           \
                                        const NCBKernIndex& ncb_index) {     \
        Im2colKerns<_pack_mode>::kerns<_src_ctype, _bias_ctype, _dst_ctype,  \
                                       DTypeTrait<_i_bias_type>::ctype,      \
                                       DTypeTrait<_i_dst_type>::ctype,       \
                                       _postprocess_mode>(                   \
                bundle, bundle_thread, param, matmul_param, matmul_algo,     \
                ncb_index, ohw_tile_size, oc_tile_size);                     \
    };

#define cb(_i_src_type, _i_bias_type, _i_dst_type, _src_ctype, _bias_ctype,   \
           _dst_ctype, _postprocess_mode, _midout_tags)                       \
    do {                                                                      \
        if (param.filter_type.enumv() == param.src_type.enumv() &&            \
            param.src_type.enumv() == DTypeTrait<_i_src_type>::enumv &&       \
            param.dst_type.enumv() == DTypeTrait<_i_dst_type>::enumv) {       \
            MIDOUT_BEGIN(megdnn_fallback_im2col, 0, 1, _midout_tags) {        \
                auto kern_padding = [bundle](const NCBKernParam& param,       \
                                             const NCBKernIndex& ncb_index) { \
                    copy_padding_kern<_src_ctype>(bundle, param, ncb_index);  \
                };                                                            \
                auto kern_default_packA = [bundle,                            \
                                           matmul_algo = m_matmul_algo,       \
                                           matmul_param](                     \
                                                  const NCBKernParam& param,  \
                                                  const NCBKernIndex&         \
                                                          ncb_index) {        \
                    Im2colKerns<Pack_Mode::DEFAULT>::packA_kern<_src_ctype>(  \
                            bundle, param, matmul_param, matmul_algo,         \
                            ncb_index);                                       \
                };                                                            \
                auto kern_only_packA =                                        \
                        [bundle, matmul_algo = m_matmul_algo, matmul_param](  \
                                const NCBKernParam& param,                    \
                                const NCBKernIndex& ncb_index) {              \
                            Im2colKerns<Pack_Mode::ONLY_PACKA>::packA_kern<   \
                                    _src_ctype>(bundle, param, matmul_param,  \
                                                matmul_algo, ncb_index);      \
                        };                                                    \
                COMPUTE_KERN(default, Pack_Mode::DEFAULT, _src_ctype,         \
                             _bias_ctype, _dst_ctype, _i_bias_type,           \
                             _i_dst_type, _postprocess_mode);                 \
                COMPUTE_KERN(nopack, Pack_Mode::NO_PACK, _src_ctype,          \
                             _bias_ctype, _dst_ctype, _i_bias_type,           \
                             _i_dst_type, _postprocess_mode);                 \
                COMPUTE_KERN(onlypackA, Pack_Mode::ONLY_PACKA, _src_ctype,    \
                             _bias_ctype, _dst_ctype, _i_bias_type,           \
                             _i_dst_type, _postprocess_mode);                 \
                RETURN_KERNS();                                               \
            }                                                                 \
            MIDOUT_END();                                                     \
            return {};                                                        \
        }                                                                     \
    } while (0);

        cb(dt_int8, dt_int32, dt_int32, dt_int8, dt_int32, dt_int32,
           PostprocessMode::NO_PROCESS, 3);

        cb(dt_int8, dt_int16, dt_int16, dt_int8, dt_int16, dt_int16,
           PostprocessMode::NO_PROCESS, 4);

        cb(dtype::QuantizedS8, dtype::QuantizedS32, dtype::QuantizedS32,
           dt_int8, dt_int32, dt_int32, PostprocessMode::NO_PROCESS, 7);

        cb(dtype::QuantizedS8, dtype::QuantizedS32, dtype::QuantizedS8, dt_int8,
           dt_int32, dt_int8, PostprocessMode::QUANTIZED, 8);
#undef COMPUTE_KERN
#undef RETURN_KERNS
#undef cb
        megdnn_throw("unsupported data type on im2col matmul algo");
    }
    MIDOUT_END();
    return {};
}

bool ConvBiasImpl::AlgoIm2col::usable(
        ConvBiasImpl* opr, const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MIDOUT_BEGIN(megdnn_fallback_im2col, 0, 2) {
        //! make sure 8x8x16 and 8x8x32 biasmode is  nobias and nonlineMode is
        //! identity otherwise return false mean that 8x8x32 and 8x8x16 not support
        //! PostProcess
        if (param.src_type.enumv() == param.filter_type.enumv() &&
            ((param.src_type.enumv() == DTypeEnum::Int8 &&
              (param.dst_type.enumv() == DTypeEnum::Int16 ||
               param.dst_type.enumv() == DTypeEnum::Int32)) ||
             ((param.src_type.enumv() == DTypeEnum::QuantizedS8 ||
               param.src_type.enumv() == DTypeEnum::Quantized8Asymm) &&
              param.dst_type.enumv() == DTypeEnum::QuantizedS32)) &&
            param.bias_mode != megdnn::BiasMode::NO_BIAS &&
            param.nonlineMode != megdnn::NonlineMode::IDENTITY) {
            return false;
        }
        fallback::MatrixMulImpl::KernSizeParam matmul_param =
                get_matmul_kern_param(param, m_ohw_tile_size, m_oc_tile_size);
        bool matmulusable = m_matmul_algo->usable(matmul_param);
        return matmulusable &&
               (opr->param().format == param::ConvBias::Format::NCHW) &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                (param.filter_meta.spatial[0] <= 7)) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT;
    }
    MIDOUT_END();
    return false;
}

// vim: syntax=cpp.doxygen
