/**
 * \file dnn/src/fallback/conv_bias/im2col/strategy_base.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "src/fallback/conv_bias/opr_impl.h"

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

using PackMode = fallback::MatrixMulImpl::AlgoBase::PackMode;
using FormatMode = param::ConvBias::Format;

struct StrategyParam {
    size_t batch_id;
    size_t group_id;
    size_t oc_tile_size;
    size_t oc_cur_index;
    size_t oc_end_index;
    size_t ohw_cur_index;
    size_t output_block_size;
    size_t output_block_oc_size;
    size_t ohw;
    size_t block_m;
    size_t block_n;
    size_t block_k;
    size_t pack_oc_size;
    size_t packA_group_size;
    bool skip_copy_dst;
    bool is_dst_8bit;
    bool is_ohw_size_bigger;
    bool enable_filter_preprocess;
};

class StrategyBase {
public:
    StrategyBase() = default;
    virtual ~StrategyBase() = default;
    virtual void copy_padding_kern(
            const WorkspaceBundle& bundle,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
            size_t pack_size) = 0;
    virtual void packA_kern(
            const WorkspaceBundle& bundle,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernSizeParam matmulparam,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
            const fallback::MatrixMulImpl::AlgoBase::MatmulDescription&
                    matmul_desec,
            const StrategyParam& sparam) = 0;

    virtual void exec_im2col(
            const WorkspaceBundle& bundle, const WorkspaceBundle& bundle_thread,
            const StrategyParam& sparam,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernParam matmul_param,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo) = 0;

    virtual void exec_matmul(
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const StrategyParam& sparam, const WorkspaceBundle& bundle,
            const WorkspaceBundle& bundle_thread,
            fallback::MatrixMulImpl::KernParam matmul_param,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
            const fallback::MatrixMulImpl::AlgoBase::MatmulDescription&
                    matmul_desc) = 0;

    virtual void exec_postprocess(
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const StrategyParam& sparam,
            const WorkspaceBundle& bundle_thread) = 0;
};

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode, PackMode packmode,
          FormatMode format>
//! this class is a new base class for StrategyDefault StrategyNoPack and so on,
//! in order to handle copy pad use the same code
class StrategyBridge : public StrategyBase {
public:
    constexpr static size_t BUNDLE_PADDING_INDEX = 0;

    StrategyBridge() = default;

    virtual void copy_padding_kern(
            const WorkspaceBundle& bundle,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
            size_t pack_oc_size) override {
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
};

namespace{
template <typename bias_ctype>
inline void* get_matmul_dst_ptr(const fallback::ConvBiasImpl::NCBKernParam& param,
                           const WorkspaceBundle& bundle_thread,
                           const StrategyParam& sparam,
                           size_t matmul_bundle_index) {
    if (sparam.is_dst_8bit || !sparam.is_ohw_size_bigger) {
        return static_cast<void*>(bundle_thread.get(matmul_bundle_index));
    } else {
        bias_ctype* dst =
                param.dst<bias_ctype>(sparam.batch_id, sparam.group_id) +
                sparam.oc_cur_index * sparam.ohw;
        return static_cast<void*>(dst);
    }
}

template <typename bias_ctype>
inline void* get_bias_temp_ptr(
        const fallback::ConvBiasImpl::NCBKernParam& param,
        const WorkspaceBundle& bundle_thread, size_t bias_bundle_index) {
    bias_ctype* bias_tmp_ptr =
            param.bias_mode == megdnn::BiasMode::BIAS
                    ? static_cast<bias_ctype*>(
                              bundle_thread.get(bias_bundle_index))
                    : nullptr;
    return bias_tmp_ptr;
}

template <typename dst_ctype>
void copy_dst(const fallback::ConvBiasImpl::NCBKernParam& param,
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

template <typename bias_ctype>
void copy_bias(const fallback::ConvBiasImpl::NCBKernParam& param,
               const WorkspaceBundle& bundle_thread,
               const StrategyParam& sparam, size_t bias_index) {
    const bias_ctype* bias_ptr = static_cast<const bias_ctype*>(
            param.bias<bias_ctype>(sparam.batch_id, sparam.group_id));
    bias_ctype* bias_temp_ptr = static_cast<bias_ctype*>(
            get_bias_temp_ptr<bias_ctype>(param, bundle_thread, bias_index));
    if (param.bias_mode == megdnn::BiasMode::BIAS) {
        bias_ctype* copy_dst = bias_temp_ptr;
        size_t pack_oc_size = sparam.pack_oc_size;
        const bias_ctype* copy_src = bias_ptr +
                                     sparam.oc_cur_index * sparam.ohw +
                                     sparam.ohw_cur_index * pack_oc_size;
        for (size_t oc = sparam.oc_cur_index / pack_oc_size;
             oc < sparam.oc_end_index / pack_oc_size; oc++) {
            std::memcpy(copy_dst, copy_src,
                        sizeof(bias_ctype) * sparam.output_block_size *
                                pack_oc_size);
            copy_dst += sparam.output_block_size * pack_oc_size;
            copy_src += sparam.ohw * pack_oc_size;
        }
    }
}

template <typename bias_ctype, typename dst_ctype, typename op_ctype,
          typename op_dtype, megdnn::PostprocessMode postprocess_mode>
void do_postprocess(const fallback::ConvBiasImpl::NCBKernParam& param,
                    const StrategyParam& sparam,
                    const WorkspaceBundle& bundle_thread,
                    size_t matmul_bundle_index, size_t bias_bundle_index) {
    copy_bias<bias_ctype>(param, bundle_thread, sparam, bias_bundle_index);
    void* matmul_dst = get_matmul_dst_ptr<bias_ctype>(
            param, bundle_thread, sparam, matmul_bundle_index);

    const bias_ctype* bias_ptr = static_cast<const bias_ctype*>(
            param.bias<bias_ctype>(sparam.batch_id, sparam.group_id));
    void* bias_temp_ptr = get_bias_temp_ptr<bias_ctype>(param, bundle_thread,
                                                        bias_bundle_index);
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
    copy_dst<dst_ctype>(param, matmul_dst, sparam);
}
}

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode, PackMode packmode,
          FormatMode format = FormatMode::NCHW>
class Strategy;

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
class Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
               postprocess_mode, PackMode::DEFAULT>
        : public StrategyBridge<src_ctype, bias_ctype, dst_ctype, op_ctype,
                                op_dtype, postprocess_mode, PackMode::DEFAULT,
                                FormatMode::NCHW> {
public:
    constexpr static size_t BUNDLE_PADDING_INDEX = 0;
    constexpr static size_t BUNDLE_PACKA_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_PACKB_INDEX = 0;
    constexpr static size_t THREAD_BUNDLE_IM2COL_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_BIAS_INDEX = 2;

    Strategy() = default;

    virtual void packA_kern(
            const WorkspaceBundle& bundle,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernSizeParam matmulparam,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
            const fallback::MatrixMulImpl::AlgoBase::MatmulDescription&
                    matmul_desc,
            const StrategyParam& sparam) override;
    virtual void exec_im2col(
            const WorkspaceBundle& bundle, const WorkspaceBundle& bundle_thread,
            const StrategyParam& sparam,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernParam matmul_param,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo) override;

    void exec_matmul(const fallback::ConvBiasImpl::NCBKernParam& param,
                     const StrategyParam& sparam, const WorkspaceBundle& bundle,
                     const WorkspaceBundle& bundle_thread,
                     fallback::MatrixMulImpl::KernParam matmul_param,
                     const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                     const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
                     const fallback::MatrixMulImpl::AlgoBase::MatmulDescription&
                             matmul_desc) override;
    void exec_postprocess(const fallback::ConvBiasImpl::NCBKernParam& param,
                          const StrategyParam& sparam,
                          const WorkspaceBundle& bundle_thread) override {
        do_postprocess<bias_ctype, dst_ctype, op_ctype, op_dtype,
                       postprocess_mode>(param, sparam, bundle_thread,
                                         THREAD_BUNDLE_IM2COL_INDEX,
                                         THREAD_BUNDLE_BIAS_INDEX);
    }

    void* get_matmul_dst_ptr(const fallback::ConvBiasImpl::NCBKernParam& param,
                             const WorkspaceBundle& bundle_thread,
                             const StrategyParam& sparam);
};

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
class Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
               postprocess_mode, PackMode::DEFAULT, FormatMode::NCHW44>
        : public Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
                          postprocess_mode, PackMode::DEFAULT> {
public:
    const size_t BUNDLE_PADDING_INDEX = 0;
    const size_t BUNDLE_PACKA_INDEX = 1;
    const size_t THREAD_BUNDLE_PACKB_INDEX = 0;
    const size_t THREAD_BUNDLE_IM2COL_INDEX = 1;
    const size_t THREAD_BUNDLE_BIAS_INDEX = 2;

    Strategy() = default;

    void exec_im2col(
            const WorkspaceBundle& bundle, const WorkspaceBundle& bundle_thread,
            const StrategyParam& sparam,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernParam matmul_param,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo) override;
};

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
class Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
               postprocess_mode, PackMode::NO_PACK>
        : public StrategyBridge<src_ctype, bias_ctype, dst_ctype, op_ctype,
                                op_dtype, postprocess_mode, PackMode::NO_PACK,
                                FormatMode::NCHW> {
public:
    constexpr static size_t BUNDLE_PADDING_INDEX = 0;
    constexpr static size_t BUNDLE_PACKA_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_IM2COL_INDEX = 0;
    constexpr static size_t THREAD_BUNDLE_MATMULDST_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_BIAS_INDEX = 2;
    constexpr static size_t THREAD_BUNDLE_MATCOMP_INDEX = 3;

    Strategy() = default;

    void packA_kern(
            const WorkspaceBundle& bundle,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernSizeParam matmulparam,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
            const fallback::MatrixMulImpl::AlgoBase::MatmulDescription& MDsec,
            const StrategyParam& sparam) override;

    void exec_matmul(const fallback::ConvBiasImpl::NCBKernParam& param,
                     const StrategyParam& sparam, const WorkspaceBundle& bundle,
                     const WorkspaceBundle& bundle_thread,
                     fallback::MatrixMulImpl::KernParam matmul_param,
                     const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                     const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
                     const fallback::MatrixMulImpl::AlgoBase::MatmulDescription&
                             matmul_desc) override;

    void* get_matmul_dst_ptr(const fallback::ConvBiasImpl::NCBKernParam& param,
                             const WorkspaceBundle& bundle_thread,
                             const StrategyParam& sparam);

    void exec_im2col(
            const WorkspaceBundle& bundle, const WorkspaceBundle& bundle_thread,
            const StrategyParam& sparam,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernParam matmul_param,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo) override;
    void exec_postprocess(const fallback::ConvBiasImpl::NCBKernParam& param,
                          const StrategyParam& sparam,
                          const WorkspaceBundle& bundle_thread) override {
        do_postprocess<bias_ctype, dst_ctype, op_ctype, op_dtype,
                       postprocess_mode>(param, sparam, bundle_thread,
                                         THREAD_BUNDLE_MATMULDST_INDEX,
                                         THREAD_BUNDLE_BIAS_INDEX);
    }
};

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
class Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
               postprocess_mode, PackMode::ONLY_PACKA>
        : public StrategyBridge<src_ctype, bias_ctype, dst_ctype, op_ctype,
                                op_dtype, postprocess_mode,
                                PackMode::ONLY_PACKA,FormatMode::NCHW> {
public:
    constexpr static size_t BUNDLE_PADDING_INDEX = 0;
    constexpr static size_t BUNDLE_PACKA_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_PACKB_INDEX = 0;
    constexpr static size_t THREAD_BUNDLE_IM2COL_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_MATMULDST_INDEX = 2;
    constexpr static size_t THREAD_BUNDLE_BIAS_INDEX = 3;

    Strategy() = default;

    void packA_kern(
            const WorkspaceBundle& bundle,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernSizeParam matmulparam,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
            const fallback::MatrixMulImpl::AlgoBase::MatmulDescription& MDsec,
            const StrategyParam& sparam) override;

    void exec_im2col(
            const WorkspaceBundle& bundle, const WorkspaceBundle& bundle_thread,
            const StrategyParam& sparam,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernParam matmul_param,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo) override;

    void exec_matmul(const fallback::ConvBiasImpl::NCBKernParam& param,
                     const StrategyParam& sparam, const WorkspaceBundle& bundle,
                     const WorkspaceBundle& bundle_thread,
                     fallback::MatrixMulImpl::KernParam matmul_param,
                     const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                     const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
                     const fallback::MatrixMulImpl::AlgoBase::MatmulDescription&
                             matmul_desc) override;

    void* get_matmul_dst_ptr(const fallback::ConvBiasImpl::NCBKernParam& param,
                             const WorkspaceBundle& bundle_thread,
                             const StrategyParam& sparam);

    void exec_postprocess(const fallback::ConvBiasImpl::NCBKernParam& param,
                          const StrategyParam& sparam,
                          const WorkspaceBundle& bundle_thread) override {
        do_postprocess<bias_ctype, dst_ctype, op_ctype, op_dtype,
                       postprocess_mode>(param, sparam, bundle_thread,
                                         THREAD_BUNDLE_MATMULDST_INDEX,
                                         THREAD_BUNDLE_BIAS_INDEX);
    }
};
#if MEGDNN_AARCH64
template <typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
class StrategyFuse4x4x16Nchw44
        : public Strategy<dt_int8, dt_int32, dt_int8, op_ctype, op_dtype,
                          postprocess_mode, PackMode::DEFAULT,
                          FormatMode::NCHW44> {
public:
    StrategyFuse4x4x16Nchw44() = default;

    constexpr static size_t BUNDLE_PADDING_INDEX = 0;
    constexpr static size_t BUNDLE_PACKA_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_PACKB_INDEX = 0;
    constexpr static size_t THREAD_BUNDLE_IM2COL_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_BIAS_INDEX = 2;

    void exec_im2col(
            const WorkspaceBundle& bundle, const WorkspaceBundle& bundle_thread,
            const StrategyParam& sparam,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernParam matmul_param,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo) override;
};

template <typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
class StrategyFuse8x12x4Nchw44Dot
        : public Strategy<dt_int8, dt_int32, dt_int8, op_ctype, op_dtype,
                          postprocess_mode, PackMode::DEFAULT,
                          FormatMode::NCHW44> {
public:
    StrategyFuse8x12x4Nchw44Dot() = default;

    constexpr static size_t BUNDLE_PADDING_INDEX = 0;
    constexpr static size_t BUNDLE_PACKA_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_PACKB_INDEX = 0;
    constexpr static size_t THREAD_BUNDLE_IM2COL_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_BIAS_INDEX = 2;

    void exec_im2col(
            const WorkspaceBundle& bundle, const WorkspaceBundle& bundle_thread,
            const StrategyParam& sparam,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernParam matmul_param,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo) override;
};
#else
template <typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
class StrategyFuse8x4x4Nchw44DotK3x3S2
        : public Strategy<dt_int8, dt_int32, dt_int8, op_ctype, op_dtype,
                          postprocess_mode, PackMode::DEFAULT,
                          FormatMode::NCHW44> {
public:
    StrategyFuse8x4x4Nchw44DotK3x3S2() = default;

    constexpr static size_t BUNDLE_PADDING_INDEX = 0;
    constexpr static size_t BUNDLE_PACKA_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_PACKB_INDEX = 0;
    constexpr static size_t THREAD_BUNDLE_IM2COL_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_BIAS_INDEX = 2;

    void exec_im2col(
            const WorkspaceBundle& bundle, const WorkspaceBundle& bundle_thread,
            const StrategyParam& sparam,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernParam matmul_param,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo) override;
};
#endif

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
template <typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
class StrategyFuseXx12x1Nchw44K3x3S2
        : public Strategy<float, float, float, op_ctype, op_dtype,
                          postprocess_mode, PackMode::DEFAULT,
                          FormatMode::NCHW44> {
public:
    StrategyFuseXx12x1Nchw44K3x3S2() = default;

    constexpr static size_t BUNDLE_PADDING_INDEX = 0;
    constexpr static size_t BUNDLE_PACKA_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_PACKB_INDEX = 0;
    constexpr static size_t THREAD_BUNDLE_IM2COL_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_BIAS_INDEX = 2;

    void exec_im2col(
            const WorkspaceBundle& bundle, const WorkspaceBundle& bundle_thread,
            const StrategyParam& sparam,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernParam matmul_param,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo) override;
};
#endif
}  // namespace megdnn

// vim: syntax=cpp.doxygen
