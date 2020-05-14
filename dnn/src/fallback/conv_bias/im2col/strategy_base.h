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
    bool skip_copy_dst;
    bool is_dst_8bit;
    bool is_ohw_size_bigger;
};

class StrategyBase {
public:
    StrategyBase() = default;
    virtual ~StrategyBase() = default;
    virtual void copy_padding_kern(
            WorkspaceBundle bundle,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
            size_t pack_size) = 0;
    virtual void packA_kern(
            WorkspaceBundle bundle,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernSizeParam matmulparam,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
            size_t pack_size) = 0;

    virtual void exec_im2col(
            WorkspaceBundle bundle, WorkspaceBundle bundle_thread,
            const StrategyParam& sparam,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernParam matmul_param,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo) = 0;

    virtual void exec_matmul(
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const StrategyParam& sparam, WorkspaceBundle bundle,
            WorkspaceBundle bundle_thread,
            fallback::MatrixMulImpl::KernParam matmul_param,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index) = 0;

    virtual void exec_postprocess(
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const StrategyParam& sparam, WorkspaceBundle bundle_thread) = 0;
};

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode, PackMode packmode,
          FormatMode format = FormatMode::NCHW>
class Strategy;

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
class Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
               postprocess_mode, PackMode::DEFAULT> : public StrategyBase {
public:
    constexpr static size_t BUNDLE_PADDING_INDEX = 0;
    constexpr static size_t BUNDLE_PACKA_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_PACKB_INDEX = 0;
    constexpr static size_t THREAD_BUNDLE_IM2COL_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_BIAS_INDEX = 2;

    Strategy() = default;

    void copy_padding_kern(
            WorkspaceBundle bundle,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
            size_t pack_size) override;

    void packA_kern(WorkspaceBundle bundle,
                    const fallback::ConvBiasImpl::NCBKernParam& param,
                    fallback::MatrixMulImpl::KernSizeParam matmulparam,
                    fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                    const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
                    size_t pack_size) override;
    virtual void exec_im2col(
            WorkspaceBundle bundle, WorkspaceBundle bundle_thread,
            const StrategyParam& sparam,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernParam matmul_param,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo) override;

    void exec_matmul(
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const StrategyParam& sparam, WorkspaceBundle bundle,
            WorkspaceBundle bundle_thread,
            fallback::MatrixMulImpl::KernParam matmul_param,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index) override;
    void exec_postprocess(const fallback::ConvBiasImpl::NCBKernParam& param,
                          const StrategyParam& sparam,
                          WorkspaceBundle bundle_thread) override;

    void copy_dst(const fallback::ConvBiasImpl::NCBKernParam& param,
                  const void* matmul_dst, const StrategyParam& sparam);

    void copy_bias(const fallback::ConvBiasImpl::NCBKernParam& param,
                   WorkspaceBundle bundle_thread, const StrategyParam& sparam);

    void* get_bias_temp_ptr(const fallback::ConvBiasImpl::NCBKernParam& param,
                            const WorkspaceBundle& bundle_thread);
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

    void exec_im2col(WorkspaceBundle bundle, WorkspaceBundle bundle_thread,
                     const StrategyParam& sparam,
                     const fallback::ConvBiasImpl::NCBKernParam& param,
                     fallback::MatrixMulImpl::KernParam matmul_param,
                     fallback::MatrixMulImpl::AlgoBase* matmul_algo) override;
};

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
class Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
               postprocess_mode, PackMode::NO_PACK> : public StrategyBase {
public:
    constexpr static size_t BUNDLE_PADDING_INDEX = 0;
    constexpr static size_t BUNDLE_PACKA_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_IM2COL_INDEX = 0;
    constexpr static size_t THREAD_BUNDLE_MATMULDST_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_BIAS_INDEX = 2;
    constexpr static size_t THREAD_BUNDLE_MATCOMP_INDEX = 3;

    Strategy() = default;

    void copy_padding_kern(
            WorkspaceBundle bundle,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
            size_t pack_size) override;

    void packA_kern(WorkspaceBundle bundle,
                    const fallback::ConvBiasImpl::NCBKernParam& param,
                    fallback::MatrixMulImpl::KernSizeParam matmulparam,
                    fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                    const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
                    size_t pack_size) override;

    void exec_matmul(
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const StrategyParam& sparam, WorkspaceBundle bundle,
            WorkspaceBundle bundle_thread,
            fallback::MatrixMulImpl::KernParam matmul_param,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index) override;

    void* get_matmul_dst_ptr(const fallback::ConvBiasImpl::NCBKernParam& param,
                             const WorkspaceBundle& bundle_thread,
                             const StrategyParam& sparam);

    inline void* get_bias_temp_ptr(
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const WorkspaceBundle& bundle_thread) {
        bias_ctype* bias_tmp_ptr =
                param.bias_mode == megdnn::BiasMode::BIAS
                        ? static_cast<bias_ctype*>(
                                  bundle_thread.get(THREAD_BUNDLE_BIAS_INDEX))
                        : nullptr;
        return bias_tmp_ptr;
    }

    void exec_im2col(WorkspaceBundle bundle, WorkspaceBundle bundle_thread,
                     const StrategyParam& sparam,
                     const fallback::ConvBiasImpl::NCBKernParam& param,
                     fallback::MatrixMulImpl::KernParam matmul_param,
                     fallback::MatrixMulImpl::AlgoBase* matmul_algo) override;
    void exec_postprocess(const fallback::ConvBiasImpl::NCBKernParam& param,
                          const StrategyParam& sparam,
                          WorkspaceBundle bundle_thread) override;
    void copy_dst(const fallback::ConvBiasImpl::NCBKernParam& param,
                  const void* matmul_dst, const StrategyParam& sparam);

    void copy_bias(const fallback::ConvBiasImpl::NCBKernParam& param,
                   WorkspaceBundle bundle_thread, const StrategyParam& sparam);
};

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
class Strategy<src_ctype, bias_ctype, dst_ctype, op_ctype, op_dtype,
               postprocess_mode, PackMode::ONLY_PACKA> : public StrategyBase {
public:
    constexpr static size_t BUNDLE_PADDING_INDEX = 0;
    constexpr static size_t BUNDLE_PACKA_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_PACKB_INDEX = 0;
    constexpr static size_t THREAD_BUNDLE_IM2COL_INDEX = 1;
    constexpr static size_t THREAD_BUNDLE_MATMULDST_INDEX = 2;
    constexpr static size_t THREAD_BUNDLE_BIAS_INDEX = 3;

    Strategy() = default;

    void copy_padding_kern(
            WorkspaceBundle bundle,
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
            size_t pack_size) override;

    void packA_kern(WorkspaceBundle bundle,
                    const fallback::ConvBiasImpl::NCBKernParam& param,
                    fallback::MatrixMulImpl::KernSizeParam matmulparam,
                    fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                    const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
                    size_t pack_size) override;

    void exec_im2col(WorkspaceBundle bundle, WorkspaceBundle bundle_thread,
                     const StrategyParam& sparam,
                     const fallback::ConvBiasImpl::NCBKernParam& param,
                     fallback::MatrixMulImpl::KernParam matmul_param,
                     fallback::MatrixMulImpl::AlgoBase* matmul_algo) override;

    void exec_matmul(
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const StrategyParam& sparam, WorkspaceBundle bundle,
            WorkspaceBundle bundle_thread,
            fallback::MatrixMulImpl::KernParam matmul_param,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index) override;

    void* get_matmul_dst_ptr(const fallback::ConvBiasImpl::NCBKernParam& param,
                             const WorkspaceBundle& bundle_thread,
                             const StrategyParam& sparam);
    inline void* get_bias_temp_ptr(
            const fallback::ConvBiasImpl::NCBKernParam& param,
            const WorkspaceBundle& bundle_thread) {
        bias_ctype* bias_tmp_ptr =
                param.bias_mode == megdnn::BiasMode::BIAS
                        ? static_cast<bias_ctype*>(
                                  bundle_thread.get(THREAD_BUNDLE_BIAS_INDEX))
                        : nullptr;
        return bias_tmp_ptr;
    }
    void exec_postprocess(const fallback::ConvBiasImpl::NCBKernParam& param,
                          const StrategyParam& sparam,
                          WorkspaceBundle bundle_thread) override;
    void copy_dst(const fallback::ConvBiasImpl::NCBKernParam& param,
                  const void* matmul_dst, const StrategyParam& sparam);

    void copy_bias(const fallback::ConvBiasImpl::NCBKernParam& param,
                   WorkspaceBundle bundle_thread, const StrategyParam& sparam);
};
}  // namespace megdnn

// vim: syntax=cpp.doxygen
