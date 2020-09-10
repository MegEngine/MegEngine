/**
 * \file dnn/src/fallback/conv_bias/conv1x1/conv1x1_strategy.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "src/fallback/conv_bias/opr_impl.h"
#include "src/fallback/conv_bias/conv1x1/conv1x1_utils.h"

#if MEGDNN_X86
#include "src/x86/conv_bias/postprocess_helper.h"
#elif (MEGDNN_ARMV7 || MEGDNN_AARCH64)
#include "src/arm_common/conv_bias/postprocess_helper.h"
#else
#include "src/common/postprocess_helper.h"
#endif

namespace megdnn {
namespace fallback {
namespace conv1x1 {

#if MEGDNN_X86
using namespace x86;
#endif

class Conv1x1StrategyBase {
public:
    virtual void packA(WorkspaceBundle& whole_bundle,
                       WorkspaceBundle& matmul_bundle,
                       size_t oc_tile_size,
                       const MatrixMulImpl::AlgoBase* matmul_algo,
                       const ConvBiasImpl::NCBKernSizeParam& param,
                       const ConvBiasImpl::NCBKernParam& ncb_param,
                       const ConvBiasImpl::NCBKernIndex& ncb_index) = 0;

    virtual void packB(WorkspaceBundle& whole_bundle,
                       WorkspaceBundle& matmul_bundle,
                       const MatrixMulImpl::AlgoBase* matmul_algo,
                       const ConvBiasImpl::NCBKernSizeParam& param,
                       const ConvBiasImpl::NCBKernParam& ncb_param,
                       const ConvBiasImpl::NCBKernIndex& ncb_index) = 0;

    virtual void exec(WorkspaceBundle& whole_bundle,
                      WorkspaceBundle& matmul_bundle,
                      WorkspaceBundle& thread_bundle,
                      size_t oc_tile_size,
                      const MatrixMulImpl::AlgoBase* matmul_algo,
                      const ConvBiasImpl::NCBKernSizeParam& param,
                      const ConvBiasImpl::NCBKernParam& ncb_param,
                      const ConvBiasImpl::NCBKernIndex& ncb_index) = 0;
    virtual ~Conv1x1StrategyBase() = default;
};

template <typename src_ctype, typename bias_ctype, typename dst_ctype,
          typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode,
          MatrixMulImpl::AlgoBase::PackMode pack_mode>
class Conv1x1Strategy : public Conv1x1StrategyBase {
public:
    explicit Conv1x1Strategy(size_t pack_size = 1) : m_pack_size(pack_size) {}

    void packA(WorkspaceBundle& whole_bundle,
               WorkspaceBundle& matmul_bundle,
               size_t oc_tile_size,
               const MatrixMulImpl::AlgoBase* matmul_algo,
               const ConvBiasImpl::NCBKernSizeParam& param,
               const ConvBiasImpl::NCBKernParam& ncb_param,
               const ConvBiasImpl::NCBKernIndex& ncb_index) override {

        if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::NO_PACK) {
            megdnn_log_error("NoPack mode has no packA kernel");
            return;
        }

        whole_bundle.set(ncb_param.workspace_ptr);

        //! packa size per group
        size_t OC = param.filter_meta.ocpg;
        size_t oc_tiles_per_group = div_ceil(OC, oc_tile_size);
        size_t packa_bytes_per_oc_tile = matmul_bundle.get_size(0);
        size_t packa_bytes_per_group =
                oc_tiles_per_group * packa_bytes_per_oc_tile;

        size_t group_id = ncb_index.ndrange_id[0];
        size_t oc_tile_id_in_group = ncb_index.ndrange_id[1];

        size_t oc_start = oc_tile_id_in_group * oc_tile_size;
        size_t oc_end = oc_start + oc_tile_size;
        oc_end = (oc_end <= OC ? oc_end : OC);

        size_t OH = param.osz[0];
        size_t OW = param.osz[1];
        size_t IC = param.filter_meta.icpg;
        MatrixMulImpl::KernParam matmul_kern_param;
        static_cast<MatrixMulImpl::KernSizeParam&>(matmul_kern_param) =
                utils::get_matmul_kern_param(param, OH * OW, oc_end - oc_start);

        size_t bytes_offset_of_a_panel =
                group_id * packa_bytes_per_group +
                oc_tile_id_in_group * packa_bytes_per_oc_tile;
        size_t numbers_offset_of_filter =
                oc_tile_size * IC * oc_tile_id_in_group;

        int8_t* tmp_ptr =
                is_enable_filter_preprocess(param)
                        ? static_cast<int8_t*>(
                                  param.preprocessed_filter->tensors[0].raw_ptr)
                        : static_cast<int8_t*>(whole_bundle.get(0));

        src_ctype* a_panel =
                reinterpret_cast<src_ctype*>(tmp_ptr + bytes_offset_of_a_panel);

        matmul_kern_param.A_ptr = const_cast<src_ctype*>(
                ncb_param.filter<src_ctype>(group_id) +
                numbers_offset_of_filter);
        matmul_algo->pack_A(matmul_kern_param, a_panel, 0,
                            oc_end - oc_start);
    }

    void packB(WorkspaceBundle& whole_bundle,
               WorkspaceBundle& matmul_bundle,
               const MatrixMulImpl::AlgoBase* matmul_algo,
               const ConvBiasImpl::NCBKernSizeParam& param,
               const ConvBiasImpl::NCBKernParam& ncb_param,
               const ConvBiasImpl::NCBKernIndex& ncb_index) override {
        MEGDNN_MARK_USED_VAR(ncb_index);
        if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::DEFAULT) {
            whole_bundle.set(ncb_param.workspace_ptr);

            //! packb size per group
            size_t packb_bytes_per_group = matmul_bundle.get_size(1);

            size_t GROUP = param.filter_meta.group;
            size_t SH = param.filter_meta.stride[0];
            size_t SW = param.filter_meta.stride[1];
            size_t OH = param.osz[0];
            size_t OW = param.osz[1];
            size_t OC = param.filter_meta.ocpg;
            size_t batch = ncb_index.ndrange_id[0];

            MatrixMulImpl::KernParam matmul_kern_param;
            static_cast<MatrixMulImpl::KernSizeParam&>(matmul_kern_param) =
                    utils::get_matmul_kern_param(param, OH * OW, OC);

            rep(g, GROUP) {
                if (SH == 2 && SW == 2)
                    megdnn_throw("no support for stride = 2");

                size_t bytes_offset_of_b_panel =
                        batch * packb_bytes_per_group * GROUP +
                        g * packb_bytes_per_group;
                src_ctype* b_panel = reinterpret_cast<src_ctype*>(
                        reinterpret_cast<int8_t*>(whole_bundle.get(1)) +
                        bytes_offset_of_b_panel);
                matmul_kern_param.B_ptr = const_cast<src_ctype*>(
                        ncb_param.src<src_ctype>(batch, g));
                matmul_algo->pack_B(matmul_kern_param, b_panel, 0, OH * OW);
            }
        } else {
            megdnn_log_error("OnlyPackA mode and NoPack mode has no packB kernel");
        }
    }

    void exec(WorkspaceBundle& whole_bundle,
              WorkspaceBundle& matmul_bundle,
              WorkspaceBundle& thread_bundle,
              size_t oc_tile_size,
              const MatrixMulImpl::AlgoBase* matmul_algo,
              const ConvBiasImpl::NCBKernSizeParam& param,
              const ConvBiasImpl::NCBKernParam& ncb_param,
              const ConvBiasImpl::NCBKernIndex& ncb_index) override {
        whole_bundle.set(ncb_param.workspace_ptr);
        size_t OC = param.filter_meta.ocpg;
        size_t IC = param.filter_meta.icpg;

        //! packa bytes per group
        size_t oc_tiles_per_group = div_ceil(OC, oc_tile_size);
        size_t packa_bytes_per_oc_tile = matmul_bundle.get_size(0);
        size_t packa_bytes_per_group =
                packa_bytes_per_oc_tile * oc_tiles_per_group;

        //! packb bytes per group
        size_t packb_bytes_per_group = matmul_bundle.get_size(1);

        //! matmul bytes per thread
        size_t matmul_bytes_per_thread = thread_bundle.get_size(0);

        size_t batch_id = ncb_index.ndrange_id[0];
        size_t group_id = ncb_index.ndrange_id[1];
        size_t oc_tile_id_in_group = ncb_index.ndrange_id[2];
        size_t thread_id = ncb_index.thread_id;

        size_t GROUP = param.filter_meta.group;
        size_t OH = param.osz[0];
        size_t OW = param.osz[1];
        size_t oc_start = oc_tile_size * oc_tile_id_in_group;
        size_t oc_end = oc_start + oc_tile_size;
        oc_end = (oc_end <= OC ? oc_end : OC);

        MatrixMulImpl::KernParam matmul_kern_param;
        static_cast<MatrixMulImpl::KernSizeParam&>(matmul_kern_param) =
                utils::get_matmul_kern_param(param, OH * OW, oc_end - oc_start);

        size_t bytes_offset_of_a_panel =
                group_id * packa_bytes_per_group +
                oc_tile_id_in_group * packa_bytes_per_oc_tile;

        int8_t* tmp_ptr =
                is_enable_filter_preprocess(param)
                        ? static_cast<int8_t*>(
                                  param.preprocessed_filter->tensors[0].raw_ptr)
                        : static_cast<int8_t*>(whole_bundle.get(0));

        int8_t* a_panel = tmp_ptr + bytes_offset_of_a_panel;

        size_t bytes_offset_of_b_panel =
                batch_id * packb_bytes_per_group * GROUP +
                group_id * packb_bytes_per_group;
        int8_t* b_panel = reinterpret_cast<int8_t*>(whole_bundle.get(1)) +
                          bytes_offset_of_b_panel;

        size_t thread_offset = thread_bundle.total_size_in_bytes() * thread_id;
        size_t bytes_offset_of_matmul_dst_this_thread =
                thread_offset + thread_bundle.get_size(0);
        int8_t* matmul_temp_dst =
                reinterpret_cast<int8_t*>(whole_bundle.get(2)) +
                bytes_offset_of_matmul_dst_this_thread;

        size_t numbers_of_ncb_dst_offset =
                oc_tile_size * OH * OW * oc_tile_id_in_group;
        void* conv_bias_dst = static_cast<void*>(
                ncb_param.dst<dst_ctype>(batch_id, group_id) +
                numbers_of_ncb_dst_offset);

        size_t numbers_of_ncb_filter_offset =
                oc_tile_size * IC * oc_tile_id_in_group;
        matmul_kern_param.A_ptr = const_cast<src_ctype*>(
                ncb_param.filter<src_ctype>(group_id) +
                numbers_of_ncb_filter_offset);

        matmul_kern_param.B_ptr = const_cast<src_ctype*>(
                ncb_param.src<src_ctype>(batch_id, group_id));

        matmul_kern_param.workspace_ptr =
                reinterpret_cast<int8_t*>(whole_bundle.get(2)) + thread_offset;
        matmul_kern_param.workspace_size = matmul_bytes_per_thread;

        bool is_dst_8bit =
                (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                 param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                (param.src_type.enumv() == DTypeEnum::Quantized8Asymm &&
                 param.dst_type.enumv() == DTypeEnum::Quantized8Asymm);
        void* matmul_dst = is_dst_8bit ? matmul_temp_dst : conv_bias_dst;

        matmul_kern_param.C_ptr = matmul_dst;

        if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::NO_PACK) {
            auto matmul_kern = matmul_algo->get_kern(matmul_kern_param);
            matmul_kern(matmul_kern_param);
        } else {
            auto matmul_kern_naked =
                    matmul_algo->get_kern_naked(matmul_kern_param);
            matmul_kern_naked(matmul_kern_param, a_panel, b_panel);
        }

        //! do postprocess
        void* bias_ptr = nullptr;
        if (param.bias_mode == megdnn::BiasMode::BIAS) {
            bias_ptr = static_cast<void*>(const_cast<bias_ctype*>(
                    ncb_param.bias<bias_ctype>(batch_id, group_id) +
                    numbers_of_ncb_dst_offset));
        } else {
            bias_ptr = static_cast<void*>(const_cast<bias_ctype*>(
                    ncb_param.bias<bias_ctype>(batch_id, group_id) + oc_start));
        }

        PostProcess<op_ctype, op_dtype, postprocess_mode>::run(
                matmul_dst, bias_ptr, conv_bias_dst, param.bias_mode,
                param.nonlineMode, param.bias_type, param.dst_type, 1_z,
                (oc_end - oc_start) / m_pack_size, OH, OW, m_pack_size);
    }
private:
    size_t m_pack_size = 1;
};

class Conv1x1Factory {
public:
    static Conv1x1StrategyBase* make_conv1x1_strategy(
            const ConvBiasImpl::NCBKernSizeParam& param,
            MatrixMulImpl::AlgoBase::PackMode pack_mode,
            param::ConvBias::Format format);

    static bool can_make_conv1x1_strategy(
            const ConvBiasImpl::NCBKernSizeParam& param,
            MatrixMulImpl::AlgoBase::PackMode pack_mode,
            param::ConvBias::Format format);
};
}  // namespace conv1x1
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
