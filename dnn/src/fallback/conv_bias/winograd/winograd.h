/**
 * \file dnn/src/fallback/conv_bias/winograd/winograd.h
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

#include <cstddef>
#include "include/megdnn/basic_types.h"
#include "include/megdnn/dtype.h"
#include "include/megdnn/thin/small_vector.h"
#include "src/fallback/conv_bias/opr_impl.h"
#include "src/fallback/matrix_mul/opr_impl.h"

#include "midout.h"
MIDOUT_DECL(megdnn_fallback_conv_bias_winograd_common)

namespace megdnn {
namespace winograd {

/**
 * \brief Winograd convolution
 *
 * The algo is refer to https://arxiv.org/abs/1509.09308.
 *
 * Format: DEFAULT
 * filter: (OC, IC, FH, FW) -> (alpha, alpha, IC, OC)
 * src: (N, C, H, W) -> (N, NR_TILES, alpha, alpha, TILE_SIZE, IC)
 *
 * We will perform gemm on:
 * (TILE_SIZE, IC) x (IC, OC) -> (TILE_SIZE, OC)
 *
 * Format: MK4
 * filter: (OC, IC, FH, FW) -> (alpha, alpha, OCB, ICB, IC_BLOCK_SIZE,
 * OC_BLOCK_SIZE)
 * src: (N, C, H, W) -> (N, NR_TILES, alpha, alpha, ICB, TILE_SIZE,
 * IC_BLOCK_SIZE)
 *
 * We will perform gemm on:
 * (OCB, ICB, IC_BLOCK_SIZE, OC_BLOCK_SIZE) x (ICB, TILE_SIZE, IC_BLOCK_SIZE)
 * = (OCB, TILE_SIZE, OC_BLOCK_SIZE)
 */
//! The default oc size of one thread in multi-threads mode
constexpr static size_t UNIT_OC_SIZE_DEFAULT = 1024;
template <typename Strategy,
          param::MatrixMul::Format format = param::MatrixMul::Format::DEFAULT>
class ConvBias {
    using output_compute_type = typename Strategy::output_compute_type;
    using input_filter_compute_type =
            typename Strategy::input_filter_compute_type;
    using stype = typename Strategy::stype;
    using dst_type = typename Strategy::dst_type;
    using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
    using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;
    using NCBKernIndex = fallback::ConvBiasImpl::NCBKernIndex;
    using NCBKern = fallback::ConvBiasImpl::NCBKern;
    static_assert(
            format == param::MatrixMul::Format::DEFAULT ||
                    (format == param::MatrixMul::Format::MK4 &&
                     Strategy::IC_BLOCK_SIZE == 4 &&
                     Strategy::OC_BLOCK_SIZE == 4) ||
                    (format == param::MatrixMul::Format::MK8 &&
                     Strategy::IC_BLOCK_SIZE == 8 &&
                     Strategy::OC_BLOCK_SIZE == 8),
            "format should be default, mk4 and mk8, if mk4 IC_BLOCK_SIZE and "
            "OC_BLOCK_SIZE should be 4, if mk8 IC_BLOCK_SIZE and "
            "OC_BLOCK_SIZE should be 8");

    Strategy m_strategy;
    size_t m_unit_tile_size;
    //! m_unit_oc_size is must be times of Strategy::OC_BLOCK_SIZE
    size_t m_unit_oc_size;

    WorkspaceBundle get_wbundle(
            const NCBKernSizeParam& param,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo) const {
        size_t OC = param.filter_meta.ocpg;
        size_t IC = param.filter_meta.icpg;
        size_t GROUP = param.filter_meta.group;
        size_t nr_threads = param.nr_threads;
        size_t filter_transform_buf_size = 0;
        //! filter : (alpha, alpha, IC, OC) or (OCB, ICB, IC_BLOCK_SIZE,
        //! OC_BLOCK_SIZE)
        if (param.preprocessed_filter == nullptr &&
            param.filter_meta.format !=
                    param::ConvBias::Format::NCHW_WINOGRAD &&
            param.filter_meta.format !=
                    param::ConvBias::Format::NCHW88_WINOGRAD &&
            param.filter_meta.format !=
                    param::ConvBias::Format::NCHW44_WINOGRAD) {
            filter_transform_buf_size = Strategy::ALPHA * Strategy::ALPHA * OC *
                                        IC * sizeof(input_filter_compute_type);
        }
        size_t winograd_comput_size =
                get_wbundle_compute(param, matmul_algo).total_size_in_bytes() *
                nr_threads;
        if (param.filter_meta.format == param::ConvBias::Format::NCHW ||
            param.filter_meta.format == param::ConvBias::Format::NCHW88 ||
            param.filter_meta.format == param::ConvBias::Format::NCHW44) {
            return WorkspaceBundle(
                    nullptr,
                    {winograd_comput_size, filter_transform_buf_size * GROUP});
        } else {
            megdnn_assert(param.filter_meta.format ==
                                  param::ConvBias::Format::NCHW_WINOGRAD ||
                          param.filter_meta.format ==
                                  param::ConvBias::Format::NCHW88_WINOGRAD ||
                          param.filter_meta.format ==
                                  param::ConvBias::Format::NCHW44_WINOGRAD);
            return WorkspaceBundle(nullptr, {winograd_comput_size});
        }
    }

    WorkspaceBundle get_wbundle_compute(
            const NCBKernSizeParam& param,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo) const {
        size_t OC = param.filter_meta.ocpg;
        size_t IC = param.filter_meta.icpg;
        size_t oc_size = std::min(OC, m_unit_oc_size);
        //! input : (alpha, alpha, unit_tile_size, IC) or (alpha, alpha,
        //! ICB, unit_tile_size, IC_BLOCK_SIZE)
        size_t input_transform_buf_size = Strategy::ALPHA * Strategy::ALPHA *
                                          IC * m_unit_tile_size *
                                          sizeof(input_filter_compute_type);
        //! output : (alpha, alpha, unit_tile_size, OC) or
        //! (alpha, alpha, OCB, unit_tile_size, OC_BLOCK_SIZE)
        size_t output_transform_buf_size = Strategy::ALPHA * Strategy::ALPHA *
                                           oc_size * m_unit_tile_size *
                                           sizeof(output_compute_type);

        //! use for inner temporary usage
        size_t transform_mid_buf_size =
                2 * Strategy::ALPHA * Strategy::ALPHA *
                sizeof(output_compute_type) *
                std::max(Strategy::IC_BLOCK_SIZE, Strategy::OC_BLOCK_SIZE);

        size_t matmul_workspace_size = matmul_algo->get_workspace(
                get_matmul_kern_param(param, m_unit_oc_size));

        //! compute workspace is independent and separated as far as possible
        //! in case of false cache line sharing
        return WorkspaceBundle(
                nullptr, {input_transform_buf_size, output_transform_buf_size,
                          transform_mid_buf_size, matmul_workspace_size});
    }

    WorkspaceBundle get_preprocess_wbundle(
            const NCBKernSizeParam& param) const {
        //! use for inner temporary usage
        size_t transform_mid_buf_size =
                2 * Strategy::ALPHA * Strategy::ALPHA *
                sizeof(output_compute_type) *
                std::max(Strategy::IC_BLOCK_SIZE, Strategy::OC_BLOCK_SIZE);
        size_t nr_threads = param.nr_threads;
        SmallVector<size_t> space_vec(nr_threads, transform_mid_buf_size);
        return WorkspaceBundle{nullptr, space_vec};
    }

public:
    //! Get the m_unit_oc_size, according to the nr_threads and
    //! output_featuremap_size. When single thread the m_unit_oc_size is set
    //! 2048 heuristicly, When multi-threads, the m_unit_oc_size is set
    //! according to nr_threads and out_featuremap_size
    ConvBias(const Strategy& strategy, size_t unit_tile_size,
             const NCBKernSizeParam& param)
            : m_strategy{strategy}, m_unit_tile_size{unit_tile_size} {
        size_t nr_threads = param.nr_threads;
        size_t OC = param.filter_meta.ocpg;
        size_t OH = param.osz[0];
        size_t OW = param.osz[1];
        if (nr_threads > 1) {
            size_t units_h = div_ceil<size_t>(OH, Strategy::OUTPUT_BLOCK_SIZE);
            size_t units_w = div_ceil<size_t>(OW, Strategy::OUTPUT_BLOCK_SIZE);
            size_t nr_units = units_h * units_w;
            size_t nr_parallism_unit =
                    div_ceil<size_t>(nr_units, unit_tile_size);
            if (nr_parallism_unit < nr_threads) {
                m_unit_oc_size = div_ceil<size_t>(OC, nr_threads);
                if (format == param::MatrixMul::Format::MK8) {
                    m_unit_oc_size = round_up<size_t>(m_unit_oc_size, 8);
                } else {
                    m_unit_oc_size = round_up<size_t>(m_unit_oc_size, 4);
                }
            } else {
                m_unit_oc_size = UNIT_OC_SIZE_DEFAULT;
            }
        } else {
            m_unit_oc_size = UNIT_OC_SIZE_DEFAULT;
        }
    }
    ConvBias(const Strategy& strategy, size_t unit_tile_size)
            : m_strategy{strategy}, m_unit_tile_size{unit_tile_size} {
        m_unit_oc_size = UNIT_OC_SIZE_DEFAULT;
    }

    size_t get_workspace_size(
            const NCBKernSizeParam& param,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo) const {
        return get_wbundle(param, matmul_algo).total_size_in_bytes();
    }

    size_t get_preprocess_workspace_size(
            const NCBKernSizeParam& param,
            fallback::MatrixMulImpl::AlgoBase*) const {
        return get_preprocess_wbundle(param).total_size_in_bytes();
    }

    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const NCBKernSizeParam& param, fallback::MatrixMulImpl::AlgoBase*) {
        if (param.filter_meta.format != param::ConvBias::Format::NCHW &&
            param.filter_meta.format != param::ConvBias::Format::NCHW88 &&
            param.filter_meta.format != param::ConvBias::Format::NCHW44) {
            return {};
        }
        size_t OC = param.filter_meta.ocpg;
        size_t IC = param.filter_meta.icpg;
        size_t GROUP = param.filter_meta.group;
        SmallVector<TensorLayout> preprocessed_layouts;
        DType dtype = m_strategy.filter_dtype;
        if (dtype.category() == DTypeCategory::QUANTIZED) {
            if (format == param::MatrixMul::Format::MK4) {
                dtype = dtype::Float32();
            } else if (format == param::MatrixMul::Format::MK8) {
                dtype = dtype::Int16();
            }
        }
        if (format == param::MatrixMul::Format::DEFAULT) {
            preprocessed_layouts.push_back(
                    {{GROUP, Strategy::ALPHA, Strategy::ALPHA, OC, IC}, dtype});
        } else if (format == param::MatrixMul::Format::MK4) {
            preprocessed_layouts.push_back(
                    {{GROUP, Strategy::ALPHA, Strategy::ALPHA, OC / 4, IC / 4,
                      4, 4},
                     dtype});
        } else {
            megdnn_assert(format == param::MatrixMul::Format::MK8);
            preprocessed_layouts.push_back(
                    {{GROUP, Strategy::ALPHA, Strategy::ALPHA, OC / 8, IC / 8,
                      8, 8},
                     dtype});
        }
        return preprocessed_layouts;
    }

    //! Used by winograd_filter_preprocess opr
    void filter_process(const stype* filter_ptr,
                        input_filter_compute_type* filter_transform_buf,
                        void* transform_mid_buf, size_t OC, size_t IC) {
        m_strategy.filter(
                filter_ptr, filter_transform_buf,
                static_cast<input_filter_compute_type*>(transform_mid_buf), OC,
                IC, 0, OC);
    }

    static void filter_process(Strategy strategy,
                               const WorkspaceBundle& bundle_top,
                               const WorkspaceBundle& bundle_compute,
                               const NCBKernParam& kern_param,
                               const NCBKernIndex& ncb_index) {
        size_t compute_workspace_size_per_thread =
                bundle_compute.total_size_in_bytes();
        size_t thread_id = ncb_index.thread_id;
        size_t oc_id = ncb_index.ndrange_id[2];
        size_t group_id = ncb_index.ndrange_id[0];
        size_t OC = kern_param.filter_meta.ocpg;
        size_t IC = kern_param.filter_meta.icpg;
        size_t filter_group_size = Strategy::ALPHA * Strategy::ALPHA * OC * IC *
                                   sizeof(input_filter_compute_type);
        //! Filter trans dst ptr
        input_filter_compute_type* filter_transform_buf =
                reinterpret_cast<input_filter_compute_type*>(
                        reinterpret_cast<uintptr_t>(bundle_top.get(1)) +
                        group_id * filter_group_size);
        //! Filter trans src ptr
        input_filter_compute_type* transform_mid_buf =
                reinterpret_cast<input_filter_compute_type*>(
                        reinterpret_cast<uintptr_t>(bundle_compute.get(2)) +
                        compute_workspace_size_per_thread * thread_id);

        const stype* filter_ptr = kern_param.filter<stype>(group_id);
        size_t oc_start = oc_id, oc_end = oc_id + 1;

        if (kern_param.filter_meta.format == param::ConvBias::Format::NCHW88) {
            oc_start = 8 * oc_id;
            oc_end = oc_start + 8;
        } else if (kern_param.filter_meta.format ==
                   param::ConvBias::Format::NCHW44) {
            oc_start = 4 * oc_id;
            oc_end = oc_start + 4;
        }
        strategy.filter(filter_ptr, filter_transform_buf, transform_mid_buf, OC,
                        IC, oc_start, oc_end);
    }

    static void filter_preprocess(Strategy strategy,
                                  const WorkspaceBundle& bundle,
                                  const TensorND& preprocessed_tensor,
                                  const NCBKernParam& kern_param,
                                  const NCBKernIndex& ncb_index) {
        size_t thread_id = ncb_index.thread_id;
        size_t oc_id = ncb_index.ndrange_id[1];
        size_t group_id = ncb_index.ndrange_id[0];
        size_t OC = kern_param.filter_meta.ocpg;
        size_t IC = kern_param.filter_meta.icpg;
        size_t filter_group_size = Strategy::ALPHA * Strategy::ALPHA * OC * IC *
                                   sizeof(input_filter_compute_type);
        //! Filter trans dst ptr
        input_filter_compute_type* filter_transform_buf =
                reinterpret_cast<input_filter_compute_type*>(
                        reinterpret_cast<uintptr_t>(
                                preprocessed_tensor.raw_ptr) +
                        group_id * filter_group_size);
        //! Filter trans src ptr
        input_filter_compute_type* transform_mid_buf =
                reinterpret_cast<input_filter_compute_type*>(
                        reinterpret_cast<uintptr_t>(bundle.get(thread_id)));

        const stype* filter_ptr = kern_param.filter<stype>(group_id);
        size_t oc_start, oc_end;

        if (kern_param.filter_meta.format == param::ConvBias::Format::NCHW88) {
            oc_start = 8 * oc_id;
            oc_end = oc_start + 8;
        } else if (kern_param.filter_meta.format ==
                   param::ConvBias::Format::NCHW44) {
            oc_start = 4 * oc_id;
            oc_end = oc_start + 4;
        } else {
            oc_start = oc_id;
            oc_end = oc_id + 1;
        }
        strategy.filter(filter_ptr, filter_transform_buf, transform_mid_buf, OC,
                        IC, oc_start, oc_end);
    }

    static void winograd_compute(
            Strategy strategy, const WorkspaceBundle& bundle_top,
            const WorkspaceBundle& bundle_compute,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            fallback::MatrixMulImpl::KernParam matmul_param,
            size_t unit_tile_size, size_t unit_oc_size,
            const NCBKernParam& ncb_param, const NCBKernIndex& ncb_index) {
        size_t OC = ncb_param.filter_meta.ocpg;
        size_t IC = ncb_param.filter_meta.icpg;
        size_t IH = ncb_param.isz[0];
        size_t IW = ncb_param.isz[1];
        size_t OH = ncb_param.osz[0];
        size_t OW = ncb_param.osz[1];
        size_t PH = ncb_param.filter_meta.padding[0];
        size_t PW = ncb_param.filter_meta.padding[1];
        size_t filter_group_size = Strategy::ALPHA * Strategy::ALPHA * OC * IC *
                                   sizeof(input_filter_compute_type);
        size_t compute_workspace_size_per_thread =
                bundle_compute.total_size_in_bytes();

        size_t units_h = div_ceil<size_t>(OH, Strategy::OUTPUT_BLOCK_SIZE);
        size_t units_w = div_ceil<size_t>(OW, Strategy::OUTPUT_BLOCK_SIZE);
        size_t nr_units = units_h * units_w;

        size_t oc_block_id = ncb_index.ndrange_id[3];
        size_t tile_id = ncb_index.ndrange_id[2];
        size_t batch_id = ncb_index.ndrange_id[1];
        size_t group_id = ncb_index.ndrange_id[0];
        size_t thread_id = ncb_index.thread_id;

        const stype* src_ptr = ncb_param.src<stype>(batch_id, group_id);
        dst_type* dst_ptr = ncb_param.dst<dst_type>(batch_id, group_id);
        const output_compute_type* bias_ptr =
                static_cast<const output_compute_type*>(
                        ncb_param.bias<output_compute_type>(batch_id,
                                                            group_id));

        input_filter_compute_type* input_transform_buf =
                reinterpret_cast<input_filter_compute_type*>(
                        reinterpret_cast<uintptr_t>(bundle_compute.get(0)) +
                        compute_workspace_size_per_thread * thread_id);

        output_compute_type* output_transform_buf =
                reinterpret_cast<output_compute_type*>(
                        reinterpret_cast<uintptr_t>(bundle_compute.get(1)) +
                        compute_workspace_size_per_thread * thread_id);
        input_filter_compute_type* transform_mid_buf =
                reinterpret_cast<input_filter_compute_type*>(
                        reinterpret_cast<uintptr_t>(bundle_compute.get(2)) +
                        compute_workspace_size_per_thread * thread_id);

        //! NCHW88_WINOGRAD and NCHW_WINOGRAD is the same offset
        const input_filter_compute_type* filter_transform_buf = nullptr;
        if (nullptr != ncb_param.preprocessed_filter) {
            auto preprocess_raw_ptr =
                    ncb_param.preprocessed_filter->tensors[0].raw_ptr;
            filter_transform_buf = reinterpret_cast<input_filter_compute_type*>(
                    reinterpret_cast<uintptr_t>(preprocess_raw_ptr) +
                    group_id * filter_group_size);
        } else {
            filter_transform_buf =
                    static_cast<const input_filter_compute_type*>(
                            ncb_param.filter<input_filter_compute_type>(
                                    group_id));
            if (ncb_param.filter_meta.format == param::ConvBias::Format::NCHW ||
                ncb_param.filter_meta.format ==
                        param::ConvBias::Format::NCHW88 ||
                ncb_param.filter_meta.format ==
                        param::ConvBias::Format::NCHW44) {
                filter_transform_buf =
                        reinterpret_cast<input_filter_compute_type*>(
                                reinterpret_cast<uintptr_t>(bundle_top.get(1)) +
                                group_id * filter_group_size);
            }
        }
        //! prepare matmul param
        matmul_param.workspace_ptr = reinterpret_cast<void*>(
                reinterpret_cast<uintptr_t>(bundle_compute.get(3)) +
                compute_workspace_size_per_thread * thread_id);
        matmul_param.workspace_size = bundle_compute.get_size(3);
        fallback::MatrixMulImpl::kern_t matmul_kern =
                matmul_algo->get_kern(matmul_param);

        size_t unit_start_idx = tile_id * unit_tile_size;
        size_t nr_tiles_in_unit =
                std::min(nr_units - unit_start_idx, unit_tile_size);
        size_t oc_start_idx = oc_block_id * unit_oc_size;
        size_t nr_oc_in_unit = std::min(OC - oc_start_idx, unit_oc_size);
        megdnn_assert(nr_oc_in_unit % Strategy::OC_BLOCK_SIZE == 0,
                      "The winograd remain oc is not times of OC_BLOCK_SIZE");
        if (format == param::MatrixMul::Format::MK4 ||
            format == param::MatrixMul::Format::MK8) {
            megdnn_assert(nr_tiles_in_unit <= unit_tile_size,
                          "nr_tiles_in_unit: %zu TILE_SIZE:%zu",
                          nr_tiles_in_unit, unit_tile_size);
        }
        //! BTdB
        strategy.input(src_ptr, input_transform_buf, transform_mid_buf,
                       IH, IW, IC, PH, PW, unit_start_idx, nr_tiles_in_unit);

        rep(i, Strategy::ALPHA) rep(j, Strategy::ALPHA) {
            if (format == param::MatrixMul::Format::DEFAULT) {
                matmul_param.A_ptr =
                        input_transform_buf +
                        (i * Strategy::ALPHA + j) * nr_tiles_in_unit * IC;
                matmul_param.B_ptr = filter_transform_buf +
                                     (i * Strategy::ALPHA + j) * OC * IC +
                                     oc_start_idx;

                matmul_param.C_ptr = output_transform_buf +
                                     (i * Strategy::ALPHA + j) *
                                             nr_tiles_in_unit * nr_oc_in_unit;

                matmul_param.M = nr_tiles_in_unit;
                matmul_param.N = nr_oc_in_unit;
                matmul_param.LDB = OC;
                matmul_param.LDC = nr_oc_in_unit;
            } else {
                matmul_param.A_ptr = filter_transform_buf +
                                     (i * Strategy::ALPHA + j) * OC * IC +
                                     oc_start_idx * IC;

                matmul_param.B_ptr =
                        input_transform_buf +
                        (i * Strategy::ALPHA + j) * nr_tiles_in_unit * IC;

                matmul_param.C_ptr = output_transform_buf +
                                     (i * Strategy::ALPHA + j) *
                                             nr_tiles_in_unit * nr_oc_in_unit;
                matmul_param.N = nr_tiles_in_unit;
                matmul_param.M = nr_oc_in_unit;
                matmul_param.LDB = matmul_param.N * Strategy::IC_BLOCK_SIZE;
                matmul_param.LDC = matmul_param.N * Strategy::IC_BLOCK_SIZE;
            }
            matmul_kern(matmul_param);
        }

        //! Y = ATmA
        size_t oc_end_idx = oc_start_idx + nr_oc_in_unit;
        strategy.output(
                output_transform_buf, bias_ptr, dst_ptr,
                reinterpret_cast<output_compute_type*>(transform_mid_buf),
                ncb_param.bias_mode, ncb_param.nonlineMode, OH, OW,
                oc_start_idx, oc_end_idx, unit_start_idx, nr_tiles_in_unit);
    };

    SmallVector<NCBKern> get_preprocess_kerns(
            const NCBKernSizeParam& param, fallback::MatrixMulImpl::AlgoBase*) {
        megdnn_assert(
                param.filter_meta.format == param::ConvBias::Format::NCHW ||
                param.filter_meta.format == param::ConvBias::Format::NCHW88 ||
                param.filter_meta.format == param::ConvBias::Format::NCHW44);
        megdnn_assert(param.preprocessed_filter &&
                      param.preprocessed_filter->tensors.size() > 0);
        size_t OC = param.filter_meta.ocpg;
        size_t GROUP = param.filter_meta.group;
        const TensorND& preprocessed_dst =
                param.preprocessed_filter->tensors[0];
        WorkspaceBundle bundle = get_preprocess_wbundle(param);

        Strategy strategy = m_strategy;
        SmallVector<NCBKern> kerns;
        auto filter_process_kern =
                [strategy, bundle, &preprocessed_dst, this](
                        const NCBKernParam& ncb_param,
                        const NCBKernIndex& ncb_index) mutable {
                    MEGDNN_MARK_USED_VAR(this);
                    MIDOUT_BEGIN(megdnn_fallback_conv_bias_winograd_common,
                                 midout_iv("filter_preprocess"_hash)) {
                        bundle.set(ncb_param.workspace_ptr);
                        filter_preprocess(strategy, bundle, preprocessed_dst,
                                          ncb_param, ncb_index);
                    }
                    MIDOUT_END();
                };
        size_t oc_parallelism = OC;
        if (param.filter_meta.format == param::ConvBias::Format::NCHW88) {
            megdnn_assert(OC % 8 == 0);
            oc_parallelism = OC / 8;
        } else if (param.filter_meta.format ==
                   param::ConvBias::Format::NCHW44) {
            megdnn_assert(OC % 4 == 0);
            oc_parallelism = OC / 4;
        }
        kerns.push_back({filter_process_kern, {GROUP, oc_parallelism}});
        return kerns;
    }

    SmallVector<NCBKern> get_kerns(
            const NCBKernSizeParam& param,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo) {
        size_t N = param.n;
        size_t OC = param.filter_meta.ocpg;
        size_t OH = param.osz[0];
        size_t OW = param.osz[1];
        size_t GROUP = param.filter_meta.group;
        WorkspaceBundle bundle_top = get_wbundle(param, matmul_algo);
        WorkspaceBundle bundle_compute =
                get_wbundle_compute(param, matmul_algo);
        fallback::MatrixMulImpl::KernParam matmul_param;
        static_cast<fallback::MatrixMulImpl::KernSizeParam&>(matmul_param) =
                get_matmul_kern_param(param, m_unit_oc_size);

        size_t unit_tile_size = m_unit_tile_size;
        size_t unit_oc_size = m_unit_oc_size;
        size_t units_h = div_ceil<size_t>(OH, Strategy::OUTPUT_BLOCK_SIZE);
        size_t units_w = div_ceil<size_t>(OW, Strategy::OUTPUT_BLOCK_SIZE);

        size_t nr_units = units_h * units_w;
        size_t nr_hw_tiles = div_ceil<size_t>(nr_units, m_unit_tile_size);
        size_t nr_oc_tiles = div_ceil<size_t>(OC, m_unit_oc_size);

        //! The filter should process ahead
        megdnn_assert(
                param.filter_meta.stride[0] == 1 &&
                param.filter_meta.stride[1] == 1 &&
                (param.filter_meta.format == param::ConvBias::Format::NCHW ||
                 param.filter_meta.format == param::ConvBias::Format::NCHW88 ||
                 param.filter_meta.format == param::ConvBias::Format::NCHW44 ||
                 param.filter_meta.format ==
                         param::ConvBias::Format::NCHW_WINOGRAD ||
                 param.filter_meta.format ==
                         param::ConvBias::Format::NCHW88_WINOGRAD ||
                 param.filter_meta.format ==
                         param::ConvBias::Format::NCHW44_WINOGRAD));

        SmallVector<NCBKern> kerns;
        if (param.preprocessed_filter == nullptr &&
            (param.filter_meta.format == param::ConvBias::Format::NCHW ||
             param.filter_meta.format == param::ConvBias::Format::NCHW88 ||
             param.filter_meta.format == param::ConvBias::Format::NCHW44)) {
            auto filter_process_kern =
                    [strategy = m_strategy, bundle_top, bundle_compute, this](
                            const NCBKernParam& ncb_param,
                            const NCBKernIndex& ncb_index) mutable {
                        MEGDNN_MARK_USED_VAR(this);
                        MIDOUT_BEGIN(megdnn_fallback_conv_bias_winograd_common,
                                     midout_iv("filter_process"_hash)) {
                            bundle_top.set(ncb_param.workspace_ptr);
                            bundle_compute.set(bundle_top.get(0));
                            filter_process(strategy, bundle_top, bundle_compute,
                                           ncb_param, std::move(ncb_index));
                        }
                        MIDOUT_END();
                    };
            size_t oc_parallelism = OC;
            if (param.filter_meta.format == param::ConvBias::Format::NCHW88) {
                megdnn_assert(OC % 8 == 0);
                oc_parallelism = OC / 8;
            } else if (param.filter_meta.format ==
                       param::ConvBias::Format::NCHW44) {
                megdnn_assert(OC % 4 == 0);
                oc_parallelism = OC / 4;
            }
            kerns.push_back({filter_process_kern, {GROUP, 1, oc_parallelism}});
        }
        auto winograd_compute_kern =
                [strategy = m_strategy, bundle_top, bundle_compute, matmul_algo,
                 matmul_param, unit_tile_size, unit_oc_size,
                 this](const NCBKernParam& ncb_param,
                       const NCBKernIndex& ncb_index) mutable {
                    MEGDNN_MARK_USED_VAR(this);
                    MIDOUT_BEGIN(megdnn_fallback_conv_bias_winograd_common,
                                 midout_iv("winograd_compute"_hash)) {
                        bundle_top.set(ncb_param.workspace_ptr);
                        bundle_compute.set(bundle_top.get(0));
                        winograd_compute(strategy, bundle_top, bundle_compute,
                                         matmul_algo, matmul_param,
                                         unit_tile_size, unit_oc_size,
                                         ncb_param, std::move(ncb_index));
                    }
                    MIDOUT_END();
                };
        kerns.push_back(
                {winograd_compute_kern, {GROUP, N, nr_hw_tiles, nr_oc_tiles}});
        return kerns;
    }

    fallback::MatrixMulImpl::KernSizeParam get_matmul_kern_param(
            const NCBKernSizeParam& param, size_t nr_oc_in_unit = 0) const {
        size_t M = 0;
        size_t N = 0;
        size_t K = 0;
        size_t LDA = 0, LDB = 0, LDC = 0;
        if (nr_oc_in_unit == 0) {
            nr_oc_in_unit = param.filter_meta.ocpg;
        }

        if (format == param::MatrixMul::Format::DEFAULT) {
            M = m_unit_tile_size;
            N = nr_oc_in_unit;
            K = param.filter_meta.icpg;
            LDA = K;
            LDB = N;
            LDC = N;
        } else {
            M = nr_oc_in_unit;
            N = m_unit_tile_size;
            K = param.filter_meta.icpg;
            megdnn_assert(K % Strategy::IC_BLOCK_SIZE == 0, "invalid K: %zu",
                          K);
            LDA = K / Strategy::IC_BLOCK_SIZE * Strategy::OC_BLOCK_SIZE *
                  Strategy::IC_BLOCK_SIZE;
            LDB = N * Strategy::IC_BLOCK_SIZE;
            LDC = N * Strategy::IC_BLOCK_SIZE;
        }

        return {DType::from_enum(DTypeTrait<input_filter_compute_type>::enumv),
                DType::from_enum(DTypeTrait<input_filter_compute_type>::enumv),
                DType::from_enum(DTypeTrait<output_compute_type>::enumv),
                M,
                N,
                K,
                LDA,
                LDB,
                LDC,
                false,
                false,
                param::MatrixMul::ComputeMode::DEFAULT,
                format};
    }
};

}  // namespace winograd
}  // namespace megdnn

#define MEGDNN_REG_WINOGRAD_STRATEGY(                                          \
        _stype, _dtype, _input_filter_ctype, _ctype, _output_block_size,       \
        _kernel_size, _ic_block_size, _oc_block_size, _strategy_cls_name)      \
    class _strategy_cls_name {                                                 \
    public:                                                                    \
        using stype = _stype;                                                  \
        using dst_type = _dtype;                                               \
        using output_compute_type = _ctype;                                    \
        using input_filter_compute_type = _input_filter_ctype;                 \
        /**                                                                    \
         * kernel size of convolution, same as \c r                            \
         * output block size, same as \c m                                     \
         */                                                                    \
        constexpr static size_t KERNEL_SIZE = _kernel_size;                    \
        constexpr static size_t OUTPUT_BLOCK_SIZE = _output_block_size;        \
        constexpr static size_t IC_BLOCK_SIZE = _ic_block_size;                \
        constexpr static size_t OC_BLOCK_SIZE = _oc_block_size;                \
        constexpr static size_t ALPHA = KERNEL_SIZE + OUTPUT_BLOCK_SIZE - 1;   \
        /**                                                                    \
         * process \c UNIT_TILE_SIZE small matrix mul once, total tiles is     \
         * N * DIV_UP(OH, OUTPUT_BLOCK_SIZE) * DIV_UP(OW, OUTPUT_BLOCK_SIZE)   \
         */                                                                    \
        const DType src_dtype;                                                 \
        const DType filter_dtype;                                              \
        const DType dst_dtype;                                                 \
        _strategy_cls_name(DType src_dtype, DType filter_dtype,                \
                           DType dst_dtype);                                   \
        void filter(const stype* filter,                                       \
                    input_filter_compute_type* filter_transform_buf,           \
                    input_filter_compute_type* transform_mid_buf, size_t OC,   \
                    size_t IC, size_t oc_start, size_t oc_end);                \
        void input(const stype* input,                                         \
                   input_filter_compute_type* input_transform_buf,             \
                   input_filter_compute_type* transform_mid_buf,               \
                   size_t IH, size_t IW, size_t IC, size_t PH, size_t PW,      \
                   size_t unit_start_idx, size_t nr_tiles_in_unit);            \
        void output(const output_compute_type* output_transform_buf,           \
                    const output_compute_type* bias, dst_type* output,         \
                    output_compute_type* transform_mid_buf, BiasMode bmode,    \
                    NonlineMode nonline_mode, size_t OH, size_t OW,            \
                    size_t oc_start, size_t oc_end, size_t unit_start_idx,     \
                    size_t nr_tiles_in_unit);                                  \
    };

#define MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(_strategy_cls_name)     \
    constexpr size_t _strategy_cls_name::KERNEL_SIZE;             \
    constexpr size_t _strategy_cls_name::OUTPUT_BLOCK_SIZE;       \
    constexpr size_t _strategy_cls_name::ALPHA;                   \
    constexpr size_t _strategy_cls_name::IC_BLOCK_SIZE;           \
    constexpr size_t _strategy_cls_name::OC_BLOCK_SIZE;           \
    _strategy_cls_name::_strategy_cls_name(                       \
            DType src_dtype, DType filter_dtype, DType dst_dtype) \
            : src_dtype(src_dtype),                               \
              filter_dtype(filter_dtype),                         \
              dst_dtype(dst_dtype) {}

#define MEGDNN_WINOGRADS_ALGO_FUN_DEFINE(_class, _fun, _strategy,              \
                                         _midout_flag, _matmul_format)         \
    MEGDNN_MARK_USED_VAR(param);                                               \
    MIDOUT_BEGIN(_midout_flag, midout_iv(#_class #_fun##_hash)) {              \
        _strategy strategy(param.src_type, param.filter_type, param.dst_type); \
        return megdnn::winograd::ConvBias<_strategy, _matmul_format>(          \
                       strategy, m_tile_size, param)                           \
                ._fun(param, m_matmul_algo);                                   \
    }                                                                          \
    MIDOUT_END();

#define MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(_class, _strategy, _midout_flag,  \
                                            _matmul_format)                   \
    size_t ConvBiasImpl::_class::get_workspace(const NCBKernSizeParam& param) \
            const {                                                           \
        MEGDNN_WINOGRADS_ALGO_FUN_DEFINE(_class, get_workspace_size,          \
                                         _strategy, _midout_flag,             \
                                         _matmul_format);                     \
        return 0;                                                             \
    }                                                                         \
    size_t ConvBiasImpl::_class::get_preprocess_workspace(                    \
            const NCBKernSizeParam& param) const {                            \
        MEGDNN_WINOGRADS_ALGO_FUN_DEFINE(                                     \
                _class, get_preprocess_workspace_size, _strategy,             \
                _midout_flag, _matmul_format);                                \
        return 0;                                                             \
    }                                                                         \
    SmallVector<TensorLayout>                                                 \
    ConvBiasImpl::_class::deduce_preprocessed_filter_layout(                  \
            const NCBKernSizeParam& param) const {                            \
        MEGDNN_WINOGRADS_ALGO_FUN_DEFINE(                                     \
                _class, deduce_preprocessed_filter_layout, _strategy,         \
                _midout_flag, _matmul_format);                                \
        return {};                                                            \
    }                                                                         \
    SmallVector<ConvBiasImpl::NCBKern>                                        \
    ConvBiasImpl::_class::dispatch_preprocess_kerns(                          \
            const NCBKernSizeParam& param) const {                            \
        MEGDNN_WINOGRADS_ALGO_FUN_DEFINE(_class, get_preprocess_kerns,        \
                                         _strategy, _midout_flag,             \
                                         _matmul_format);                     \
        return {};                                                            \
    }                                                                         \
    SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::_class::dispatch_kerns(  \
            const NCBKernSizeParam& param) const {                            \
        MEGDNN_WINOGRADS_ALGO_FUN_DEFINE(_class, get_kerns, _strategy,        \
                                         _midout_flag, _matmul_format);       \
        return {};                                                            \
    }

// vim: syntax=cpp.doxygen
