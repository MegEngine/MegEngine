/**
 * \file dnn/src/cuda/convolution3d/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "./opr_impl.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/handle.h"
#include "src/common/utils.h"
#include "src/common/algo_chooser.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {
namespace convolution3d {
    using CanonizedFilterMeta = Convolution3DForward::CanonizedFilterMeta;

    //! conv size descriptor in the forward view
    struct ForwardSizeArgs {
        HandleImpl *handle;
        const TensorLayout *src_layout;
        CanonizedFilterMeta filter_meta;
        const TensorLayout *dst_layout;
        param::Convolution3D::DataType data_type;
    };

    //! whether cudnn is supported for a filter meta
    bool is_cudnn_supported(const ForwardSizeArgs &args);

    struct CUDNNForwardDescs {
        Tensor3DDesc src_desc, dst_desc;
        Filter3DDesc filter_desc;
        Conv3DDesc conv_desc;
        void set(const TensorLayout &src,
                const CanonizedFilterMeta &filter,
                const TensorLayout &dst,
                const param::Convolution3D &param)
        {
            src_desc.set(src);
            filter_desc.set(filter);
            dst_desc.set(dst);
            conv_desc.set(param, filter.group);
        }
    };

    struct CUDNNBwdDataDescs {
        Tensor3DDesc diff_desc, grad_desc;
        Filter3DDesc filter_desc;
        Conv3DDesc conv_desc;
        void set(const CanonizedFilterMeta &filter,
                const TensorLayout &diff,
                const TensorLayout &grad,
                const param::Convolution3D &param)
        {
            filter_desc.set(filter);
            diff_desc.set(diff);
            grad_desc.set(grad);
            conv_desc.set(param, filter.group);
        }
    };

    struct CUDNNBwdFilterDescs {
        Tensor3DDesc diff_desc, src_desc;
        Filter3DDesc grad_desc;
        Conv3DDesc conv_desc;
        void set(const TensorLayout &src,
                const TensorLayout &diff,
                const CanonizedFilterMeta &grad,
                const param::Convolution3D &param)
        {
            src_desc.set(src);
            diff_desc.set(diff);
            grad_desc.set(grad);
            conv_desc.set(param, grad.group);
        }
    };

    /*!
     * \brief flip conv filter
     *
     * Flip conv filter pointed by \p raw_ptr, store result in workspace, and
     * change \p raw_ptr to workspace.
     */
    void flip_filter(const ForwardSizeArgs &args,
            const Workspace &workspace, void *&raw_ptr);

    inline bool cudnn_get_convolution_fwd_algo_helper(
            cudnnHandle_t cudnn_handle, const cudnnTensorDescriptor_t x_desc,
            const cudnnFilterDescriptor_t w_desc,
            const cudnnConvolutionDescriptor_t conv_desc,
            const cudnnTensorDescriptor_t y_desc,
            size_t workspace_limit_in_bytes, cudnnConvolutionFwdAlgo_t* algo,
            bool reproducible) {
        MEGDNN_MARK_USED_VAR(reproducible);
#if CUDNN_MAJOR >= 7
        int algo_max_count = 0;
        cudnn_check(cudnnGetConvolutionForwardAlgorithmMaxCount(
                cudnn_handle, &algo_max_count));
        SmallVector<cudnnConvolutionFwdAlgoPerf_t> algo_perf(algo_max_count);
        int algo_count = 0;
        cudnn_check(cudnnGetConvolutionForwardAlgorithm_v7(
                cudnn_handle, x_desc, w_desc, conv_desc, y_desc, algo_max_count,
                &algo_count, algo_perf.data()));
        for (int i = 0; i < algo_count; ++i) {
            if (algo_perf[i].algo ==
                cudnnConvolutionFwdAlgo_t::
                        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
                continue;
            size_t workspace_size = 0;
            cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(
                    cudnn_handle, x_desc, w_desc, conv_desc, y_desc,
                    algo_perf[i].algo, &workspace_size));
            if (workspace_size > workspace_limit_in_bytes) continue;
            if (!reproducible) {
                *algo = algo_perf[i].algo;
                return true;
            } else {
                if (algo_perf[i].determinism == CUDNN_DETERMINISTIC) {
                    *algo = algo_perf[i].algo;
                    return true;
                }
            }
        }
        return false;
#else
        cudnn_check(cudnnGetConvolutionForwardAlgorithm(
                cudnn_handle, x_desc, w_desc, conv_desc, y_desc,
                CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                workspace_limit_in_bytes, algo));
        return true;
#endif
    }

    inline bool cudnn_get_convolution_bwd_data_algo_helper(
            cudnnHandle_t cudnn_handle, const cudnnFilterDescriptor_t w_desc,
            const cudnnTensorDescriptor_t dy_desc,
            const cudnnConvolutionDescriptor_t conv_desc,
            const cudnnTensorDescriptor_t dx_desc,
            size_t workspace_limit_in_bytes,
            cudnnConvolutionBwdDataAlgo_t* algo, bool reproducible) {
        MEGDNN_MARK_USED_VAR(reproducible);
#if CUDNN_MAJOR >= 7
        int algo_max_count = 0;
        cudnn_check(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
                cudnn_handle, &algo_max_count));
        SmallVector<cudnnConvolutionBwdDataAlgoPerf_t> algo_perf(
                algo_max_count);
        int algo_count = 0;
        cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                cudnn_handle, w_desc, dy_desc, conv_desc, dx_desc,
                algo_max_count, &algo_count, algo_perf.data()));
        for (int i = 0; i < algo_count; ++i) {
            if (algo_perf[i].algo ==
                cudnnConvolutionBwdDataAlgo_t::
                        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING)
                continue;
            size_t workspace_size = 0;
            cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(
                    cudnn_handle, w_desc, dy_desc, conv_desc, dx_desc,
                    algo_perf[i].algo, &workspace_size));
            if (workspace_size > workspace_limit_in_bytes) continue;
            if (!reproducible) {
                *algo = algo_perf[i].algo;
                return true;
            } else {
                if (algo_perf[i].determinism == CUDNN_DETERMINISTIC) {
                    *algo = algo_perf[i].algo;
                    return true;
                }
            }
        }
        return false;
#else
        cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle,
                    w_desc, dy_desc, conv_desc, dx_desc,
                    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                    workspace_limit_in_bytes,
                    algo));
        return true;
#endif
    }

    inline bool cudnn_get_convolution_bwd_filter_algo_helper(
            cudnnHandle_t cudnn_handle, const cudnnTensorDescriptor_t x_desc,
            const cudnnTensorDescriptor_t dy_desc,
            const cudnnConvolutionDescriptor_t conv_desc,
            const cudnnFilterDescriptor_t dw_desc,
            size_t workspace_limit_in_bytes,
            cudnnConvolutionBwdFilterAlgo_t* algo, bool reproducible) {
        MEGDNN_MARK_USED_VAR(reproducible);
#if CUDNN_MAJOR >= 7
        int algo_max_count = 0;
        cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
                cudnn_handle, &algo_max_count));
        SmallVector<cudnnConvolutionBwdFilterAlgoPerf_t> algo_perf(
                algo_max_count);
        int algo_count = 0;
        cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                cudnn_handle, x_desc, dy_desc, conv_desc, dw_desc,
                algo_max_count, &algo_count, algo_perf.data()));
        for (int i = 0; i < algo_count; ++i) {
            if (algo_perf[i].algo ==
                    cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING)
                continue;
            size_t workspace_size = 0;
            cudnn_check(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    cudnn_handle, x_desc, dy_desc, conv_desc, dw_desc,
                    algo_perf[i].algo, &workspace_size));
            if (workspace_size > workspace_limit_in_bytes) continue;
            if (!reproducible) {
                *algo = algo_perf[i].algo;
                return true;
            } else {
                if (algo_perf[i].determinism == CUDNN_DETERMINISTIC) {
                    *algo = algo_perf[i].algo;
                    return true;
                }
            }
        }
        return false;
#else
        cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithm(
                cudnn_handle, x_desc, dy_desc, conv_desc, dw_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                workspace_limit_in_bytes, algo));
        return true;
#endif
    }


} // namespace convolution3d
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
