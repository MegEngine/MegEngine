/**
 * \file dnn/src/cuda/handle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/handle_impl.h"
#include "src/common/version_symbol.h"
#include "src/common/api_cache.h"

#include "src/cuda/handle.h"
#include "src/cuda/utils.h"
#include "src/cuda/api_cache.h"
#include "megdnn/common.h"

#include <cuda.h>
#include <cstring>
#include <memory>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define CUDNN_VERSION_STR STR(CUDNN_MAJOR) "." STR(CUDNN_MINOR) "." STR(CUDNN_PATCHLEVEL)

#pragma message "compile with cuDNN " CUDNN_VERSION_STR " "

static_assert(!(CUDNN_MAJOR == 5 && CUDNN_MINOR == 1),
        "cuDNN 5.1.x series has bugs. Use 5.0.x instead.");

#undef STR
#undef STR_HELPER

namespace megdnn {
namespace cuda {

HandleImpl::HandleImpl(megcoreComputingHandle_t comp_handle):
    HandleImplHelper(comp_handle, HandleType::CUDA)
{
    // Get megcore device handle
    megcoreDeviceHandle_t dev_handle;
    megcoreGetDeviceHandle(comp_handle, &dev_handle);
    int dev_id;
    megcoreGetDeviceID(dev_handle, &dev_id);
    if (dev_id < 0) {
        cuda_check(cudaGetDevice(&dev_id));
    }
    m_device_id = dev_id;
    m_device_prop = get_device_prop(dev_id);
    // Get stream from MegCore computing handle.
    megdnn_assert(CUDNN_VERSION == cudnnGetVersion(),
        "cudnn version mismatch: compiled with %d; detected %zu at runtime",
        CUDNN_VERSION, cudnnGetVersion());
#if CUDA_VERSION >= 10010
    megdnn_assert(cublasLtGetVersion() >= 10010,
        "cuda library version is too low to run cublasLt");
#endif
#if CUDNN_VERSION >= 8000
    if (!MGB_GETENV("CUDA_CACHE_PATH")) {
        megdnn_log_warn(R"(
            Cudnn8 will jit ptx code with cache. You can set 
            CUDA_CACHE_MAXSIZE and CUDA_CACHE_PATH environment var to avoid repeat jit(very slow).
            For example `export CUDA_CACHE_MAXSIZE=2147483647` and `export CUDA_CACHE_PATH=/data/.cuda_cache`)");
    }
#endif
    cudnn_check(cudnnCreate(&m_cudnn_handle));
    cublas_check(cublasCreate(&m_cublas_handle));
#if CUDA_VERSION >= 10010
    cublas_check(cublasLtCreate(&m_cublasLt_handle));
#endif
    megcore::getCUDAContext(comp_handle, &m_megcore_context);

    // Set stream for cuDNN and cublas handles.
    cudnn_check(cudnnSetStream(m_cudnn_handle, stream()));
    cublas_check(cublasSetStream(m_cublas_handle, stream()));

    // Note that all cublas scalars (alpha, beta) and scalar results such as dot
    // output resides at device side.
    cublas_check(cublasSetPointerMode(m_cublas_handle,
                CUBLAS_POINTER_MODE_DEVICE));

    // init const scalars
    cuda_check(cudaMalloc(&m_const_scalars, sizeof(ConstScalars)));
    ConstScalars const_scalars_val;
    const_scalars_val.init();
    cuda_check(cudaMemcpyAsync(m_const_scalars, &const_scalars_val,
                sizeof(ConstScalars), cudaMemcpyHostToDevice, stream()));
    cuda_check(cudaStreamSynchronize(stream()));

    // check tk1
    m_is_tegra_k1 = (strcmp(m_device_prop->name, "GK20A") == 0);
    m_cusolver_handle = nullptr;

    m_cudnn_api_cache = std::make_unique<CUDNN>(m_cudnn_handle);
}

HandleImpl::~HandleImpl() noexcept {
    cudnn_check(cudnnDestroy(m_cudnn_handle));
    cublas_check(cublasDestroy(m_cublas_handle));
#if CUDA_VERSION >= 10010
    cublas_check(cublasLtDestroy(m_cublasLt_handle));
#endif
    if (m_cusolver_handle) {
        cusolver_check(cusolverDnDestroy(m_cusolver_handle));
    }
    cuda_check(cudaFree(m_const_scalars));
}

void HandleImpl::ConstScalars::init() {
    f16[0].megdnn_x = 0; f16[1].megdnn_x = 1;
    f32[0] = 0; f32[1] = 1;
    i32[0] = 0; i32[1] = 1;
}

size_t HandleImpl::alignment_requirement() const {
    auto &&prop = m_device_prop;
    return std::max(prop->textureAlignment, prop->texturePitchAlignment);
}

bool HandleImpl::check_cross_dev_copy_constraint(const TensorLayout& src) {
    // is contiguous or can be hold by
    // relayout::param::try_copy_2d/try_copy_last_contig
    return src.is_contiguous() || src.stride[src.ndim - 1] == 1;
}

void HandleImpl::initialize_cusolver() {
    cusolver_check(cusolverDnCreate(&m_cusolver_handle));
    cusolver_check(cusolverDnSetStream(m_cusolver_handle, stream()));
}

size_t HandleImpl::image2d_pitch_alignment() const {
    size_t align = device_prop().texturePitchAlignment;
    return align;
}

HandleImpl::HandleVendorType HandleImpl::vendor_type() const {
    return HandleVendorType::CUDA;
}

HandleImpl::CUDNN& HandleImpl::cudnn() {
    return *m_cudnn_api_cache;
}

HandleImpl::CUDNN::CUDNN(cudnnHandle_t handle) {
    m_handle = handle;
    GetConvolutionForwardWorkspaceSize =
            FunctionCacheBuilder<>()
                    .input<Param<cudnnHandle_t>>()
                    .input<CudnnTensorDescParam>()
                    .input<CudnnFilterDescParam>()
                    .input<CudnnConvDescParam>()
                    .input<CudnnTensorDescParam>()
                    .input<Param<cudnnConvolutionFwdAlgo_t>>()
                    .output<RefParam<size_t>>()
                    .ret<Param<cudnnStatus_t>>()
                    .build(&cudnnGetConvolutionForwardWorkspaceSize);
#if CUDNN_MAJOR >= 7
    GetConvolutionForwardAlgorithm_v7 =
            FunctionCacheBuilder<>()
                    .input<Param<cudnnHandle_t>>()
                    .input<CudnnTensorDescParam>()
                    .input<CudnnFilterDescParam>()
                    .input<CudnnConvDescParam>()
                    .input<CudnnTensorDescParam>()
                    .input<Param<int>>()
                    .output<RefArraySizeParam<int>>()
                    .output<ArrayParam<int,
                                       Param<cudnnConvolutionFwdAlgoPerf_t>>>()
                    .ret<Param<cudnnStatus_t>>()
                    .build(&cudnnGetConvolutionForwardAlgorithm_v7);
    GetConvolutionForwardAlgorithmMaxCount =
            FunctionCacheBuilder<>()
                    .input<Param<cudnnHandle_t>>()
                    .output<RefParam<int>>()
                    .ret<Param<cudnnStatus_t>>()
                    .build(&cudnnGetConvolutionForwardAlgorithmMaxCount);
#endif
    GetConvolutionBackwardDataWorkspaceSize =
            FunctionCacheBuilder<>()
                    .input<Param<cudnnHandle_t>>()
                    .input<CudnnFilterDescParam>()
                    .input<CudnnTensorDescParam>()
                    .input<CudnnConvDescParam>()
                    .input<CudnnTensorDescParam>()
                    .input<Param<cudnnConvolutionBwdDataAlgo_t>>()
                    .output<RefParam<size_t>>()
                    .ret<Param<cudnnStatus_t>>()
                    .build(&cudnnGetConvolutionBackwardDataWorkspaceSize);
#if CUDNN_MAJOR >= 7
    GetConvolutionBackwardDataAlgorithm_v7 =
            FunctionCacheBuilder<>()
                    .input<Param<cudnnHandle_t>>()
                    .input<CudnnFilterDescParam>()
                    .input<CudnnTensorDescParam>()
                    .input<CudnnConvDescParam>()
                    .input<CudnnTensorDescParam>()
                    .input<Param<int>>()
                    .output<RefArraySizeParam<int>>()
                    .output<ArrayParam<
                            int, Param<cudnnConvolutionBwdDataAlgoPerf_t>>>()
                    .ret<Param<cudnnStatus_t>>()
                    .build(&cudnnGetConvolutionBackwardDataAlgorithm_v7);
    GetConvolutionBackwardDataAlgorithmMaxCount =
            FunctionCacheBuilder<>()
                    .input<Param<cudnnHandle_t>>()
                    .output<RefParam<int>>()
                    .ret<Param<cudnnStatus_t>>()
                    .build(&cudnnGetConvolutionBackwardDataAlgorithmMaxCount);
#endif
    GetConvolutionBackwardFilterWorkspaceSize =
            FunctionCacheBuilder<>()
                    .input<Param<cudnnHandle_t>>()
                    .input<CudnnTensorDescParam>()
                    .input<CudnnTensorDescParam>()
                    .input<CudnnConvDescParam>()
                    .input<CudnnFilterDescParam>()
                    .input<Param<cudnnConvolutionBwdFilterAlgo_t>>()
                    .output<RefParam<size_t>>()
                    .ret<Param<cudnnStatus_t>>()
                    .build(&cudnnGetConvolutionBackwardFilterWorkspaceSize);
#if CUDNN_MAJOR >= 7
    GetConvolutionBackwardFilterAlgorithm_v7 =
            FunctionCacheBuilder<>()
                    .input<Param<cudnnHandle_t>>()
                    .input<CudnnTensorDescParam>()
                    .input<CudnnTensorDescParam>()
                    .input<CudnnConvDescParam>()
                    .input<CudnnFilterDescParam>()
                    .input<Param<int>>()
                    .output<RefArraySizeParam<int>>()
                    .output<ArrayParam<
                            int, Param<cudnnConvolutionBwdFilterAlgoPerf_t>>>()
                    .ret<Param<cudnnStatus_t>>()
                    .build(&cudnnGetConvolutionBackwardFilterAlgorithm_v7);
    GetConvolutionBackwardFilterAlgorithmMaxCount =
            FunctionCacheBuilder<>()
                    .input<Param<cudnnHandle_t>>()
                    .output<RefParam<int>>()
                    .ret<Param<cudnnStatus_t>>()
                    .build(&cudnnGetConvolutionBackwardFilterAlgorithmMaxCount);
#endif
}

}  // namespace cuda
}  // namespace megdnn

MEGDNN_VERSION_SYMBOL(CUDA, CUDA_VERSION);
MEGDNN_VERSION_SYMBOL3(CUDNN, CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);

// vim: syntax=cpp.doxygen
