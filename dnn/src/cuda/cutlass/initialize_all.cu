/***************************************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/**
 * \file dnn/src/cuda/cutlass/initialize_all.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/cutlass/manifest.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////
#if ((__CUDACC_VER_MAJOR__ > 10) || \
     (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))
#define CUTLASS_ARCH_MMA_SM75_SUPPORTED 1
#endif

#if __CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ >= 2)

void initialize_all_gemm_simt_operations(Manifest& manifest);
void initialize_all_conv2d_simt_operations(Manifest& manifest);
void initialize_all_deconv_simt_operations(Manifest& manifest);
void initialize_all_dwconv2d_fprop_simt_operations(Manifest& manifest);
#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED) && CUTLASS_ARCH_MMA_SM75_SUPPORTED
void initialize_all_gemm_tensorop884_operations(Manifest& manifest);
void initialize_all_gemm_tensorop1688_operations(Manifest& manifest);
void initialize_all_conv2d_tensorop8816_operations(Manifest& manifest);
void initialize_all_conv2d_tensorop8832_operations(Manifest& manifest);
void initialize_all_deconv_tensorop8816_operations(Manifest& manifest);
void initialize_all_dwconv2d_fprop_tensorop884_operations(Manifest& manifest);
#endif

void initialize_all(Manifest& manifest) {
    initialize_all_gemm_simt_operations(manifest);
    initialize_all_conv2d_simt_operations(manifest);
    initialize_all_deconv_simt_operations(manifest);
    initialize_all_dwconv2d_fprop_simt_operations(manifest);
#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED) && CUTLASS_ARCH_MMA_SM75_SUPPORTED
    initialize_all_gemm_tensorop884_operations(manifest);
    initialize_all_gemm_tensorop1688_operations(manifest);
    initialize_all_conv2d_tensorop8816_operations(manifest);
    initialize_all_conv2d_tensorop8832_operations(manifest);
    initialize_all_deconv_tensorop8816_operations(manifest);
    initialize_all_dwconv2d_fprop_tensorop884_operations(manifest);
#endif
}

#else

void initialize_all(Manifest& manifest) {}

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace library
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
