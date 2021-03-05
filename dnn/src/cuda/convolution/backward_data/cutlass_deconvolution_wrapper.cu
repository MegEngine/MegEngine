/**
 * \file src/cuda/convolution/backward_data/cutlass_deconvolution_wrapper.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
// ignore warning of cutlass
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#if !MEGDNN_TEGRA_X1
#include "cutlass/convolution/device/convolution.h"
#endif
#include "src/common/opr_param_defs_enumv.cuh"
#include "src/cuda/convolution/backward_data/cutlass_deconvolution_wrapper.cuh"
#pragma GCC diagnostic pop

using namespace megdnn;
using namespace cuda;
using namespace cutlass_wrapper;

/* ================ cutlass kernel wrapper for nchw4 layout ================= */
#if MEGDNN_TEGRA_X1
void megdnn::cuda::cutlass_wrapper::do_deconv_int8_implicit_gemm_dp4a_ncdiv4hw4(
        const int8_t* /* d_src */, const int8_t* /* d_filter */,
        int8_t* /* d_dst */, int* /* workspace */,
        const convolution::ConvParam& /* param */, float /* alpha */,
        const GemmCoord& /* threadblock_shape */,
        const GemmCoord& /* warp_shape */, int /* stages */,
        cudaStream_t /* stream */) {}
#else
void megdnn::cuda::cutlass_wrapper::do_deconv_int8_implicit_gemm_dp4a_ncdiv4hw4(
        const int8_t* d_src, const int8_t* d_filter, int8_t* d_dst,
        int* workspace, const convolution::ConvParam& param, float alpha,
        const GemmCoord& threadblock_shape, const GemmCoord& warp_shape,
        int stages, cudaStream_t stream) {
#define DISPATCH_KERNEL_WITH_TILE_SHAPE(threadblock_m_, threadblock_n_,        \
                                        threadblock_k_, warp_m_, warp_n_,      \
                                        warp_k_, stage_, aligned_)             \
    if (threadblock_shape.m() == threadblock_m_ &&                             \
        threadblock_shape.n() == threadblock_n_ &&                             \
        threadblock_shape.k() == threadblock_k_ &&                             \
        warp_shape.m() == warp_m_ && warp_shape.n() == warp_n_ &&              \
        warp_shape.k() == warp_k_ && stages == stage_) {                       \
        using ThreadBlockShape =                                               \
                cutlass::gemm::GemmShape<threadblock_m_, threadblock_n_,       \
                                         threadblock_k_>;                      \
        using WarpShape = cutlass::gemm::GemmShape<warp_m_, warp_n_, warp_k_>; \
        using InstructionShape = cutlass::gemm::GemmShape<1, 1, 4>;            \
        using Deconvolution = cutlass::conv::device::Deconvolution<            \
                int8_t, cutlass::layout::TensorNCxHWx<4>, int8_t,              \
                cutlass::layout::TensorKxRSCx<4>, ElementOutput,               \
                cutlass::layout::TensorNCxHWx<4>, int32_t,                     \
                cutlass::layout::TensorNCxHWx<4>, int32_t,                     \
                cutlass::arch::OpClassSimt, cutlass::arch::Sm61,               \
                ThreadBlockShape, WarpShape, InstructionShape, EpilogueOp,     \
                cutlass::conv::threadblock::                                   \
                        ConvolutionDgradNCxHWxThreadblockSwizzle,              \
                stage_, 4, aligned_>;                                          \
        typename Deconvolution::ConvolutionParameter conv_param(               \
                param.n, param.hi, param.wi, param.ci, param.co, param.fh,     \
                param.fw, param.ho, param.wo, param.ph, param.pw, param.sh,    \
                param.sw, 1, 1, cutlass::conv::Mode::kCrossCorrelation);       \
        return cutlass_deconvolution_wrapper<Deconvolution>(                   \
                d_src, d_filter, nullptr, nullptr, d_dst, workspace,           \
                conv_param, epilogue, stream);                                 \
    }
#define DISPATCH_KERNEL                                                      \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(16, 64, 8, 16, 64, 8, 2, 4);             \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(16, 128, 16, 16, 64, 16, 2, 4);          \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(16, 128, 16, 16, 128, 16, 1, 8);         \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(32, 128, 32, 32, 64, 32, 2, 16);         \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(64, 128, 32, 64, 32, 32, 2, 16);         \
    megdnn_assert(false,                                                     \
                  "unsupported threadblock shape (%dx%dx%d) and warp shape " \
                  "(%dx%dx%d)",                                              \
                  threadblock_shape.m(), threadblock_shape.n(),              \
                  threadblock_shape.k(), warp_shape.m(), warp_shape.n(),     \
                  warp_shape.k());
    using ElementOutput = int8_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;
    using EpilogueOp = cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
            ElementOutput, 4, ElementAccumulator, ElementBias, ElementCompute>;
    typename EpilogueOp::Params epilogue{alpha, 0, 0};
    DISPATCH_KERNEL;

#undef DISPATCH_KERNEL_WITH_TILE_SHAPE
#undef DISPATCH_KERNEL
}
#endif

// vim: syntax=cuda.doxygen
