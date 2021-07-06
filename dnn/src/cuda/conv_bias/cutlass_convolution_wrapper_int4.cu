/**
 * \file dnn/src/cuda/conv_bias/cutlass_convolution_wrapper.cu
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
#include "src/cuda/conv_bias/cutlass_convolution_wrapper.cuh"
#pragma GCC diagnostic pop

using namespace megdnn;
using namespace cuda;
using namespace cutlass_wrapper;

/* ====== cutlass kernel wrapper for int4 x int4 nchw64 layout ====== */

#if MEGDNN_TEGRA_X1
template <bool NeedLoadFromConstMem>
void megdnn::cuda::cutlass_wrapper::
        do_conv_bias_int4_int4_implicit_gemm_imma_ncdiv64hw64(
                const int8_t* /* d_src */, const int8_t* /* d_filter */,
                const int32_t* /* d_bias */, const int8_t* /* d_z */,
                int8_t* /* d_dst */, int* /* workspace */,
                const convolution::ConvParam& /* param */,
                uint32_t /* nonlinear_mode */, float /* alpha */,
                float /* beta */, float /* gamma */, float /* scale */,
                const GemmCoord& /* threadblock_shape */,
                const GemmCoord& /* warp_shape */, int /* stages */,
                cudaStream_t /* stream */) {}
#else
template <bool NeedLoadFromConstMem>
void megdnn::cuda::cutlass_wrapper::
        do_conv_bias_int4_int4_implicit_gemm_imma_ncdiv64hw64(
                const int8_t* d_src, const int8_t* d_filter,
                const int32_t* d_bias, const int8_t* d_z, int8_t* d_dst,
                int* workspace, const convolution::ConvParam& param,
                uint32_t nonlinear_mode, float alpha, float beta, float gamma,
                float scale, const GemmCoord& threadblock_shape,
                const GemmCoord& warp_shape, int stages, cudaStream_t stream) {
#define DISPATCH_KERNEL_WITH_TILE_SHAPE(threadblock_m_, threadblock_n_,        \
                                        threadblock_k_, warp_m_, warp_n_,      \
                                        warp_k_, stage_)                       \
    if (threadblock_shape.m() == threadblock_m_ &&                             \
        threadblock_shape.n() == threadblock_n_ &&                             \
        threadblock_shape.k() == threadblock_k_ &&                             \
        warp_shape.m() == warp_m_ && warp_shape.n() == warp_n_ &&              \
        warp_shape.k() == warp_k_ && stages == stage_) {                       \
        using ThreadBlockShape =                                               \
                cutlass::gemm::GemmShape<threadblock_m_, threadblock_n_,       \
                                         threadblock_k_>;                      \
        using WarpShape = cutlass::gemm::GemmShape<warp_m_, warp_n_, warp_k_>; \
        using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;           \
        using Convolution = cutlass::conv::device::Convolution<                \
                cutlass::int4b_t, cutlass::layout::TensorNCxHWx<64>,           \
                cutlass::int4b_t, cutlass::layout::TensorCxRSKx<64>,           \
                ElementOutput, cutlass::layout::TensorNCxHWx<64>, int32_t,     \
                cutlass::layout::TensorNCxHWx<64>, int32_t,                    \
                cutlass::conv::ConvType::kConvolution,                         \
                cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,           \
                ThreadBlockShape, WarpShape, InstructionShape, EpilogueOp,     \
                cutlass::conv::threadblock::                                   \
                        ConvolutionFpropTransThreadblockSwizzle,               \
                stage_, 32, 32, NeedLoadFromConstMem,                          \
                cutlass::arch::OpMultiplyAddSaturate,                          \
                cutlass::conv::ImplicitGemmMode::GEMM_TN, true>;               \
        typename Convolution::ConvolutionParameter conv_param(                 \
                param.n, param.hi, param.wi, param.ci, param.co, param.fh,     \
                param.fw, param.ho, param.wo, param.ph, param.pw, param.sh,    \
                param.sw, 1, 1, cutlass::conv::Mode::kCrossCorrelation);       \
        return cutlass_convolution_wrapper<Convolution>(                       \
                reinterpret_cast<const cutlass::int4b_t*>(d_src),              \
                reinterpret_cast<const cutlass::int4b_t*>(d_filter), d_bias,   \
                reinterpret_cast<const cutlass::int4b_t*>(d_z),                \
                reinterpret_cast<cutlass::int4b_t*>(d_dst), workspace,         \
                conv_param, epilogue, stream);                                 \
    }
#define DISPATCH_KERNEL                                                      \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 128, 128, 64, 64, 128, 2);          \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 256, 128, 64, 64, 128, 2);          \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 128, 64, 64, 128, 2);           \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 64, 64, 64, 64, 1);             \
    megdnn_assert(false,                                                     \
                  "unsupported threadblock shape (%dx%dx%d) and warp shape " \
                  "(%dx%dx%d)",                                              \
                  threadblock_shape.m(), threadblock_shape.n(),              \
                  threadblock_shape.k(), warp_shape.m(), warp_shape.n(),     \
                  warp_shape.k());
    using ElementOutput = cutlass::int4b_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;
    using NonlineMode = megdnn::param_enumv::ConvBias::NonlineMode;
    switch (nonlinear_mode) {
        case NonlineMode::IDENTITY: {
            using EpilogueOp =
                    cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                            ElementOutput, 16, ElementAccumulator, ElementBias,
                            ElementCompute>;
            typename EpilogueOp::Params epilogue{alpha, beta, gamma};
            DISPATCH_KERNEL;
        }
        case NonlineMode::RELU: {
            using EpilogueOp = cutlass::epilogue::thread::
                    BiasAddLinearCombinationReluClamp<
                            ElementOutput, 16, ElementAccumulator, ElementBias,
                            ElementCompute>;
            typename EpilogueOp::Params epilogue{alpha, beta, gamma, 0};
            DISPATCH_KERNEL;
        }
        case NonlineMode::H_SWISH: {
            using EpilogueOp = cutlass::epilogue::thread::
                    BiasAddLinearCombinationHSwishClamp<
                            ElementOutput, 16, ElementAccumulator, ElementBias,
                            ElementCompute>;
            typename EpilogueOp::Params epilogue{alpha, beta, gamma, scale};
            DISPATCH_KERNEL;
        }
        default:
            megdnn_assert(false,
                          "unsupported nonlinear mode for conv bias operator");
    }
#undef DISPATCH_KERNEL_WITH_TILE_SHAPE
#undef DISPATCH_KERNEL
}
#endif

#define INST(need_load_from_const_mem)                                       \
    template void megdnn::cuda::cutlass_wrapper::                            \
            do_conv_bias_int4_int4_implicit_gemm_imma_ncdiv64hw64<           \
                    need_load_from_const_mem>(                               \
                    const int8_t* d_src, const int8_t* d_filter,             \
                    const int32_t* d_bias, const int8_t* d_z, int8_t* d_dst, \
                    int* workspace, const convolution::ConvParam& param,     \
                    uint32_t nonlinear_mode, float alpha, float beta,        \
                    float gamma, float scale,                                \
                    const GemmCoord& threadblock_shape,                      \
                    const GemmCoord& warp_shape, int stages,                 \
                    cudaStream_t stream);
INST(true);
#undef INST

/* ====== cutlass kernel wrapper for uint4 x int4 nchw64 layout ====== */

#if MEGDNN_TEGRA_X1
template <bool NeedLoadFromConstMem>
void megdnn::cuda::cutlass_wrapper::
        do_conv_bias_uint4_int4_implicit_gemm_imma_ncdiv64hw64(
                const uint8_t* /* d_src */, const int8_t* /* d_filter */,
                const int32_t* /* d_bias */, const uint8_t* /* d_z */,
                uint8_t* /* d_dst */, int* /* workspace */,
                const convolution::ConvParam& /* param */,
                uint32_t /* nonlinear_mode */, float /* alpha */,
                float /* beta */, float /* gamma */, float /* delta */,
                float /* theta */, float /* scale */,
                uint8_t /* src_zero_point */,
                const GemmCoord& /* threadblock_shape */,
                const GemmCoord& /* warp_shape */, int /* stages */,
                cudaStream_t /* stream */) {}
#else
template <bool NeedLoadFromConstMem>
void megdnn::cuda::cutlass_wrapper::
        do_conv_bias_uint4_int4_implicit_gemm_imma_ncdiv64hw64(
                const uint8_t* d_src, const int8_t* d_filter,
                const int32_t* d_bias, const uint8_t* d_z, uint8_t* d_dst,
                int* workspace, const convolution::ConvParam& param,
                uint32_t nonlinear_mode, float alpha, float beta, float gamma,
                float delta, float theta, float /* scale */,
                uint8_t src_zero_point, const GemmCoord& threadblock_shape,
                const GemmCoord& warp_shape, int stages, cudaStream_t stream) {
#define DISPATCH_KERNEL_WITH_TILE_SHAPE(threadblock_m_, threadblock_n_,        \
                                        threadblock_k_, warp_m_, warp_n_,      \
                                        warp_k_, stage_)                       \
    if (threadblock_shape.m() == threadblock_m_ &&                             \
        threadblock_shape.n() == threadblock_n_ &&                             \
        threadblock_shape.k() == threadblock_k_ &&                             \
        warp_shape.m() == warp_m_ && warp_shape.n() == warp_n_ &&              \
        warp_shape.k() == warp_k_ && stages == stage_) {                       \
        using ThreadBlockShape =                                               \
                cutlass::gemm::GemmShape<threadblock_m_, threadblock_n_,       \
                                         threadblock_k_>;                      \
        using WarpShape = cutlass::gemm::GemmShape<warp_m_, warp_n_, warp_k_>; \
        using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;           \
        using Convolution = cutlass::conv::device::Convolution<                \
                cutlass::uint4b_t, cutlass::layout::TensorNCxHWx<64>,          \
                cutlass::int4b_t, cutlass::layout::TensorCxRSKx<64>,           \
                ElementOutput, cutlass::layout::TensorNCxHWx<64>, int32_t,     \
                cutlass::layout::TensorNCxHWx<64>, int32_t,                    \
                cutlass::conv::ConvType::kConvolution,                         \
                cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,           \
                ThreadBlockShape, WarpShape, InstructionShape, EpilogueOp,     \
                cutlass::conv::threadblock::                                   \
                        ConvolutionFpropTransThreadblockSwizzle,               \
                stage_, 32, 32, NeedLoadFromConstMem,                          \
                cutlass::arch::OpMultiplyAddSaturate,                          \
                cutlass::conv::ImplicitGemmMode::GEMM_TN, true>;               \
        typename Convolution::ConvolutionParameter conv_param(                 \
                param.n, param.hi, param.wi, param.ci, param.co, param.fh,     \
                param.fw, param.ho, param.wo, param.ph, param.pw, param.sh,    \
                param.sw, 1, 1, cutlass::conv::Mode::kCrossCorrelation);       \
        return cutlass_convolution_wrapper<Convolution>(                       \
                reinterpret_cast<const cutlass::uint4b_t*>(d_src),             \
                reinterpret_cast<const cutlass::int4b_t*>(d_filter), d_bias,   \
                reinterpret_cast<const cutlass::uint4b_t*>(d_z),               \
                reinterpret_cast<cutlass::uint4b_t*>(d_dst), workspace,        \
                conv_param, epilogue, stream, {src_zero_point});               \
    }
#define DISPATCH_KERNEL                                                      \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 128, 128, 64, 64, 128, 2);          \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 256, 128, 64, 64, 128, 2);          \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 128, 64, 64, 128, 2);           \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 64, 64, 64, 64, 1);             \
    megdnn_assert(false,                                                     \
                  "unsupported threadblock shape (%dx%dx%d) and warp shape " \
                  "(%dx%dx%d)",                                              \
                  threadblock_shape.m(), threadblock_shape.n(),              \
                  threadblock_shape.k(), warp_shape.m(), warp_shape.n(),     \
                  warp_shape.k());
    using ElementOutput = cutlass::uint4b_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;
    using NonlineMode = megdnn::param_enumv::ConvBias::NonlineMode;
    switch (nonlinear_mode) {
        case NonlineMode::IDENTITY: {
            using EpilogueOp =
                    cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                            ElementOutput, 16, ElementAccumulator, ElementBias,
                            ElementCompute>;
            typename EpilogueOp::Params epilogue{alpha, beta, gamma,
                                                 delta + theta};
            DISPATCH_KERNEL;
        }
        case NonlineMode::RELU: {
            using EpilogueOp = cutlass::epilogue::thread::
                    BiasAddLinearCombinationReluClamp<
                            ElementOutput, 16, ElementAccumulator, ElementBias,
                            ElementCompute>;
            typename EpilogueOp::Params epilogue{alpha, beta,  gamma,
                                                 0,     delta, theta};
            DISPATCH_KERNEL;
        }
        default:
            megdnn_assert(false,
                          "unsupported nonlinear mode for conv bias operator");
    }
#undef DISPATCH_KERNEL_WITH_TILE_SHAPE
#undef DISPATCH_KERNEL
}
#endif

#define INST(need_load_from_const_mem)                                         \
    template void megdnn::cuda::cutlass_wrapper::                              \
            do_conv_bias_uint4_int4_implicit_gemm_imma_ncdiv64hw64<            \
                    need_load_from_const_mem>(                                 \
                    const uint8_t* d_src, const int8_t* d_filter,              \
                    const int32_t* d_bias, const uint8_t* d_z, uint8_t* d_dst, \
                    int* workspace, const convolution::ConvParam& param,       \
                    uint32_t nonlinear_mode, float alpha, float beta,          \
                    float gamma, float delta, float theta, float scale,        \
                    uint8_t src_zero_point,                                    \
                    const GemmCoord& threadblock_shape,                        \
                    const GemmCoord& warp_shape, int stages,                   \
                    cudaStream_t stream);
INST(true);
#undef INST

/* ====== cutlass kernel wrapper for int4 x int4 nhwc layout ====== */

#if MEGDNN_TEGRA_X1
template <bool NeedLoadFromConstMem>
void megdnn::cuda::cutlass_wrapper::
        do_conv_bias_int4_int4_implicit_gemm_imma_nhwc(
                const int8_t* /* d_src */, const int8_t* /* d_filter */,
                const int32_t* /* d_bias */, const int8_t* /* d_z */,
                int8_t* /* d_dst */, int* /* workspace */,
                const convolution::ConvParam& /* param */,
                uint32_t /* nonlinear_mode */, float /* alpha */,
                float /* beta */, float /* gamma */, float /* scale */,
                const GemmCoord& /* threadblock_shape */,
                const GemmCoord& /* warp_shape */,
                const int32_t /* access_size */, int /* stages */,
                cudaStream_t /* stream */) {}
#else
template <bool NeedLoadFromConstMem>
void megdnn::cuda::cutlass_wrapper::
        do_conv_bias_int4_int4_implicit_gemm_imma_nhwc(
                const int8_t* d_src, const int8_t* d_filter,
                const int32_t* d_bias, const int8_t* d_z, int8_t* d_dst,
                int* workspace, const convolution::ConvParam& param,
                uint32_t nonlinear_mode, float alpha, float beta, float gamma,
                float scale, const GemmCoord& threadblock_shape,
                const GemmCoord& warp_shape, const int32_t access_size,
                int stages, cudaStream_t stream) {
    bool without_shared_load =
            ((param.co % threadblock_shape.n() == 0) &&
             (threadblock_shape.n() == 32 || threadblock_shape.n() == 64));
    int out_elements_per_access =
            without_shared_load ? threadblock_shape.n() / 4 : 8;

#define RUN_CUTLASS_WRAPPER(stage_, access_size_, without_shared_load_)        \
    using Convolution = cutlass::conv::device::Convolution<                    \
            cutlass::int4b_t, cutlass::layout::TensorNHWC, cutlass::int4b_t,   \
            cutlass::layout::TensorNCxHWx<access_size_>, ElementOutput,        \
            cutlass::layout::TensorNHWC, int32_t, cutlass::layout::TensorNHWC, \
            int32_t, cutlass::conv::ConvType::kConvolution,                    \
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,               \
            ThreadBlockShape, WarpShape, InstructionShape, EpilogueOp,         \
            cutlass::conv::threadblock::                                       \
                    ConvolutionFpropTransThreadblockSwizzle,                   \
            stage_, access_size_, access_size_, NeedLoadFromConstMem,          \
            cutlass::arch::OpMultiplyAddSaturate,                              \
            cutlass::conv::ImplicitGemmMode::GEMM_TN, without_shared_load_>;   \
    typename Convolution::ConvolutionParameter conv_param(                     \
            param.n, param.hi, param.wi, param.ci, param.co, param.fh,         \
            param.fw, param.ho, param.wo, param.ph, param.pw, param.sh,        \
            param.sw, 1, 1, cutlass::conv::Mode::kCrossCorrelation);           \
    return cutlass_convolution_wrapper<Convolution>(                           \
            reinterpret_cast<const cutlass::int4b_t*>(d_src),                  \
            reinterpret_cast<const cutlass::int4b_t*>(d_filter), d_bias,       \
            reinterpret_cast<const cutlass::int4b_t*>(d_z),                    \
            reinterpret_cast<cutlass::int4b_t*>(d_dst), workspace, conv_param, \
            epilogue, stream);
#define DISPATCH_KERNEL_WITH_TILE_SHAPE(                                       \
        threadblock_m_, threadblock_n_, threadblock_k_, warp_m_, warp_n_,      \
        warp_k_, stage_, access_size_, out_elements_per_access_,               \
        without_shared_load_)                                                  \
    if (threadblock_shape.m() == threadblock_m_ &&                             \
        threadblock_shape.n() == threadblock_n_ &&                             \
        threadblock_shape.k() == threadblock_k_ &&                             \
        warp_shape.m() == warp_m_ && warp_shape.n() == warp_n_ &&              \
        warp_shape.k() == warp_k_ && stages == stage_ &&                       \
        access_size == access_size_ &&                                         \
        out_elements_per_access == out_elements_per_access_ &&                 \
        without_shared_load == without_shared_load_) {                         \
        using ThreadBlockShape =                                               \
                cutlass::gemm::GemmShape<threadblock_m_, threadblock_n_,       \
                                         threadblock_k_>;                      \
        using WarpShape = cutlass::gemm::GemmShape<warp_m_, warp_n_, warp_k_>; \
        using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;           \
        using ElementOutput = cutlass::int4b_t;                                \
        using ElementAccumulator = int32_t;                                    \
        using ElementBias = int32_t;                                           \
        using ElementCompute = float;                                          \
        using NonlineMode = megdnn::param_enumv::ConvBias::NonlineMode;        \
        switch (nonlinear_mode) {                                              \
            case NonlineMode::IDENTITY: {                                      \
                using EpilogueOp = cutlass::epilogue::thread::                 \
                        BiasAddLinearCombinationClamp<                         \
                                ElementOutput, out_elements_per_access_,       \
                                ElementAccumulator, ElementBias,               \
                                ElementCompute>;                               \
                typename EpilogueOp::Params epilogue{alpha, beta, gamma};      \
                RUN_CUTLASS_WRAPPER(stage_, access_size_,                      \
                                    without_shared_load_);                     \
            }                                                                  \
            case NonlineMode::RELU: {                                          \
                using EpilogueOp = cutlass::epilogue::thread::                 \
                        BiasAddLinearCombinationReluClamp<                     \
                                ElementOutput, out_elements_per_access_,       \
                                ElementAccumulator, ElementBias,               \
                                ElementCompute>;                               \
                typename EpilogueOp::Params epilogue{alpha, beta, gamma, 0};   \
                RUN_CUTLASS_WRAPPER(stage_, access_size_,                      \
                                    without_shared_load_);                     \
            }                                                                  \
            case NonlineMode::H_SWISH: {                                       \
                using EpilogueOp = cutlass::epilogue::thread::                 \
                        BiasAddLinearCombinationHSwishClamp<                   \
                                ElementOutput, out_elements_per_access_,       \
                                ElementAccumulator, ElementBias,               \
                                ElementCompute>;                               \
                typename EpilogueOp::Params epilogue{alpha, beta, gamma,       \
                                                     scale};                   \
                RUN_CUTLASS_WRAPPER(stage_, access_size_,                      \
                                    without_shared_load_);                     \
            }                                                                  \
            default:                                                           \
                megdnn_assert(                                                 \
                        false,                                                 \
                        "unsupported nonlinear mode for conv bias operator");  \
        }                                                                      \
    }
#define DISPATCH_KERNEL                                                        \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 32, 64, 64, 32, 64, 1, 32, 8, false); \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 32, 64, 64, 32, 64, 1, 16, 8, false); \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 32, 64, 64, 32, 64, 1, 8, 8, false);  \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 64, 64, 64, 64, 1, 32, 8, false); \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 64, 64, 64, 64, 1, 16, 8, false); \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 64, 64, 64, 64, 1, 8, 8, false);  \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 32, 64, 64, 32, 64, 1, 32, 8, true);  \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 32, 64, 64, 32, 64, 1, 16, 8, true);  \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 32, 64, 64, 32, 64, 1, 8, 8, true);   \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 64, 64, 64, 64, 1, 32, 16, true); \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 64, 64, 64, 64, 1, 16, 16, true); \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 64, 64, 64, 64, 1, 8, 16, true);  \
    megdnn_assert(false,                                                       \
                  "unsupported threadblock shape (%dx%dx%d) and warp shape "   \
                  "(%dx%dx%d) and access_size (%d)",                           \
                  threadblock_shape.m(), threadblock_shape.n(),                \
                  threadblock_shape.k(), warp_shape.m(), warp_shape.n(),       \
                  warp_shape.k(), access_size);
    DISPATCH_KERNEL;

#undef RUN_CUTLASS_WRAPPER
#undef DISPATCH_KERNEL_WITH_TILE_SHAPE
#undef DISPATCH_KERNEL
}
#endif

#define INST(need_load_from_const_mem)                                       \
    template void megdnn::cuda::cutlass_wrapper::                            \
            do_conv_bias_int4_int4_implicit_gemm_imma_nhwc<                  \
                    need_load_from_const_mem>(                               \
                    const int8_t* d_src, const int8_t* d_filter,             \
                    const int32_t* d_bias, const int8_t* d_z, int8_t* d_dst, \
                    int* workspace, const convolution::ConvParam& param,     \
                    uint32_t nonlinear_mode, float alpha, float beta,        \
                    float gamma, float scale,                                \
                    const GemmCoord& threadblock_shape,                      \
                    const GemmCoord& warp_shape, const int32_t access_size,  \
                    int stages, cudaStream_t stream);
INST(true);
INST(false);
#undef INST

/* ====== cutlass kernel wrapper for uint4 x int4 nhwc layout ====== */

#if MEGDNN_TEGRA_X1
template <bool NeedLoadFromConstMem>
void megdnn::cuda::cutlass_wrapper::
        do_conv_bias_uint4_int4_implicit_gemm_imma_nhwc(
                const uint8_t* /* d_src */, const int8_t* /* d_filter */,
                const int32_t* /* d_bias */, const uint8_t* /* d_z */,
                uint8_t* /* d_dst */, int* /* workspace */,
                const convolution::ConvParam& /* param */,
                uint32_t /* nonlinear_mode */, float /* alpha */,
                float /* beta */, float /* gamma */, float /* delta */,
                float /* theta */, float /* scale */,
                uint8_t /* src_zero_point */,
                const GemmCoord& /* threadblock_shape */,
                const GemmCoord& /* warp_shape */,
                const int32_t /* access_size */, int /* stages */,
                cudaStream_t /* stream */) {}
#else
template <bool NeedLoadFromConstMem>
void megdnn::cuda::cutlass_wrapper::
        do_conv_bias_uint4_int4_implicit_gemm_imma_nhwc(
                const uint8_t* d_src, const int8_t* d_filter,
                const int32_t* d_bias, const uint8_t* d_z, uint8_t* d_dst,
                int* workspace, const convolution::ConvParam& param,
                uint32_t nonlinear_mode, float alpha, float beta, float gamma,
                float delta, float theta, float /* scale */,
                uint8_t src_zero_point, const GemmCoord& threadblock_shape,
                const GemmCoord& warp_shape, const int32_t access_size,
                int stages, cudaStream_t stream) {
    bool without_shared_load =
            ((param.co % threadblock_shape.n() == 0) &&
             (threadblock_shape.n() == 32 || threadblock_shape.n() == 64));
    int out_elements_per_access =
            without_shared_load ? threadblock_shape.n() / 4 : 8;

#define RUN_CUTLASS_WRAPPER(stage_, access_size_, without_shared_load_)        \
    using Convolution = cutlass::conv::device::Convolution<                    \
            cutlass::uint4b_t, cutlass::layout::TensorNHWC, cutlass::int4b_t,  \
            cutlass::layout::TensorNCxHWx<access_size_>, ElementOutput,        \
            cutlass::layout::TensorNHWC, int32_t, cutlass::layout::TensorNHWC, \
            int32_t, cutlass::conv::ConvType::kConvolution,                    \
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,               \
            ThreadBlockShape, WarpShape, InstructionShape, EpilogueOp,         \
            cutlass::conv::threadblock::                                       \
                    ConvolutionFpropTransThreadblockSwizzle,                   \
            stage_, access_size_, access_size_, NeedLoadFromConstMem,          \
            cutlass::arch::OpMultiplyAddSaturate,                              \
            cutlass::conv::ImplicitGemmMode::GEMM_TN, without_shared_load_>;   \
    typename Convolution::ConvolutionParameter conv_param(                     \
            param.n, param.hi, param.wi, param.ci, param.co, param.fh,         \
            param.fw, param.ho, param.wo, param.ph, param.pw, param.sh,        \
            param.sw, 1, 1, cutlass::conv::Mode::kCrossCorrelation);           \
    return cutlass_convolution_wrapper<Convolution>(                           \
            reinterpret_cast<const cutlass::uint4b_t*>(d_src),                 \
            reinterpret_cast<const cutlass::int4b_t*>(d_filter), d_bias,       \
            reinterpret_cast<const cutlass::uint4b_t*>(d_z),                   \
            reinterpret_cast<cutlass::uint4b_t*>(d_dst), workspace,            \
            conv_param, epilogue, stream, {src_zero_point});

#define DISPATCH_KERNEL_WITH_TILE_SHAPE(                                       \
        threadblock_m_, threadblock_n_, threadblock_k_, warp_m_, warp_n_,      \
        warp_k_, stage_, access_size_, out_elements_per_access_,               \
        without_shared_load_)                                                  \
    if (threadblock_shape.m() == threadblock_m_ &&                             \
        threadblock_shape.n() == threadblock_n_ &&                             \
        threadblock_shape.k() == threadblock_k_ &&                             \
        warp_shape.m() == warp_m_ && warp_shape.n() == warp_n_ &&              \
        warp_shape.k() == warp_k_ && stages == stage_ &&                       \
        access_size == access_size_ &&                                         \
        out_elements_per_access == out_elements_per_access_ &&                 \
        without_shared_load == without_shared_load_) {                         \
        using ThreadBlockShape =                                               \
                cutlass::gemm::GemmShape<threadblock_m_, threadblock_n_,       \
                                         threadblock_k_>;                      \
        using WarpShape = cutlass::gemm::GemmShape<warp_m_, warp_n_, warp_k_>; \
        using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;           \
        using ElementOutput = cutlass::uint4b_t;                               \
        using ElementAccumulator = int32_t;                                    \
        using ElementBias = int32_t;                                           \
        using ElementCompute = float;                                          \
        using NonlineMode = megdnn::param_enumv::ConvBias::NonlineMode;        \
        switch (nonlinear_mode) {                                              \
            case NonlineMode::IDENTITY: {                                      \
                using EpilogueOp = cutlass::epilogue::thread::                 \
                        BiasAddLinearCombinationClamp<                         \
                                ElementOutput, out_elements_per_access_,       \
                                ElementAccumulator, ElementBias,               \
                                ElementCompute>;                               \
                typename EpilogueOp::Params epilogue{alpha, beta, gamma,       \
                                                     delta + theta};           \
                RUN_CUTLASS_WRAPPER(stage_, access_size_,                      \
                                    without_shared_load_);                     \
            }                                                                  \
            case NonlineMode::RELU: {                                          \
                using EpilogueOp = cutlass::epilogue::thread::                 \
                        BiasAddLinearCombinationReluClamp<                     \
                                ElementOutput, out_elements_per_access_,       \
                                ElementAccumulator, ElementBias,               \
                                ElementCompute>;                               \
                typename EpilogueOp::Params epilogue{alpha, beta,  gamma,      \
                                                     0,     delta, theta};     \
                RUN_CUTLASS_WRAPPER(stage_, access_size_,                      \
                                    without_shared_load_);                     \
            }                                                                  \
            default:                                                           \
                megdnn_assert(                                                 \
                        false,                                                 \
                        "unsupported nonlinear mode for conv bias operator");  \
        }                                                                      \
    }
#define DISPATCH_KERNEL                                                        \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 32, 64, 64, 32, 64, 1, 32, 8, false); \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 32, 64, 64, 32, 64, 1, 16, 8, false); \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 32, 64, 64, 32, 64, 1, 8, 8, false);  \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 64, 64, 64, 64, 1, 32, 8, false); \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 64, 64, 64, 64, 1, 16, 8, false); \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 64, 64, 64, 64, 1, 8, 8, false);  \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 32, 64, 64, 32, 64, 1, 32, 8, true);  \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 32, 64, 64, 32, 64, 1, 16, 8, true);  \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 32, 64, 64, 32, 64, 1, 8, 8, true);   \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 64, 64, 64, 64, 1, 32, 16, true); \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 64, 64, 64, 64, 1, 16, 16, true); \
    DISPATCH_KERNEL_WITH_TILE_SHAPE(128, 64, 64, 64, 64, 64, 1, 8, 16, true);  \
    megdnn_assert(false,                                                       \
                  "unsupported threadblock shape (%dx%dx%d) and warp shape "   \
                  "(%dx%dx%d) and access_size (%d)",                           \
                  threadblock_shape.m(), threadblock_shape.n(),                \
                  threadblock_shape.k(), warp_shape.m(), warp_shape.n(),       \
                  warp_shape.k(), access_size);

    DISPATCH_KERNEL;

#undef RUN_CUTLASS_WRAPPER
#undef DISPATCH_KERNEL_WITH_TILE_SHAPE
#undef DISPATCH_KERNEL
}
#endif

#define INST(need_load_from_const_mem)                                         \
    template void megdnn::cuda::cutlass_wrapper::                              \
            do_conv_bias_uint4_int4_implicit_gemm_imma_nhwc<                   \
                    need_load_from_const_mem>(                                 \
                    const uint8_t* d_src, const int8_t* d_filter,              \
                    const int32_t* d_bias, const uint8_t* d_z, uint8_t* d_dst, \
                    int* workspace, const convolution::ConvParam& param,       \
                    uint32_t nonlinear_mode, float alpha, float beta,          \
                    float gamma, float delta, float theta, float scale,        \
                    uint8_t src_zero_point,                                    \
                    const GemmCoord& threadblock_shape,                        \
                    const GemmCoord& warp_shape, const int32_t access_size,    \
                    int stages, cudaStream_t stream);
INST(true);
INST(false);
#undef INST

// vim: syntax=cuda.doxygen
