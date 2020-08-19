/**
 * \file dnn/src/cuda/conv_bias/conv_bias_int8.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/cuda/convolution_helper/parameter.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace conv_bias_int8 {

struct LaunchConfig {
    int nr_threads_x;
    int nr_threads_y;
    int nr_threads_z;
    int nr_blocks_x;
    int nr_blocks_y;
    int nr_blocks_z;
    int smem_size_in_bytes;
    LaunchConfig()
            : nr_threads_x{1},
              nr_threads_y{1},
              nr_threads_z{1},
              nr_blocks_x{1},
              nr_blocks_y{1},
              nr_blocks_z{1},
              smem_size_in_bytes{1} {}
};

template <typename BiasVisitor, typename Epilogue>
void do_conv_bias_int8_implicit_gemm_cdiv4hwn4(
        const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias,
        Epilogue epilogue, const convolution::ConvParam& param, float alpha,
        float beta, cudaStream_t stream);

template <typename BiasVisitor, typename Epilogue>
void do_conv_bias_int8_implicit_gemm_cdiv4hwn4_unroll_width(
        const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias,
        Epilogue epilogue, const convolution::ConvParam& param, float alpha,
        float beta, cudaStream_t stream);

template <typename BiasVisitor, typename Epilogue>
void do_conv_bias_int8_implicit_gemm_cdiv4hwn4_ld_64bit(
        const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias,
        Epilogue epilogue, const convolution::ConvParam& param, float alpha,
        float beta, cudaStream_t stream);

template <typename BiasVisitor, typename Epilogue>
void do_conv_bias_int8_implicit_gemm_cdiv4hwn4_ld_64bit_unroll_width(
        const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias,
        Epilogue epilogue, const convolution::ConvParam& param, float alpha,
        float beta, cudaStream_t stream);

template <typename BiasVisitor, typename Epilogue>
void do_conv_bias_int8_implicit_gemm_imma16x16x16_cdiv4hwn4(
        const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias,
        Epilogue epilogue, const convolution::ConvParam& param, float alpha,
        float beta, cudaStream_t stream);

template <typename BiasVisitor, typename Epilogue>
void do_conv_bias_int8_implicit_gemm_imma32x8x16_cdiv4hwn4(
        const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias,
        Epilogue epilogue, const convolution::ConvParam& param, float alpha,
        float beta, cudaStream_t stream);

template <typename BiasVisitor, typename Epilogue>
void do_conv_bias_int8_implicit_gemm_imma8x32x16_cdiv4hwn4(
        const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias,
        Epilogue epilogue, const convolution::ConvParam& param, float alpha,
        float beta, cudaStream_t stream);

template <typename BiasVisitor, typename Epilogue>
void do_conv_bias_int8_implicit_gemm_imma16x16x16_cdiv4hwn4_reorder_filter(
        const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias,
        Epilogue epilogue, const convolution::ConvParam& param, float alpha,
        float beta, cudaStream_t stream);

template <typename BiasVisitor, typename Epilogue>
void do_conv_bias_int8_implicit_gemm_imma32x8x16_cdiv4hwn4_reorder_filter(
        const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias,
        Epilogue epilogue, const convolution::ConvParam& param, float alpha,
        float beta, cudaStream_t stream);

template <typename BiasVisitor, typename Epilogue>
void do_conv_bias_int8_implicit_gemm_imma8x32x16_cdiv4hwn4_reorder_filter(
        const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias,
        Epilogue epilogue, const convolution::ConvParam& param, float alpha,
        float beta, cudaStream_t stream);

template <typename BiasVisitor, typename Epilogue>
void do_conv_bias_int8_implicit_gemm_imma16x16x16_cdiv4hwn4_unroll_width(
        const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias,
        Epilogue epilogue, const convolution::ConvParam& param, float alpha,
        float beta, cudaStream_t stream);

template <typename BiasVisitor, typename Epilogue>
void do_conv_bias_int8_implicit_gemm_imma32x8x16_cdiv4hwn4_unroll_width(
        const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias,
        Epilogue epilogue, const convolution::ConvParam& param, float alpha,
        float beta, cudaStream_t stream);

template <typename BiasVisitor, typename Epilogue>
void do_conv_bias_int8_implicit_gemm_imma8x32x16_cdiv4hwn4_unroll_width(
        const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias,
        Epilogue epilogue, const convolution::ConvParam& param, float alpha,
        float beta, cudaStream_t stream);

}  // namespace conv_bias_int8
}  // namespace cuda
}  // namespace megdnn

#define MARK_USED_VAR                                                          \
    MEGDNN_MARK_USED_VAR(n + ci + hi + wi + co + fh + fw + ho + wo + ph + pw + \
                         sh + sw + dh + dw);

#define UNPACK_CONV_PARAMETER(_filter_meta, _param)    \
    size_t ph = _param.pad_h, pw = _param.pad_w;       \
    size_t sh = _param.stride_h, sw = _param.stride_w; \
    size_t dh = _param.dilate_h, dw = _param.dilate_w; \
    size_t fh = _filter_meta.spatial[0], fw = _filter_meta.spatial[1];

#define UNPACK_CONV_BIAS_NCHW4_PARAM(_src, _filter_meta, _dst, _param)        \
    using Format = param::ConvBias::Format;                                   \
    megdnn_assert(_param.format == Format::NCHW4);                            \
    size_t n = (_src)[0], ci = (_src)[1] * 4, hi = (_src)[2], wi = (_src)[3]; \
    size_t co = (_dst)[1] * 4, ho = (_dst)[2], wo = (_dst)[3];                \
    UNPACK_CONV_PARAMETER(_filter_meta, _param);                              \
    MARK_USED_VAR

#define UNPACK_CONV_BIAS_CHWN4_PARAM(_src, _filter_meta, _dst, _param)        \
    using Format = param::ConvBias::Format;                                   \
    megdnn_assert(_param.format == Format::CHWN4);                            \
    size_t ci = (_src)[0] * 4, hi = (_src)[1], wi = (_src)[2], n = (_src)[3]; \
    size_t co = (_dst)[0] * 4, ho = (_dst)[1], wo = (_dst)[2];                \
    UNPACK_CONV_PARAMETER(_filter_meta, _param);                              \
    MARK_USED_VAR

#define UNPACK_CONV_BIAS_NCHW32_PARAM(_src, _filter_meta, _dst, _param)        \
    using Format = param::ConvBias::Format;                                    \
    megdnn_assert(_param.format == Format::NCHW32);                            \
    size_t n = (_src)[0], ci = (_src)[1] * 32, hi = (_src)[2], wi = (_src)[3]; \
    size_t co = (_dst)[1] * 32, ho = (_dst)[2], wo = (_dst)[3];                \
    UNPACK_CONV_PARAMETER(_filter_meta, _param);                               \
    MARK_USED_VAR

// vim: syntax=cuda.doxygen
