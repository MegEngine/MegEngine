/**
 * \file dnn/src/cuda/batch_conv_bias/batch_conv_bias.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/convolution_helper/parameter.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace batch_conv_bias {

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
void do_batch_conv_bias_int8_gemm_ncdiv4hw4(const int8_t* d_src,
                                            const int8_t* d_filter,
                                            BiasVisitor bias, Epilogue epilogue,
                                            const convolution::ConvParam& param,
                                            float alpha, float beta,
                                            cudaStream_t stream);

template <typename BiasVisitor, typename Epilogue>
void do_batch_conv_bias_int8_gemm_ncdiv4hw4_ldg_128(
        const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias,
        Epilogue epilogue, const convolution::ConvParam& param, float alpha,
        float beta, cudaStream_t stream);

template <typename BiasVisitor, typename Epilogue>
void do_batch_conv_bias_int8_implicit_gemm_precomp_ncdiv4hw4(
        const int8_t* d_src, const int8_t* d_filter, int* workspace,
        BiasVisitor bias, Epilogue epilogue,
        const convolution::ConvParam& param, float alpha, float beta,
        cudaStream_t stream);

}  // namespace batch_conv_bias
}  // namespace cuda
}  // namespace megdnn

#define MARK_USED_VAR                                                          \
    MEGDNN_MARK_USED_VAR(n + ci + hi + wi + co + fh + fw + ho + wo + ph + pw + \
                         sh + sw + dh + dw);

#define UNPACK_BATCH_CONV_PARAMETER(_param)            \
    size_t ph = _param.pad_h, pw = _param.pad_w;       \
    size_t sh = _param.stride_h, sw = _param.stride_w; \
    size_t dh = _param.dilate_h, dw = _param.dilate_w;

#define UNPACK_BATCH_CONV_BIAS_NCHW4_PARAM(_src, _filter, _dst, _param)       \
    using Format = param::BatchConvBias::Format;                              \
    megdnn_assert(_param.format == Format::NCHW4);                            \
    size_t n = (_src)[0], ci = (_src)[1] * 4, hi = (_src)[2], wi = (_src)[3]; \
    size_t fh = (_filter)[3], fw = (_filter)[4];                              \
    size_t co = (_dst)[1] * 4, ho = (_dst)[2], wo = (_dst)[3];                \
    UNPACK_BATCH_CONV_PARAMETER(_param);                                      \
    MARK_USED_VAR

// vim: syntax=cuda.doxygen
