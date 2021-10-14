/**
 * \file dnn/src/cuda/convolution3d/chanwise/kern.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/cuda/utils.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

#if MEGDNN_CC_HOST
#include "src/cuda/convolution3d/helper.h"
#endif

namespace megdnn {
namespace cuda {
namespace convolution3d {
namespace chanwise {

struct Param {
    uint32_t batch, src_chl, src_d, src_h, src_w, chl_mul, flt_d, flt_h, flt_w, out_d,
            out_h, out_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, dilation_d,
            dilation_h, dilation_w;
#if MEGDNN_CC_HOST
    static Param from_fwd_args(const ForwardSizeArgs& args) {
#define U(v) static_cast<uint32_t>(v)
        auto&& src = args.src_layout->shape;
        auto&& dst = args.dst_layout->shape;
        auto&& fm = args.filter_meta;
        size_t c_pos, hw_pos;
        if (fm.format == param::Convolution3D::Format::NCDHW) {
            c_pos = 1;
            hw_pos = 2;
        } else {  // NDHWC
            c_pos = 4;
            hw_pos = 1;
        }
        return {
                U(src[0]),          U(src[c_pos]),      U(src[hw_pos]),
                U(src[hw_pos + 1]), U(src[hw_pos + 2]), U(fm.ocpg),
                U(fm.spatial[0]),   U(fm.spatial[1]),   U(fm.spatial[2]),
                U(dst[hw_pos]),     U(dst[hw_pos + 1]), U(dst[hw_pos + 2]),
                U(fm.padding[0]),   U(fm.padding[1]),   U(fm.padding[2]),
                U(fm.stride[0]),    U(fm.stride[1]),    U(fm.stride[2]),
                U(fm.dilation[0]),  U(fm.dilation[1]),  U(fm.dilation[2]),
        };
#undef U
    }
#endif
};

template <typename T>
void run_fwd(
        T* dst, const T* src, const T* flt, const Param& param, cudaStream_t stream);

template <typename T>
void run_bwd_data(
        T* src_grad, const T* dst_grad, const T* flt, const Param& param,
        cudaStream_t stream);

template <typename T>
void run_bwd_filter(
        T* filter_grad, const T* src, const T* dst_grad, const Param& param,
        cudaStream_t stream);

}  // namespace chanwise
}  // namespace convolution3d
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
