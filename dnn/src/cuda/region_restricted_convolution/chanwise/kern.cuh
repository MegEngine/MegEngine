#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace region_restricted_convolution {
namespace chanwise {

struct Param {
    int batch, src_chl, src_h, src_w, chl_mul, flt_h, flt_w, out_h, out_w, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w;
    bool is_compute_deafult;
#if MEGDNN_CC_HOST
    static Param load(
            const TensorShape& src, const TensorShape& dst,
            const RegionRestrictedConvolutionForward::CanonizedFilterMeta& fm,
            bool is_compute_deafult_ = true) {
#define U(v) static_cast<int>(v)
        size_t c_pos, hw_pos;
        if (fm.format == param::Convolution::Format::NCHW) {
            c_pos = 1;
            hw_pos = 2;
        } else {
            megdnn_assert_internal(0);
        }
        return {
                U(src[0]),           U(src[c_pos]),     U(src[hw_pos]),
                U(src[hw_pos + 1]),  U(fm.ocpg),        U(fm.spatial[0]),
                U(fm.spatial[1]),    U(dst[hw_pos]),    U(dst[hw_pos + 1]),
                U(fm.padding[0]),    U(fm.padding[1]),  U(fm.stride[0]),
                U(fm.stride[1]),     U(fm.dilation[0]), U(fm.dilation[1]),
                is_compute_deafult_,
        };
#undef U
    }
#endif
};

template <typename T, typename RT>
void run_fwd_depthwise_large_filter(
        T* dst, const T* src, const T* flt, const RT* rin, const RT* rout,
        const Param& param, cudaStream_t stream);

template <typename T, typename RT>
void run_bwd_depthwise_large_filter(
        T* dst, const T* src, const T* flt, const RT* rin, const RT* rout,
        const Param& param, cudaStream_t stream);

}  // namespace chanwise
}  // namespace region_restricted_convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
