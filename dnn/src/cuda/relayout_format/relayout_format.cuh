#pragma once

#include "megdnn/basic_types.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace relayout_format {

void relayout_format_cuda_nchw_nchwx(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale = 1.f, const float dst_scale = 1.f,
        const uint8_t src_zero_point = 0, const uint8_t dst_zero_point = 0,
        const int group = 1);

void relayout_format_cuda_nchw4_nchw(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const int group);

void relayout_format_cuda_nchwx_nchw(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale = 1.f, const float dst_scale = 1.f,
        const uint8_t src_zero_point = 0, const uint8_t dst_zero_point = 0);

void relayout_format_cuda_nchw_nhwc(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale = 1.f, const float dst_scale = 1.f,
        const uint8_t src_zero_point = 0, const uint8_t dst_zero_point = 0);

void relayout_format_cuda_nhwc_nchw(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale = 1.f, const float dst_scale = 1.f,
        const uint8_t src_zero_point = 0, const uint8_t dst_zero_point = 0);

void relayout_format_cuda_nchw_nchw4_weight(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream);

}  // namespace relayout_format
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
