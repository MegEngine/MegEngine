#pragma once

#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace pooling2d {

struct Param {
    int n, c, hi, wi, ho, wo, ph, pw, window_h, window_w, sh, sw;
};

void do_pooling2d_int8_cdiv4hwn4(
        const int8_t* d_src, int8_t* d_dst, const Param& param, cudaStream_t stream,
        uint32_t mode);

void do_pooling2d_int8_ncdiv4hw4(
        const int8_t* d_src, int8_t* d_dst, const Param& param, cudaStream_t stream,
        uint32_t mode, bool uint_case = false, int zero_point = 0);

void do_pooling2d_int8_ncdiv32hw32(
        const int8_t* d_src, int8_t* d_dst, const Param& param, cudaStream_t stream,
        uint32_t mode, bool uint_case = false, int zero_point = 0);

void do_pooling2d_int4_ncdiv64hw64(
        const int8_t* d_src, int8_t* d_dst, const Param& param, cudaStream_t stream,
        uint32_t mode, bool uint_case = false, int zero_point = 0);

void do_pooling2d_int4_nhwc(
        const int8_t* d_src, int8_t* d_dst, const Param& param, cudaStream_t stream,
        uint32_t mode, bool uint_case = false, int zero_point = 0);

}  // namespace pooling2d
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
