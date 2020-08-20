/**
 * \file dnn/src/cuda/conv_bias/chanwise/fwd.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "cuda.h"
#include "cuda_fp16.h"
#include "src/cuda/conv_bias/chanwise/kern.cuh"
#include "src/cuda/conv_bias/chanwise/kern_helper.cuh"
#include "src/cuda/fp16_help.cuh"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;
using namespace chanwise;

namespace {

// grid idx is (inp_chl, worker_index)
// each y-slice of a block works on an (N, CHL_MUL, OH, OW) spatial image at
// given inp_chl
template <typename T, int CHL_MUL_SET, int FH_SET, int FW_SET, int SW_SET>
__global__ void kern_fwd_float(T* dst, const T* src, const T* flt_tot,
                               Param param) {
    extern __shared__ uint8_t flt_storage[];
    T* const flt = reinterpret_cast<T*>(flt_storage);

    const uint32_t N = param.batch, IC = param.src_chl, ic = blockIdx.x,
                   IH = param.src_h, IW = param.src_w,
                   CHL_MUL = CHL_MUL_SET ? CHL_MUL_SET : param.chl_mul,
                   FH = FH_SET ? FH_SET : param.flt_h,
                   FW = FW_SET ? FW_SET : param.flt_w, FSIZE = FH * FW,
                   PH = param.pad_h, PW = param.pad_w, SH = param.stride_h,
                   SW = param.stride_w, OH = param.out_h, OW = param.out_w,
                   TOT_OUT = N * CHL_MUL * OH * OW;

    block_memcpy(flt, flt_tot + ic * FSIZE * CHL_MUL, FSIZE * CHL_MUL);

    uint32_t out_idx_ = blockIdx.y * blockDim.x + threadIdx.x,
             nr_out_per_launch = blockDim.x * gridDim.y;
    for (; out_idx_ < TOT_OUT; out_idx_ += nr_out_per_launch) {
        uint32_t out_idx = out_idx_, n, chl_mul, oh, ow;
        out_idx = div_mod(out_idx, OW, ow);
        out_idx = div_mod(out_idx, OH, oh);
        if (CHL_MUL_SET == 1) {
            chl_mul = 0;
            n = out_idx;
        } else {
            n = div_mod(out_idx, CHL_MUL, chl_mul);
        }

        int ih = int(oh * SH) - int(PH), iw = int(ow * SW) - int(PW);
        const T* flt_base = flt + chl_mul * FSIZE;
        const T* src_base = src + int(((n * IC + ic) * IH + ih) * IW + iw);

        T sum(0);

        if (FH_SET && FW_SET) {
#pragma unroll
            for (uint32_t fh = 0; fh < FH; ++fh) {
                // fh + ih < 0 would overflow, so we do not need to check it
                if (static_cast<uint32_t>(fh + ih) < IH) {
#pragma unroll
                    for (uint32_t fw = 0; fw < FW; ++fw) {
                        if (static_cast<uint32_t>(fw + iw) < IW) {
                            sum += flt_base[fh * FW + fw] *
                                   src_base[fh * IW + fw];
                        }
                    }
                }
            }
        } else {
            int fhmax = min(int(FH), int(IH - ih)),
                fwmax = min(int(FW), int(IW - iw));
            for (int fh = max(0, -ih); fh < fhmax; ++fh) {
                for (int fw = max(0, -iw); fw < fwmax; ++fw) {
                    sum += flt_base[fh * FW + fw] * src_base[fh * IW + fw];
                }
            }
        }
        dst[(((n * IC + ic) * CHL_MUL + chl_mul) * OH + oh) * OW + ow] = sum;
    }
}

#if CUDA_VERSION >= 9000
template <typename T, int CHL_MUL_SET, int FH_SET, int FW_SET, int SW_SET>
__global__ void kern_fwd_half(__half* dst, const __half* src,
                              const __half* flt_tot, Param param) {
    extern __shared__ uint8_t flt_storage[];
    __half* const flt = reinterpret_cast<__half*>(flt_storage);

    const uint32_t N = param.batch, IC = param.src_chl, ic = blockIdx.x,
                   IH = param.src_h, IW = param.src_w,
                   CHL_MUL = CHL_MUL_SET ? CHL_MUL_SET : param.chl_mul,
                   FH = FH_SET ? FH_SET : param.flt_h,
                   FW = FW_SET ? FW_SET : param.flt_w, FSIZE = FH * FW,
                   PH = param.pad_h, PW = param.pad_w, SH = param.stride_h,
                   SW = param.stride_w, OH = param.out_h, OW = param.out_w,
                   TOT_OUT = N * CHL_MUL * OH * OW;

    block_memcpy(flt, flt_tot + ic * FSIZE * CHL_MUL, FSIZE * CHL_MUL);

    uint32_t out_idx_ = (blockIdx.y * blockDim.x + threadIdx.x) * 2,
             nr_out_per_launch = (blockDim.x * gridDim.y) * 2;
    for (; out_idx_ < TOT_OUT; out_idx_ += nr_out_per_launch) {
        if (out_idx_ % OW < OW - 1) {
            uint32_t out_idx = out_idx_, n, chl_mul, oh, ow;
            out_idx = div_mod(out_idx, OW, ow);
            out_idx = div_mod(out_idx, OH, oh);
            if (CHL_MUL_SET == 1) {
                chl_mul = 0;
                n = out_idx;
            } else {
                n = div_mod(out_idx, CHL_MUL, chl_mul);
            }

            int ih = int(oh * SH) - int(PH), iw = int(ow * SW) - int(PW);
            const __half* flt_base = flt + chl_mul * FSIZE;
            const __half* src_base =
                    src + int(((n * IC + ic) * IH + ih) * IW + iw);

            __half2 sum{0.0, 0.0};

#pragma unroll
            for (uint32_t fh = 0; fh < FH; ++fh) {
                // fh + ih < 0 would overflow, so we do not need to
                // check it
                if (static_cast<uint32_t>(fh + ih) < IH) {
                    if (FH_SET == 3 && FW_SET == 3 && SW_SET == 1) {
                        __half2 fil0 = {flt_base[fh * FW], flt_base[fh * FW]};
                        __half2 fil1 = {flt_base[fh * FW + 1],
                                        flt_base[fh * FW + 1]};
                        __half2 fil2 = {flt_base[fh * FW + 2],
                                        flt_base[fh * FW + 2]};

                        __half2 src0 = {0.0, 0.0};
                        if (static_cast<uint32_t>(iw) < IW)
                            src0.x = src_base[fh * IW];
                        if (static_cast<uint32_t>(iw + 1) < IW)
                            src0.y = src_base[fh * IW + 1];
                        sum = fma2(src0, fil0, sum);

                        __half2 src2 = {0.0, 0.0};
                        if (static_cast<uint32_t>(iw + 2) < IW)
                            src2.x = src_base[fh * IW + 2];
                        if (static_cast<uint32_t>(iw + 3) < IW)
                            src2.y = src_base[fh * IW + 3];
                        sum = fma2(src2, fil2, sum);

                        __half2 src1 = {src0.y, src2.x};
                        sum = fma2(src1, fil1, sum);
                    } else if (FH_SET == 5 && FW_SET == 5 && SW_SET == 1) {
                        __half2 fil0 = {flt_base[fh * FW], flt_base[fh * FW]};
                        __half2 fil1 = {flt_base[fh * FW + 1],
                                        flt_base[fh * FW + 1]};
                        __half2 fil2 = {flt_base[fh * FW + 2],
                                        flt_base[fh * FW + 2]};
                        __half2 fil3 = {flt_base[fh * FW + 3],
                                        flt_base[fh * FW + 3]};
                        __half2 fil4 = {flt_base[fh * FW + 4],
                                        flt_base[fh * FW + 4]};

                        __half2 src0 = {0.0, 0.0};
                        if (static_cast<uint32_t>(iw) < IW)
                            src0.x = src_base[fh * IW];
                        if (static_cast<uint32_t>(iw + 1) < IW)
                            src0.y = src_base[fh * IW + 1];
                        sum = fma2(src0, fil0, sum);

                        __half2 src2 = {0.0, 0.0};
                        if (static_cast<uint32_t>(iw + 2) < IW)
                            src2.x = src_base[fh * IW + 2];
                        if (static_cast<uint32_t>(iw + 3) < IW)
                            src2.y = src_base[fh * IW + 3];
                        sum = fma2(src2, fil2, sum);

                        __half2 src1 = {src0.y, src2.x};
                        sum = fma2(src1, fil1, sum);

                        __half2 src4 = {0.0, 0.0};
                        if (static_cast<uint32_t>(iw + 4) < IW)
                            src4.x = src_base[fh * IW + 4];
                        if (static_cast<uint32_t>(iw + 5) < IW)
                            src4.y = src_base[fh * IW + 5];
                        sum = fma2(src4, fil4, sum);

                        __half2 src3 = {src2.y, src4.x};
                        sum = fma2(src3, fil3, sum);

                    } else {
#pragma unroll
                        for (uint32_t fw = 0; fw < FW; ++fw) {
                            __half2 fil = {flt_base[fh * FW + fw],
                                           flt_base[fh * FW + fw]};
                            __half2 src = {0.0, 0.0};
                            if (static_cast<uint32_t>(static_cast<int>(fw) +
                                                      iw) < IW)
                                src.x = src_base[fh * IW + fw];
                            if (static_cast<uint32_t>(static_cast<int>(fw) +
                                                      iw + SW) < IW)
                                src.y = src_base[fh * IW + fw + SW];
                            sum = fma2(src, fil, sum);
                        }
                    }
                }
            }

            dst[(((n * IC + ic) * CHL_MUL + chl_mul) * OH + oh) * OW + ow] =
                    sum.x;
            dst[(((n * IC + ic) * CHL_MUL + chl_mul) * OH + oh) * OW + ow + 1] =
                    sum.y;

            continue;
        }
        // two discontinuous output
        for (size_t offset = 0; offset < 2; ++offset) {
            uint32_t out_idx = out_idx_ + offset, n, chl_mul, oh, ow;
            out_idx = div_mod(out_idx, OW, ow);
            out_idx = div_mod(out_idx, OH, oh);
            if (CHL_MUL_SET == 1) {
                chl_mul = 0;
                n = out_idx;
            } else {
                n = div_mod(out_idx, CHL_MUL, chl_mul);
            }

            int ih = int(oh * SH) - int(PH), iw = int(ow * SW) - int(PW);
            const __half* flt_base = flt + chl_mul * FSIZE;
            const __half* src_base =
                    src + int(((n * IC + ic) * IH + ih) * IW + iw);

            __half sum(0);

            if (FH_SET && FW_SET) {
#pragma unroll
                for (uint32_t fh = 0; fh < FH; ++fh) {
                    // fh + ih < 0 would overflow, so we do not need to
                    // check it
                    if (static_cast<uint32_t>(fh + ih) < IH) {
#pragma unroll
                        for (uint32_t fw = 0; fw < FW; ++fw) {
                            if (static_cast<uint32_t>(fw + iw) < IW) {
                                sum = fma(flt_base[fh * FW + fw],
                                          src_base[fh * IW + fw], sum);
                            }
                        }
                    }
                }
            } else {
                int fhmax = min(int(FH), int(IH - ih)),
                    fwmax = min(int(FW), int(IW - iw));
                for (int fh = max(0, -ih); fh < fhmax; ++fh) {
                    for (int fw = max(0, -iw); fw < fwmax; ++fw) {
                        sum = fma(flt_base[fh * FW + fw],
                                  src_base[fh * IW + fw], sum);
                    }
                }
            }
            dst[(((n * IC + ic) * CHL_MUL + chl_mul) * OH + oh) * OW + ow] =
                    sum;

            if (n == N - 1 && chl_mul == CHL_MUL - 1 && ow == OW - 1 &&
                oh == OH - 1)
                break;
        }
    }
}
#endif

#define SET_SW(func, type, sw)                         \
    if (param.flt_h == 2 && param.flt_w == 2) {        \
        f_struct.f = func<type, 1, 2, 2, sw>;          \
    } else if (param.flt_h == 3 && param.flt_w == 3) { \
        f_struct.f = func<type, 1, 3, 3, sw>;          \
    } else if (param.flt_h == 5 && param.flt_w == 5) { \
        f_struct.f = func<type, 1, 5, 5, sw>;          \
    } else if (param.flt_h == 7 && param.flt_w == 7) { \
        f_struct.f = func<type, 1, 7, 7, sw>;          \
    } else {                                           \
        f_struct.f = func<type, 1, 0, 0, sw>;          \
    }

#define GET_KERN(func, type)                 \
    FixFunction<type> f_struct;              \
    if (param.chl_mul == 1) {                \
        if (param.stride_w == 1) {           \
            SET_SW(func, type, 1)            \
        } else {                             \
            SET_SW(func, type, 0)            \
        }                                    \
    } else {                                 \
        f_struct.f = func<type, 0, 0, 0, 0>; \
    }                                        \
    return f_struct;

template <typename T>
struct FixFunction {
    void (*f)(T*, const T*, const T*, Param);
};

template <typename T>
FixFunction<T> get_kern(const Param& param);

template <>
FixFunction<float> get_kern<float>(const Param& param) {
    GET_KERN(kern_fwd_float, float);
}

#if CUDA_VERSION >= 9000
template <>
FixFunction<__half> get_kern<__half>(const Param& param) {
    GET_KERN(kern_fwd_half, __half);
}
#endif

template <>
FixFunction<dt_float16> get_kern<dt_float16>(const Param& param) {
    GET_KERN(kern_fwd_float, dt_float16);
}

#undef SET_SW
#undef GET_KERN

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace conv_bias {
namespace chanwise {

template <typename T>
void run_fwd(T* dst, const T* src, const T* flt, const Param& param,
             cudaStream_t stream) {
    void (*kern)(T*, const T*, const T*, Param);
    kern = get_kern<T>(param).f;

    int nr_thread = query_blocksize_for_kernel(kern),
        nr_out_dimx = param.out_h * param.out_w * param.batch * param.chl_mul;
    dim3 nr_block(param.src_chl,
                  std::min(512, max(nr_out_dimx / (nr_thread * 4), 1)));
    uint32_t shared = param.chl_mul * param.flt_h * param.flt_w * sizeof(T);
    kern<<<nr_block, nr_thread, shared, stream>>>(dst, src, flt, param);
    after_kernel_launch();
}

template void run_fwd(float*, const float*, const float*, const Param&,
                      cudaStream_t);

#if CUDA_VERSION >= 9000
template void run_fwd(__half*, const __half*, const __half*, const Param&,
                      cudaStream_t);
#endif

template void run_fwd(dt_float16*, const dt_float16*, const dt_float16*,
                      const Param&, cudaStream_t);

}  // namespace chanwise
}  // namespace conv_bias
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
