/**
 * \file dnn/src/cuda/convolution/chanwise/bwd_data.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./kern.cuh"
#include "./kern_helper.cuh"
#include "cuda_fp16.h"
#include "src/cuda/fp16_help.cuh"

using namespace megdnn;
using namespace cuda;
using namespace convolution;
using namespace chanwise;

namespace {

// grid idx is (inp_chl, worker_index)
// each y-slice of a block works on an (N, IH, IW) spatial image at given
// inp_chl
template <typename T, int CHL_MUL_SET, int FH_SET, int FW_SET, int SH_SET,
          int SW_SET>
__global__ void kern_bwd_data_float(T* src_grad, const T* dst_grad,
                                    const T* flt_tot, Param param) {
    // extern __shared__ of dt_float16 does not work
    extern __shared__ uint8_t flt_storage[];

    T* const flt = reinterpret_cast<T*>(flt_storage);

    const uint32_t N = param.batch, IC = param.src_chl, ic = blockIdx.x,
                   IH = param.src_h, IW = param.src_w,
                   CHL_MUL = CHL_MUL_SET ? CHL_MUL_SET : param.chl_mul,
                   FH = FH_SET ? FH_SET : param.flt_h,
                   FW = FW_SET ? FW_SET : param.flt_w, FSIZE = FH * FW,
                   PH = param.pad_h, PW = param.pad_w,
                   SH = SH_SET ? SH_SET : param.stride_h,
                   SW = SW_SET ? SW_SET : param.stride_w, OH = param.out_h,
                   OW = param.out_w, TOT_OUT = N * IH * IW;

    block_memcpy(flt, flt_tot + ic * FSIZE * CHL_MUL, FSIZE * CHL_MUL);
    dst_grad += ic * CHL_MUL * OH * OW;
    src_grad += ic * IH * IW;

    uint32_t out_idx_ = blockIdx.y * blockDim.x + threadIdx.x,
             nr_out_per_launch = blockDim.x * gridDim.y;
    for (; out_idx_ < TOT_OUT; out_idx_ += nr_out_per_launch) {
        uint32_t out_idx = out_idx_, n, ih, iw;
        out_idx = div_mod(out_idx, IW, iw);
        out_idx = div_mod(out_idx, IH, ih);
        n = out_idx;

        const T* dst_grad_base = dst_grad + n * (IC * CHL_MUL * OH * OW);

        T sum(0);

        // o >= max(0, floor_div((i+P-F+1), S))
        uint32_t ohmin = max(int32_t(ih + PH - FH + SH), 0) / SH,
                 owmin = max(int32_t(iw + PW - FW + SW), 0) / SW,
                 ohmax = min((ih + PH) / SH, OH - 1),
                 owmax = min((iw + PW) / SW, OW - 1);
        if (SH_SET == 1 && SW_SET == 1 && FH_SET && FW_SET) {
#pragma unroll
            for (uint32_t doh = 0; doh < FH; ++doh) {
                uint32_t oh = ohmin + doh;
                if (oh <= ohmax) {
                    uint32_t fh = ih - oh * SH + PH;
#pragma unroll
                    for (uint32_t dow = 0; dow < FW; ++dow) {
                        uint32_t ow = owmin + dow;
                        if (ow <= owmax) {
                            uint32_t fw = iw - ow * SW + PW;
                            const T* pd = dst_grad_base + oh * OW + ow;
                            const T* pf = flt + fh * FW + fw;
#pragma unroll
                            for (uint32_t chl_mul = 0; chl_mul < CHL_MUL;
                                 ++chl_mul) {
                                sum += *pd * *pf;
                                pd += OH * OW;
                                pf += FSIZE;
                            }
                        }
                    }
                }
            }
        } else {
            for (uint32_t oh = ohmin; oh <= ohmax; ++oh) {
                uint32_t fh = ih - oh * SH + PH;
                for (uint32_t ow = owmin; ow <= owmax; ++ow) {
                    uint32_t fw = iw - ow * SW + PW;
                    const T* pd = dst_grad_base + oh * OW + ow;
                    const T* pf = flt + fh * FW + fw;
#pragma unroll
                    for (uint32_t chl_mul = 0; chl_mul < CHL_MUL; ++chl_mul) {
                        sum += *pd * *pf;
                        pd += OH * OW;
                        pf += FSIZE;
                    }
                }
            }
        }

        src_grad[(n * (IC * IH) + ih) * IW + iw] = sum;
    }
}

#if CUDA_VERSION >= 9000
template <typename T, int CHL_MUL_SET, int FH_SET, int FW_SET, int SH_SET,
          int SW_SET>
__global__ void kern_bwd_data_hf(__half* src_grad, const __half* dst_grad,
                                 const __half* flt_tot, Param param) {
    extern __shared__ uint8_t flt_storage[];

    __half* const flt = reinterpret_cast<__half*>(flt_storage);

    const uint32_t N = param.batch, IC = param.src_chl, ic = blockIdx.x,
                   IH = param.src_h, IW = param.src_w,
                   CHL_MUL = CHL_MUL_SET ? CHL_MUL_SET : param.chl_mul,
                   FH = FH_SET ? FH_SET : param.flt_h,
                   FW = FW_SET ? FW_SET : param.flt_w, FSIZE = FH * FW,
                   PH = param.pad_h, PW = param.pad_w,
                   SH = SH_SET ? SH_SET : param.stride_h,
                   SW = SW_SET ? SW_SET : param.stride_w, OH = param.out_h,
                   OW = param.out_w, TOT_OUT = N * IH * IW;

    block_memcpy(flt, flt_tot + ic * FSIZE * CHL_MUL, FSIZE * CHL_MUL);
    dst_grad += ic * CHL_MUL * OH * OW;
    src_grad += ic * IH * IW;

    uint32_t out_idx_ = (blockIdx.y * blockDim.x + threadIdx.x) * 2,
             nr_out_per_launch = (blockDim.x * gridDim.y) * 2;
    for (; out_idx_ < TOT_OUT; out_idx_ += nr_out_per_launch) {
        if (out_idx_ % IW < IW - 1) {
            uint32_t out_idx = out_idx_, n, ih, iw;
            out_idx = div_mod(out_idx, IW, iw);
            out_idx = div_mod(out_idx, IH, ih);
            n = out_idx;

            const __half* dst_grad_base =
                    dst_grad + n * (IC * CHL_MUL * OH * OW);

            __half2 sum{0.0, 0.0};
            __half2 pd2{0.0, 0.0};
            __half2 pf2{0.0, 0.0};

            uint32_t ohmin = max(int32_t(ih + PH - FH + SH), 0) / SH,
                     owmin_x = max(int32_t(iw + PW - FW + SW), 0) / SW,
                     owmin_y = max(int32_t(iw + 1 + PW - FW + SW), 0) / SW,
                     ohmax = min((ih + PH) / SH, OH - 1),
                     owmax_x = min((iw + PW) / SW, OW - 1),
                     owmax_y = min((iw + 1 + PW) / SW, OW - 1);
            if (SH_SET == 1 && SW_SET == 1 && FH_SET && FW_SET) {
#pragma unroll
                for (uint32_t doh = 0; doh < FH; ++doh) {
                    uint32_t oh = ohmin + doh;
                    if (oh <= ohmax) {
                        uint32_t fh = ih - oh + PH;
                        uint32_t owmin = owmin_x, owmax = owmax_y;

                        const __half* pd = dst_grad_base + oh * OW;
                        const __half* pf = flt + fh * FW;

                        if (FW == 3) {
#pragma unroll
                            for (uint32_t chl_mul = 0; chl_mul < CHL_MUL;
                                 ++chl_mul) {
                                __half2 flt0 = {0.0, *(pf)},
                                        flt1 = {*(pf), *(pf + 1)},
                                        flt2 = {*(pf + 1), *(pf + 2)},
                                        flt3 = {*(pf + 2), 0.0};
                                uint32_t ow = owmin;
                                uint32_t fw = iw - ow + PW;
                                __half2 dst2 = {0.0, 0.0};
                                if (static_cast<uint32_t>(ow) <
                                    static_cast<uint32_t>(owmin_y)) {
                                    dst2.x = *(pd + ow);
                                    dst2.y = 0.0;
                                    sum = fma2(dst2, flt3, sum);
                                    ++ow;
                                    --fw;
                                }
                                if (static_cast<uint32_t>(owmax_x) <
                                    static_cast<uint32_t>(owmax)) {
                                    dst2.x = 0.0;
                                    dst2.y = *(pd + owmax);
                                    sum = fma2(dst2, flt0, sum);
                                }
                                if (static_cast<uint32_t>(fw) == 1) {
                                    dst2.x = *(pd + ow);
                                    dst2.y = *(pd + ow);
                                    sum = fma2(dst2, flt2, sum);
                                    ++ow;
                                    --fw;
                                }
                                if (static_cast<uint32_t>(ow) <=
                                    static_cast<uint32_t>(owmax_x)) {
                                    dst2.x = *(pd + ow);
                                    dst2.y = *(pd + ow);
                                    sum = fma2(dst2, flt1, sum);
                                }

                                pd += OH * OW;
                                pf += FSIZE;
                            }
                        } else if (FW == 5) {
#pragma unroll
                            for (uint32_t chl_mul = 0; chl_mul < CHL_MUL;
                                 ++chl_mul) {
                                __half2 flt0 = {0.0, *(pf)},
                                        flt1 = {*(pf), *(pf + 1)},
                                        flt2 = {*(pf + 1), *(pf + 2)},
                                        flt3 = {*(pf + 2), *(pf + 3)},
                                        flt4 = {*(pf + 3), *(pf + 4)},
                                        flt5 = {*(pf + 4), 0.0};
                                uint32_t ow = owmin;
                                uint32_t fw = iw - ow + PW;
                                __half2 dst2 = {0.0, 0.0};
                                if (static_cast<uint32_t>(ow) <
                                    static_cast<uint32_t>(owmin_y)) {
                                    dst2.x = *(pd + ow);
                                    dst2.y = 0.0;
                                    sum = fma2(dst2, flt5, sum);
                                    ++ow;
                                    --fw;
                                }
                                if (static_cast<uint32_t>(owmax_x) <
                                    static_cast<uint32_t>(owmax)) {
                                    dst2.x = 0.0;
                                    dst2.y = *(pd + owmax);
                                    sum = fma2(dst2, flt0, sum);
                                }
                                if (static_cast<uint32_t>(fw) == 3) {
                                    dst2.x = *(pd + ow);
                                    dst2.y = *(pd + ow);
                                    sum = fma2(dst2, flt4, sum);
                                    ++ow;
                                    --fw;
                                }
                                if (static_cast<uint32_t>(fw) == 2 &&
                                    static_cast<uint32_t>(ow) <=
                                            static_cast<uint32_t>(owmax_x)) {
                                    dst2.x = *(pd + ow);
                                    dst2.y = *(pd + ow);
                                    sum = fma2(dst2, flt3, sum);
                                    ++ow;
                                    --fw;
                                }
                                if (static_cast<uint32_t>(fw) == 1 &&
                                    static_cast<uint32_t>(ow) <=
                                            static_cast<uint32_t>(owmax_x)) {
                                    dst2.x = *(pd + ow);
                                    dst2.y = *(pd + ow);
                                    sum = fma2(dst2, flt2, sum);
                                    ++ow;
                                    --fw;
                                }
                                if (static_cast<uint32_t>(fw) == 0 &&
                                    static_cast<uint32_t>(ow) <=
                                            static_cast<uint32_t>(owmax_x)) {
                                    dst2.x = *(pd + ow);
                                    dst2.y = *(pd + ow);
                                    sum = fma2(dst2, flt1, sum);
                                }

                                pd += OH * OW;
                                pf += FSIZE;
                            }
                        } else {
#pragma unroll
                            for (uint32_t chl_mul = 0; chl_mul < CHL_MUL;
                                 ++chl_mul) {
#pragma unroll
                                for (uint32_t dow = 0; dow <= FW; ++dow) {
                                    uint32_t ow = owmin + dow;
                                    uint32_t fw = iw - ow + PW;
                                    if (static_cast<uint32_t>(ow) <=
                                        static_cast<uint32_t>(owmax)) {
                                        pd2.x = *(pd + ow);
                                        pd2.y = *(pd + ow);
                                        pf2.x = 0.0;
                                        pf2.y = 0.0;
                                        if (static_cast<uint32_t>(ow) >=
                                            static_cast<uint32_t>(owmin_y))
                                            pf2.y = *(pf + fw + 1);
                                        if (static_cast<uint32_t>(ow) <=
                                            static_cast<uint32_t>(owmax_x))
                                            pf2.x = *(pf + fw);
                                        sum = fma2(pd2, pf2, sum);
                                    }
                                }
                                pd += OH * OW;
                                pf += FSIZE;
                            }
                        }
                    }
                }
            } else {
#pragma unroll
                for (uint32_t oh = ohmin; oh <= ohmax; ++oh) {
                    uint32_t fh = ih - oh * SH + PH;

                    if (owmin_x < owmin_y) {
                        uint32_t fw = iw - owmin_x * SW + PW;
                        const __half* pd = dst_grad_base + oh * OW + owmin_x;
                        const __half* pf = flt + fh * FW + fw;
#pragma unroll
                        for (uint32_t chl_mul = 0; chl_mul < CHL_MUL;
                             ++chl_mul) {
                            pd2.x = *pd;
                            pd2.y = 0.0;
                            pf2.x = *pf;
                            pf2.y = 0.0;
                            sum = fma2(pd2, pf2, sum);
                            pd += OH * OW;
                            pf += FSIZE;
                        }
                    }

                    if (owmax_x < owmax_y) {
                        uint32_t fw = iw + 1 - owmax_y * SW + PW;
                        const __half* pd = dst_grad_base + oh * OW + owmax_y;
                        const __half* pf = flt + fh * FW + fw;
#pragma unroll
                        for (uint32_t chl_mul = 0; chl_mul < CHL_MUL;
                             ++chl_mul) {
                            pd2.x = 0.0;
                            pd2.y = *pd;
                            pf2.x = 0.0;
                            pf2.y = *pf;
                            sum = fma2(pd2, pf2, sum);
                            pd += OH * OW;
                            pf += FSIZE;
                        }
                    }

                    uint32_t ow = owmin_y;
                    uint32_t owmax = owmax_x;
#pragma unroll
                    for (; ow <= owmax; ++ow) {
                        uint32_t fw = iw - ow * SW + PW;
                        const __half* pd = dst_grad_base + oh * OW + ow;
                        const __half* pf = flt + fh * FW + fw;
#pragma unroll
                        for (uint32_t chl_mul = 0; chl_mul < CHL_MUL;
                             ++chl_mul) {
                            pd2.x = *pd;
                            pd2.y = *pd;
                            pf2.x = *pf;
                            pf2.y = *(pf + 1);
                            sum = fma2(pd2, pf2, sum);
                            pd += OW * OH;
                            pf += FSIZE;
                        }
                    }
                }
            }

            src_grad[(n * (IC * IH) + ih) * IW + iw] = sum.x;
            src_grad[(n * (IC * IH) + ih) * IW + iw + 1] = sum.y;
        } else {
            size_t offset = 0;

            for (offset = 0; offset < 2; ++offset) {
                uint32_t out_idx = out_idx_ + offset, n, ih, iw;
                out_idx = div_mod(out_idx, IW, iw);
                out_idx = div_mod(out_idx, IH, ih);
                n = out_idx;

                const __half* dst_grad_base =
                        dst_grad + n * (IC * CHL_MUL * OH * OW);

                __half sum(0);

                uint32_t ohmin = max(int32_t(ih + PH - FH + SH), 0) / SH,
                         owmin = max(int32_t(iw + PW - FW + SW), 0) / SW,
                         ohmax = min((ih + PH) / SH, OH - 1),
                         owmax = min((iw + PW) / SW, OW - 1);
                if (SH_SET == 1 && SW_SET == 1 && FH_SET && FW_SET) {
#pragma unroll
                    for (uint32_t doh = 0; doh < FH; ++doh) {
                        uint32_t oh = ohmin + doh;
                        if (oh <= ohmax) {
                            uint32_t fh = ih - oh * SH + PH;
#pragma unroll
                            for (uint32_t dow = 0; dow < FW; ++dow) {
                                uint32_t ow = owmin + dow;
                                if (ow <= owmax) {
                                    uint32_t fw = iw - ow * SW + PW;
                                    const __half* pd =
                                            dst_grad_base + oh * OW + ow;
                                    const __half* pf = flt + fh * FW + fw;
#pragma unroll
                                    for (uint32_t chl_mul = 0;
                                         chl_mul < CHL_MUL; ++chl_mul) {
                                        sum = fma(*pd, *pf, sum);
                                        pd += OH * OW;
                                        pf += FSIZE;
                                    }
                                }
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (uint32_t oh = ohmin; oh <= ohmax; ++oh) {
                        uint32_t fh = ih - oh * SH + PH;
#pragma unroll
                        for (uint32_t ow = owmin; ow <= owmax; ++ow) {
                            uint32_t fw = iw - ow * SW + PW;
                            const __half* pd = dst_grad_base + oh * OW + ow;
                            const __half* pf = flt + fh * FW + fw;
#pragma unroll
                            for (uint32_t chl_mul = 0; chl_mul < CHL_MUL;
                                 ++chl_mul) {
                                sum = fma(*pd, *pf, sum);
                                pd += OH * OW;
                                pf += FSIZE;
                            }
                        }
                    }
                }

                src_grad[(n * (IC * IH) + ih) * IW + iw] = sum;

                if (ih == IH - 1 && iw == IW - 1 && n == N - 1)
                    break;
            }
        }
    }
}
#endif

#define sh param.stride_h
#define sw param.stride_w
#define SET_STRIDE(func, type, chl_mul, fh, fw)         \
    if (sh == 1 && sw == 1) {                           \
        f_struct.f = func<type, chl_mul, fh, fw, 1, 1>; \
    } else if (sh == 2 && sw == 2) {                    \
        f_struct.f = func<type, chl_mul, fh, fw, 2, 2>; \
    } else {                                            \
        f_struct.f = func<type, chl_mul, fh, fw, 0, 0>; \
    }

#define GET_KERN(func, type)                               \
    FixFunction<type> f_struct;                            \
    if (param.chl_mul == 1) {                              \
        if (param.flt_h == 3 && param.flt_w == 3) {        \
            SET_STRIDE(func, type, 1, 3, 3);               \
        } else if (param.flt_h == 5 && param.flt_w == 5) { \
            SET_STRIDE(func, type, 1, 5, 5);               \
        } else if (param.flt_h == 7 && param.flt_w == 7) { \
            SET_STRIDE(func, type, 1, 7, 7);               \
        } else {                                           \
            SET_STRIDE(func, type, 0, 0, 0);               \
        }                                                  \
    } else {                                               \
        SET_STRIDE(func, type, 0, 0, 0);                   \
    }                                                      \
    return f_struct;

template <typename T>
struct FixFunction {
    void (*f)(T*, const T*, const T*, const Param);
};

template <typename T>
FixFunction<T> get_kern(const Param& param);

template <>
FixFunction<float> get_kern<float>(const Param& param) {
    GET_KERN(kern_bwd_data_float, float);
}

#if CUDA_VERSION >= 9000
template <>
FixFunction<__half> get_kern<__half>(const Param& param) {
    GET_KERN(kern_bwd_data_hf, __half);
}
#endif

template <>
FixFunction<dt_float16> get_kern<dt_float16>(const Param& param) {
    GET_KERN(kern_bwd_data_float, dt_float16);
}

#undef sh
#undef sw
#undef SET_STRIDE
#undef GET_KERN
}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace convolution {
namespace chanwise {

template <typename T>
void run_bwd_data(T* src_grad, const T* dst_grad, const T* flt,
                  const Param& param, cudaStream_t stream) {
    void (*kern)(T*, const T*, const T*, Param);
    kern = get_kern<T>(param).f;

    int nr_thread = query_blocksize_for_kernel(kern),
        nr_out_dimx = param.src_h * param.src_w * param.batch;
    dim3 nr_block(param.src_chl,
                  std::min(512, max(nr_out_dimx / (nr_thread * 4), 1)));
    uint32_t shared = param.chl_mul * param.flt_h * param.flt_w * sizeof(T);
    kern<<<nr_block, nr_thread, shared, stream>>>(src_grad, dst_grad, flt,
                                                  param);
    after_kernel_launch();
}

template void run_bwd_data(float*, const float*, const float*, const Param&,
                           cudaStream_t);

#if CUDA_VERSION >= 9000
template void run_bwd_data(__half*, const __half*, const __half*, const Param&,
                           cudaStream_t);
#endif

template void run_bwd_data(dt_float16*, const dt_float16*, const dt_float16*,
                           const Param&, cudaStream_t);

}  // namespace chanwise
}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen

