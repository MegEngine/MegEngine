/**
 * \file src/rocm/convolution/chanwise/bwd_data.cpp.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */

#include "hip_header.h"
#include "./kern.h.hip"
#include "./kern_helper.h.hip"

using namespace megdnn;
using namespace rocm;
using namespace convolution;
using namespace chanwise;

namespace {

// grid idx is (inp_chl, worker_index)
// each y-slice of a block works on an (N, IH, IW) spatial image at given
// inp_chl
template <typename T, int CHL_MUL_SET, int FH_SET, int FW_SET, int SH_SET,
          int SW_SET>
__global__ void kern_bwd_data(T* src_grad, const T* dst_grad, const T* flt_tot,
                              Param param) {
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

template <typename T>
class KernDispatch {
public:
    typedef void (*kern_ptr_t)(T*, const T*, const T*, Param);

    static kern_ptr_t dispatch(int chl_mul, int fh, int fw, int sh, int sw) {
        if (chl_mul == 1) {
            if (fh == 3 && fw == 3)
                return d1<1, 3, 3>(sh, sw);
            if (fh == 4 && fw == 4)
                return d1<1, 4, 4>(sh, sw);
        }
        return d1<0, 0, 0>(sh, sw);
    }

private:
    template <int chl_mul, int fh, int fw>
    static kern_ptr_t d1(int sh, int sw) {
        if (sh == 1 && sw == 1)
            return kern_bwd_data<T, chl_mul, fh, fw, 1, 1>;
        if (sh == 1 && sw == 2)
            return kern_bwd_data<T, chl_mul, fh, fw, 1, 2>;
        if (sh == 2 && sw == 1)
            return kern_bwd_data<T, chl_mul, fh, fw, 2, 1>;
        if (sh == 2 && sw == 2)
            return kern_bwd_data<T, chl_mul, fh, fw, 2, 2>;
        return kern_bwd_data<T, chl_mul, fh, fw, 0, 0>;
    }
};

}  // anonymous namespace

template <typename T>
void chanwise::run_bwd_data(T* src_grad, const T* dst_grad, const T* flt,
                            const Param& param, hipStream_t stream) {
    typename KernDispatch<T>::kern_ptr_t kern =
            KernDispatch<T>::dispatch(param.chl_mul, param.flt_h, param.flt_w,
                                      param.stride_h, param.stride_w);
    int nr_thread = 256, nr_out_dimx = param.src_h * param.src_w * param.batch;
    dim3 nr_block(param.src_chl,
                  std::min(512, max(nr_out_dimx / (nr_thread * 4), 1)));
    uint32_t shared = param.chl_mul * param.flt_h * param.flt_w * sizeof(T);
    kern<<<nr_block, nr_thread, shared, stream>>>(src_grad, dst_grad, flt,
                                                  param);
    after_kernel_launch();
}

namespace megdnn {
namespace rocm {
namespace convolution {
namespace chanwise {

#define INST(_dt)                                                   \
    template void run_bwd_data(                                     \
            DTypeTrait<_dt>::ctype*, const DTypeTrait<_dt>::ctype*, \
            const DTypeTrait<_dt>::ctype*, const Param&, hipStream_t);
MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(INST)
#undef INST
#undef DO_INST

} // namespace chanwise
} // namespace convolution
} // namespace rocm
} // namespace megdnn

// vim: syntax=cuda.doxygen
