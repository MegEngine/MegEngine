/**
 * \file src/rocm/convolution/chanwise/fwd.cpp.hip
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
// each y-slice of a block works on an (N, CHL_MUL, OH, OW) spatial image at
// given inp_chl
template <typename T, int CHL_MUL_SET, int FH_SET, int FW_SET>
__global__ void kern_fwd(T* dst, const T* src, const T* flt_tot, Param param) {
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

}  // anonymous namespace

template <typename T>
void chanwise::run_fwd(T* dst, const T* src, const T* flt, const Param& param,
                       hipStream_t stream) {
    void (*kern)(T*, const T*, const T*, Param);
    if (param.chl_mul == 1) {
        if (param.flt_h == 3 && param.flt_w == 3) {
            kern = kern_fwd<T, 1, 3, 3>;
        } else if (param.flt_h == 4 && param.flt_w == 4) {
            kern = kern_fwd<T, 1, 4, 4>;
        } else {
            kern = kern_fwd<T, 1, 0, 0>;
        }
    } else {
        kern = kern_fwd<T, 0, 0, 0>;
    }
    int nr_thread = 256,
        nr_out_dimx = param.out_h * param.out_w * param.batch * param.chl_mul;
    dim3 nr_block(param.src_chl,
                  std::min(512, max(nr_out_dimx / (nr_thread * 4), 1)));
    uint32_t shared = param.chl_mul * param.flt_h * param.flt_w * sizeof(T);
    kern<<<nr_block, nr_thread, shared, stream>>>(dst, src, flt, param);
    after_kernel_launch();
}

namespace megdnn {
namespace rocm {
namespace convolution {
namespace chanwise {

#define DO_INST(_ct)                                                  \
    template void run_fwd(_ct*, const _ct*, const _ct*, const Param&, \
                          hipStream_t);
#define INST(_dt) DO_INST(DTypeTrait<_dt>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(INST)

#undef INST
#undef DO_INST

} // namespace chanwise
} // namespace convolution
} // namespace rocm
} // namespace megdnn

// vim: syntax=cuda.doxygen
