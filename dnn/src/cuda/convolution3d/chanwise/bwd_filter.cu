/**
 * \file dnn/src/cuda/convolution3d/chanwise/bwd_filter.cu
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

const uint32_t WARP_SIZE = 32, BATCH_UNROLL = 4;

using namespace megdnn;
using namespace cuda;
using namespace convolution3d;
using namespace chanwise;

namespace {

template<typename T, uint32_t nr_thpf>
__global__ void kern_bwd_filter(
        T *flt_grad, const T *src, const T *dst_grad, Param param) {

    const uint32_t
        N = param.batch, IC = param.src_chl, 
        ID = param.src_d, IH = param.src_h, IW = param.src_w,
        CHL_MUL = param.chl_mul,
        FD = param.flt_d, FH = param.flt_h, FW = param.flt_w,
        PD = param.pad_d, PH = param.pad_h, PW = param.pad_w,
        SD = param.stride_d, SH = param.stride_h, SW = param.stride_w,
        OD = param.out_d, OH = param.out_h, OW = param.out_w,
        SRC_BATCH_STRIDE = IC * ID * IH * IW,
        DST_BATCH_STRIDE = IC * CHL_MUL * OD * OH * OW,
        BLKDIM_X = blockDim.x / nr_thpf,
        THREADID_X = threadIdx.x / nr_thpf,
        OUT_IDX = blockIdx.x * BLKDIM_X + THREADID_X;

    uint32_t ic, chl_mul, fd, fh, fw;
    {
        uint32_t i = OUT_IDX;
        i = div_mod(i, FW, fw);
        i = div_mod(i, FH, fh);
        i = div_mod(i, FD, fd);
        i = div_mod(i, CHL_MUL, chl_mul);
        ic = i;
    }
    if (ic >= IC) {
        return;
    }
    src += ic * ID * IH * IW;
    dst_grad += (ic * CHL_MUL + chl_mul) * OD * OH * OW;

    const uint32_t
        od_lo = max(int32_t(PD - fd + SD - 1), 0) / SD,
        od_hi = min((ID - 1 + PD - fd) / SD + 1, OD),
        oh_lo = max(int32_t(PH - fh + SH - 1), 0) / SH,
        oh_hi = min((IH - 1 + PH - fh) / SH + 1, OH),
        ow_lo = max(int32_t(PW - fw + SW - 1), 0) / SW,
        ow_hi = min((IW - 1 + PW - fw) / SW + 1, OW),
        oblk_d = od_hi - od_lo,
        oblk_h = oh_hi - oh_lo,
        oblk_w = ow_hi - ow_lo,
        oblk_tot = oblk_d * oblk_h * oblk_w * ((N + BATCH_UNROLL - 1) / BATCH_UNROLL),
        tid = threadIdx.x % nr_thpf;

    if (ID + PD < fd + 1 || od_lo >= od_hi ||
        IH + PH < fh + 1 || oh_lo >= oh_hi ||
        IW + PW < fw + 1 || ow_lo >= ow_hi) {
        if (!tid)
            flt_grad[OUT_IDX] = 0;
        return;
    }

    T sum(0);
    for (uint32_t oblk_idx = tid; oblk_idx < oblk_tot; oblk_idx += nr_thpf) {
        uint32_t n, oh, ow, od;
        n = div_mod(div_mod(div_mod(oblk_idx, oblk_w, ow), oblk_h, oh), oblk_d, od) * BATCH_UNROLL;
        od += od_lo;
        oh += oh_lo;
        ow += ow_lo;
        uint32_t id = od * SD - PD + fd,
                 ih = oh * SH - PH + fh,
                 iw = ow * SW - PW + fw,
                 soff = id * IH * IW + ih * IW + iw + n * SRC_BATCH_STRIDE,
                 doff = od * OH * OW + oh * OW + ow + n * DST_BATCH_STRIDE;
#pragma unroll
        for (uint32_t i = 0; i < BATCH_UNROLL; ++ i) {
            if (!i || n + i < N) {
                sum += src[soff] * dst_grad[doff];
            }
            soff += SRC_BATCH_STRIDE;
            doff += DST_BATCH_STRIDE;
        }
    }

    if (nr_thpf == 1) {
        flt_grad[OUT_IDX] = sum;
    } else {
        // reduce all sums in a block
        extern __shared__ uint8_t shared_storage[];
        volatile T *thread_sum = reinterpret_cast<T*>(shared_storage);
        thread_sum += THREADID_X * nr_thpf;
        thread_sum[tid] = sum;
#pragma unroll
        for (uint32_t i = nr_thpf / 2; i; i >>= 1) {
            bool cond = nr_thpf >= i * 2 && tid < i;
            if (i >= WARP_SIZE) {
                __syncthreads();
            }
            T v0 = thread_sum[tid],
            v1 = v0 + thread_sum[tid + i];
            thread_sum[tid] = cond ? v1 : v0;
        }

        if (!tid)
            flt_grad[OUT_IDX] = thread_sum[0];
    }
}

} // anonymous namespace

template<typename T>
void convolution3d::chanwise::run_bwd_filter(
        T *filter_grad, const T *src, const T *dst_grad,
        const Param &param, cudaStream_t stream) {
    void (*kern)(T*, const T*, const T*, Param) = NULL;
    uint32_t
        nr_thread = query_blocksize_for_kernel(kern_bwd_filter<T, 1024>),
        nr_thpf = std::min(nr_thread,
                std::max<uint32_t>(
                    1,
                    param.out_d * param.out_h * param.out_w * param.batch /
                    (BATCH_UNROLL * 16)));

    // find nearest power-of-2 of nr_thpf
    do {
#define CK(_n) \
        if(nr_thpf >= _n) { \
            kern = kern_bwd_filter<T, _n>; \
            nr_thpf = _n; \
            break; \
        }
        CK(1<<10);
        CK(1<<9);
        CK(1<<8);
        CK(1<<7);
        CK(1<<6);
        CK(1<<5);
        CK(1<<4);
        CK(1<<3);
        CK(1<<2);
        CK(1<<1);
        CK(1<<0);
#undef CK
    } while(0);

    megdnn_assert(kern);
    nr_thread = query_blocksize_for_kernel(kern);

    uint32_t nr_flt_per_blk = nr_thread / nr_thpf;
    while (nr_flt_per_blk * nr_thpf % WARP_SIZE)
        -- nr_flt_per_blk;
    megdnn_assert(nr_flt_per_blk);

    int nr_block = DIVUP(
            param.flt_d * param.flt_h * param.flt_w * 
            param.src_chl * param.chl_mul,
            nr_flt_per_blk);
    nr_thread = nr_flt_per_blk * nr_thpf;
    uint32_t shared = nr_thread * 2 * sizeof(T);
    kern <<< nr_block, nr_thread, shared, stream >>> (
            filter_grad, src, dst_grad, param);
    after_kernel_launch();
}

namespace megdnn {
namespace cuda {
namespace convolution3d {
namespace chanwise {

#define DO_INST(_ct) template void run_bwd_filter( \
        _ct*, const _ct*, const _ct*, const Param&, cudaStream_t);
#define INST(_dt) DO_INST(DTypeTrait<_dt>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(INST)

#undef INST
#undef DO_INST

} // namespace chanwise
} // namespace convolution3d
} // namespace cuda
} // namespace megdnn


// vim: syntax=cuda.doxygen

