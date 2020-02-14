/**
 * \file dnn/src/cuda/convolution3d/chanwise/bwd_data.cu
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

using namespace megdnn;
using namespace cuda;
using namespace convolution3d;
using namespace chanwise;

namespace {

template<typename T, int CHL_MUL_SET,
    int FD_SET, int FH_SET, int FW_SET, 
    int SD_SET, int SH_SET, int SW_SET>
__global__ void kern_bwd_data(
        T *src_grad, const T *dst_grad, const T *flt_tot, Param param) {

    extern __shared__ uint8_t flt_storage[];

    T * const flt = reinterpret_cast<T*>(flt_storage);

    const uint32_t
        N = param.batch, IC = param.src_chl, ic = blockIdx.x,
        ID = param.src_d, IH = param.src_h, IW = param.src_w,
        CHL_MUL = CHL_MUL_SET ? CHL_MUL_SET : param.chl_mul,
        FD = FD_SET ? FD_SET : param.flt_d,
        FH = FH_SET ? FH_SET : param.flt_h,
        FW = FW_SET ? FW_SET : param.flt_w,
        FSIZE = FD * FH * FW,
        PD = param.pad_d, 
        PH = param.pad_h, 
        PW = param.pad_w,
        SD = SD_SET ? SD_SET : param.stride_d,
        SH = SH_SET ? SH_SET : param.stride_h,
        SW = SW_SET ? SW_SET : param.stride_w,
        OD = param.out_d, 
        OH = param.out_h, 
        OW = param.out_w,
        TOT_OUT = N * ID * IH * IW;

    block_memcpy(flt, flt_tot + ic * FSIZE * CHL_MUL, FSIZE * CHL_MUL);
    dst_grad += ic * CHL_MUL * OD * OH * OW;
    src_grad += ic * ID * IH * IW;

    uint32_t out_idx_ = blockIdx.y * blockDim.x + threadIdx.x,
             nr_out_per_launch = blockDim.x * gridDim.y;
    for (; out_idx_ < TOT_OUT; out_idx_ += nr_out_per_launch) {
        uint32_t out_idx = out_idx_, n, id, ih, iw;
        out_idx = div_mod(out_idx, IW, iw);
        out_idx = div_mod(out_idx, IH, ih);
        out_idx = div_mod(out_idx, ID, id);
        n = out_idx;
        const T *dst_grad_base = dst_grad + n * (IC * CHL_MUL * OD * OH * OW);

        T sum(0);

        uint32_t odmin = max(int32_t(id + PD - FD + SD), 0) / SD,
                 ohmin = max(int32_t(ih + PH - FH + SH), 0) / SH,
                 owmin = max(int32_t(iw + PW - FW + SW), 0) / SW,
                 odmax = min((id + PD) / SD, OD - 1),
                 ohmax = min((ih + PH) / SH, OH - 1),
                 owmax = min((iw + PW) / SW, OW - 1);
        if (SD_SET == 1 && SH_SET == 1 && SW_SET == 1 && 
            FD_SET && FH_SET && FW_SET) {
#pragma unroll
            for (uint32_t dod = 0; dod < FD; ++ dod) {
                uint32_t od = odmin + dod;
                if (od <= odmax) {
                    uint32_t fd = id - od * SD + PD;
#pragma unroll
                    for (uint32_t doh = 0; doh < FH; ++ doh) {
                        uint32_t oh = ohmin + doh;
                        if (oh <= ohmax) {
                            uint32_t fh = ih - oh * SH + PH;
#pragma unroll
                            for (uint32_t dow = 0; dow < FW; ++ dow) {
                                uint32_t ow = owmin + dow;
                                if (ow <= owmax) {
                                    uint32_t fw = iw - ow * SW + PW;
                                    const T *pd = dst_grad_base + 
                                        od * OH * OW + oh * OW + ow;
                                    const T *pf = flt + 
                                        fd * FH * FW + fh * FW + fw;
#pragma unroll
                                    for (uint32_t chl_mul = 0; chl_mul < CHL_MUL;
                                            ++ chl_mul) {
                                        sum += *pd * *pf;
                                        pd += OD * OH * OW;
                                        pf += FSIZE;
                                    }
                                }
                            }
                        }
                    }
                }   
            }
        } else {
            for (uint32_t od = odmin; od <= odmax; ++ od) {
                uint32_t fd = id - od * SD + PD;
                for (uint32_t oh = ohmin; oh <= ohmax; ++ oh) {
                    uint32_t fh = ih - oh * SH + PH;
                    for (uint32_t ow = owmin; ow <= owmax; ++ ow) {
                        uint32_t fw = iw - ow * SW + PW;
                        const T *pd = dst_grad_base + 
                            od * OH * OW + oh * OW + ow;
                        const T *pf = flt + 
                            fd * FH * FW + fh * FW + fw;
#pragma unroll
                        for (uint32_t chl_mul = 0; chl_mul < CHL_MUL; ++ chl_mul) {
                            sum += *pd * *pf;
                            pd += OD * OH * OW;
                            pf += FSIZE;
                        }
                    }
                }
            }
        }
        src_grad[n * IC * ID * IH * IW + 
                id * IH * IW + ih * IW + iw] = sum;
    }
}

template<typename T>
class KernDispatch {
    public:
        typedef void (*kern_ptr_t)(T*, const T*, const T*, Param);

        static kern_ptr_t dispatch(
                int chl_mul, 
                int fd, int fh, int fw, 
                int sd, int sh, int sw) {
            if (chl_mul == 1) {
                if (fd == 2 && fh == 2 && fw == 2)
                    return d1<1, 2, 2, 2>(sd, sh, sw);
                if (fd == 3 && fh == 3 && fw == 3)
                    return d1<1, 3, 3, 3>(sd, sh, sw);
            }
            return d1<0, 0, 0, 0>(sd, sh, sw);
        }

    private:
        template<int chl_mul, int fd, int fh, int fw>
        static kern_ptr_t d1(int sd, int sh, int sw) {
            if (sd == 1 && sh == 1 && sw == 1) 
                return kern_bwd_data<T, chl_mul, fd, fh, fw, 1, 1, 1>;
            if (sd == 1 && sh == 1 && sw == 2) 
                return kern_bwd_data<T, chl_mul, fd, fh, fw, 1, 1, 2>;
            if (sd == 1 && sh == 2 && sw == 1) 
                return kern_bwd_data<T, chl_mul, fd, fh, fw, 1, 2, 1>;
            if (sd == 1 && sh == 2 && sw == 2) 
                return kern_bwd_data<T, chl_mul, fd, fh, fw, 1, 2, 2>;
            if (sd == 2 && sh == 1 && sw == 1) 
                return kern_bwd_data<T, chl_mul, fd, fh, fw, 2, 1, 1>;
            if (sd == 2 && sh == 1 && sw == 2) 
                return kern_bwd_data<T, chl_mul, fd, fh, fw, 2, 1, 2>;
            if (sd == 2 && sh == 2 && sw == 1) 
                return kern_bwd_data<T, chl_mul, fd, fh, fw, 2, 2, 1>;
            if (sd == 2 && sh == 2 && sw == 2) 
                return kern_bwd_data<T, chl_mul, fd, fh, fw, 2, 2, 2>;
            return kern_bwd_data<T, chl_mul, fd, fh, fw, 0, 0, 0>;
        }
};

} // anonymous namespace

template<typename T>
void chanwise::run_bwd_data(T *src_grad, const T *dst_grad, const T *flt,
        const Param &param, cudaStream_t stream) {
    typename KernDispatch<T>::kern_ptr_t kern = KernDispatch<T>::dispatch(
            param.chl_mul, 
            param.flt_d, param.flt_h, param.flt_w,
            param.stride_d, param.stride_h, param.stride_w);
    int nr_thread = query_blocksize_for_kernel(kern),
        nr_out_dimx = param.src_d * param.src_h * param.src_w * param.batch;
    dim3 nr_block(
            param.src_chl,
            std::min(512, max(nr_out_dimx / (nr_thread * 4), 1)));
    uint32_t shared = param.chl_mul * param.flt_d * 
        param.flt_h * param.flt_w * sizeof(T);
    kern <<< nr_block, nr_thread, shared, stream >>> (
            src_grad, dst_grad, flt, param);
    after_kernel_launch();
}

namespace megdnn {
namespace cuda {
namespace convolution3d {
namespace chanwise {

#define DO_INST(_ct) template void run_bwd_data( \
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

