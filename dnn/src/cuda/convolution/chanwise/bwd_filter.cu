/**
 * \file dnn/src/cuda/convolution/chanwise/bwd_filter.cu
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
#include "src/cuda/cub/util_ptx.cuh"
#include "cuda_fp16.h"
#include "src/cuda/fp16_help.cuh"

const uint32_t WARP_SIZE = 32, BATCH_UNROLL = 4;

using namespace megdnn;
using namespace cuda;
using namespace convolution;
using namespace chanwise;

namespace {

/*!
 * \brief compute grad w.r.t. filter
 *
 * block dim: out_id * kern_id
 * threads with the same out_id computes grad for corresponding kernel element
 * \tparam nr_thpf number of threads for one element in the filter; must be
 *      power of 2;
 */
template<typename T, uint32_t nr_thpf>
__global__ void kern_bwd_filter_float(
        T* flt_grad, const T* src, const T* dst_grad, Param param) {

    const uint32_t
        N = param.batch, IC = param.src_chl, IH = param.src_h, IW = param.src_w,
        CHL_MUL = param.chl_mul,
        FH = param.flt_h, FW = param.flt_w,
        PH = param.pad_h, PW = param.pad_w,
        SH = param.stride_h, SW = param.stride_w,
        OH = param.out_h, OW = param.out_w,
        SRC_BATCH_STRIDE = IC * IH * IW,
        DST_BATCH_STRIDE = IC * CHL_MUL * OH * OW,
        BLKDIM_X = blockDim.x / nr_thpf,
        THREADID_X = threadIdx.x / nr_thpf,
        OUT_IDX = blockIdx.x * BLKDIM_X + THREADID_X;

    uint32_t ic, chl_mul, fh, fw;
    {
        uint32_t i = OUT_IDX;
        i = div_mod(i, FW, fw);
        i = div_mod(i, FH, fh);
        i = div_mod(i, CHL_MUL, chl_mul);
        ic = i;
    }
    if (ic >= IC) {
        return;
    }
    src += ic * IH * IW;
    dst_grad += (ic * CHL_MUL + chl_mul) * OH * OW;

    const uint32_t
        oh_lo = max(int32_t(PH - fh + SH - 1), 0) / SH,
        oh_hi = min((IH - 1 + PH - fh) / SH + 1, OH),
        ow_lo = max(int32_t(PW - fw + SW - 1), 0) / SW,
        ow_hi = min((IW - 1 + PW - fw) / SW + 1, OW),
        oblk_h = oh_hi - oh_lo,
        oblk_w = ow_hi - ow_lo,
        oblk_tot = oblk_h * oblk_w * ((N + BATCH_UNROLL - 1) / BATCH_UNROLL),
        tid = threadIdx.x % nr_thpf;

    if (IH + PH < fh + 1 || oh_lo >= oh_hi ||
            IW + PW < fw + 1 || ow_lo >= ow_hi) {
        if (!tid)
            flt_grad[OUT_IDX] = 0;
        return;
    }

    T sum(0);
    for (uint32_t oblk_idx = tid; oblk_idx < oblk_tot; oblk_idx += nr_thpf) {
        uint32_t n, oh, ow;
        n = div_mod(div_mod(oblk_idx, oblk_w, ow), oblk_h, oh) * BATCH_UNROLL;
        oh += oh_lo;
        ow += ow_lo;
        uint32_t ih = oh * SH - PH + fh,
                 iw = ow * SW - PW + fw,
                 soff = ih * IW + iw + n * SRC_BATCH_STRIDE,
                 doff = oh * OW + ow + n * DST_BATCH_STRIDE;
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
        volatile T* thread_sum = reinterpret_cast<T*>(shared_storage);
        thread_sum += THREADID_X * nr_thpf;
        thread_sum[tid] = sum;
#pragma unroll
        for (uint32_t i = nr_thpf / 2; i; i >>= 1) {
            bool cond = nr_thpf >= i * 2 && tid < i;
            if (i >= WARP_SIZE) {
                __syncthreads();
            } else {
                cub::WARP_SYNC(0xffffffff);
            }
            if (cond) {
                T v0 = thread_sum[tid], v1 = v0 + thread_sum[tid + i];
                thread_sum[tid] = v1;
            }
        }

        if (!tid) {
            flt_grad[OUT_IDX] = thread_sum[0];
        }
    }
}

#if CUDA_VERSION >= 9000
template<typename T, uint32_t nr_thpf>
__global__ void kern_bwd_filter_hf(
		__half* flt_grad, const __half* src, const __half* dst_grad, Param param) {
	const uint32_t
		N = param.batch, IC = param.src_chl, IH = param.src_h, IW = param.src_w,
		CHL_MUL = param.chl_mul,
		FH = param.flt_h, FW = param.flt_w,
		PH = param.pad_h, PW = param.pad_w,
		SH = param.stride_h, SW = param.stride_w,
		OH = param.out_h, OW = param.out_w,
		SRC_BATCH_STRIDE = IC * IH * IW,
		DST_BATCH_STRIDE = IC * CHL_MUL * OH * OW,
		BLKDIM_X = (blockDim.x / nr_thpf) * 2,
		THREADID_X = (threadIdx.x / nr_thpf) * 2,
		OUT_IDX = blockIdx.x * BLKDIM_X + THREADID_X,
        LAST_IDX = FH * FW * CHL_MUL * IC,
        tid = threadIdx.x % nr_thpf;
    __half2 sum2{0.0, 0.0};

	if (OUT_IDX % FW != FW - 1) {
		uint32_t ic, chl_mul, fh, fw;
		{
			uint32_t i = OUT_IDX;
			i = div_mod(i, FW, fw);
			i = div_mod(i, FH, fh);
			i = div_mod(i, CHL_MUL, chl_mul);
			ic = i;
		}
		if (ic >= IC) {
			return;
		}
		src += ic * IH * IW;
		dst_grad += (ic * CHL_MUL + chl_mul) * OH * OW;

		const uint32_t
			oh_lo = max(int32_t(PH - fh + SH - 1), 0) / SH,
			oh_hi = min((IH - 1 + PH - fh) / SH + 1, OH),
			ow_lox = max(int32_t(PW - fw + SW - 1), 0) / SW,
			ow_loy = max(int32_t(PW - fw + SW - 2), 0) / SW,
			ow_hix = min((IW - 1 + PW - fw) / SW + 1, OW),
			ow_hiy = min((IW - 2 + PW - fw) / SW + 1, OW),
			oblk_h = oh_hi - oh_lo,
			oblk_wx = ow_hix - ow_lox,
			oblk_wy = ow_hiy - ow_loy;
        if (IH + PH < fh + 1 || oh_lo >= oh_hi || IW + PW < fw + 1) {
            if (!tid) {
                flt_grad[OUT_IDX] = 0;
                flt_grad[OUT_IDX + 1] = 0;
            }
            return;
        }
	
		if (ow_lox >= ow_hix) {
			if (!tid)
				flt_grad[OUT_IDX] = 0;
		}

		if (IW + PW < fw + 2 || ow_loy >= ow_hiy) {
			if (!tid)
				flt_grad[OUT_IDX + 1] = 0;
            if (ow_lox >= ow_hix)
                return;
		}

		sum2.x = 0.0;
		sum2.y = 0.0;
		__half2 src2{0.0, 0.0};
		__half2 dst2{0.0, 0.0};

		const uint32_t
			oblk_w = max(ow_hix, ow_hiy) - min(ow_lox, ow_loy),
			oblk_tot = oblk_h * oblk_w * ((N + BATCH_UNROLL - 1) / BATCH_UNROLL);

		for (uint32_t oblk_idx = tid; oblk_idx < oblk_tot; oblk_idx += nr_thpf) {
			uint32_t n_x, n_y, oh, ow_x, ow_y;
			n_x = div_mod(div_mod(oblk_idx, oblk_wx, ow_x), oblk_h, oh) * BATCH_UNROLL;
			n_y = div_mod(div_mod(oblk_idx, oblk_wy, ow_y), oblk_h, oh) * BATCH_UNROLL;
			oh += oh_lo;
			ow_x += ow_lox;
			ow_y += ow_loy;
			uint32_t ih = oh * SH - PH + fh,
					 iw_x = ow_x * SW - PW + fw,
					 iw_y = ow_y * SW - PW + fw + 1,
					 soff_x = ih * IW + iw_x + n_x * SRC_BATCH_STRIDE,
					 soff_y = ih * IW + iw_y + n_y * SRC_BATCH_STRIDE,
					 doff_x = oh * OW + ow_x + n_x * DST_BATCH_STRIDE,
					 doff_y = oh * OW + ow_y + n_y * DST_BATCH_STRIDE;
#pragma unroll
			for (uint32_t i = 0; i < BATCH_UNROLL; ++ i) {
				if (!i || n_x + i < N || n_y + i < N) {
					src2.x = 0.0;
					src2.y = 0.0;
					dst2.x = 0.0;
					dst2.y = 0.0;
					if (n_x + i < N && ow_x < ow_hix) {
						src2.x = src[soff_x];
						dst2.x = dst_grad[doff_x];
					}
					if (n_y + i < N && ow_y < ow_hiy) {
						src2.y = src[soff_y];
						dst2.y = dst_grad[doff_y];
					}
					sum2 = fma2(src2, dst2, sum2);
				}
				soff_x += SRC_BATCH_STRIDE;
				soff_y += SRC_BATCH_STRIDE;
				doff_x += DST_BATCH_STRIDE;
				doff_y += DST_BATCH_STRIDE;
			}
		}
	} else {
		for (size_t offset = 0; offset < 2; ++ offset) {
			uint32_t ic, chl_mul, fh, fw;
			{
				uint32_t i = OUT_IDX + offset;
				i = div_mod(i, FW, fw);
				i = div_mod(i, FH, fh);
				i = div_mod(i, CHL_MUL, chl_mul);
				ic = i;
			}
			if (ic >= IC) {
				if (offset == 0)
                    return;
                else
                    break;
			}
			const uint32_t
				oh_lo = max(int32_t(PH - fh + SH - 1), 0) / SH,
				oh_hi = min((IH - 1 + PH - fh) / SH + 1, OH),
				ow_lo = max(int32_t(PW - fw + SW - 1), 0) / SW,
				ow_hi = min((IW - 1 + PW - fw) / SW + 1, OW),
				oblk_h = oh_hi - oh_lo,
				oblk_w = ow_hi - ow_lo,
				oblk_tot = oblk_h * oblk_w * ((N + BATCH_UNROLL - 1) / BATCH_UNROLL);

			if (IH + PH < fh + 1 || oh_lo >= oh_hi ||
					IW + PW < fw + 1 || ow_lo >= ow_hi) {
				if (!tid)
					flt_grad[OUT_IDX + offset] = 0;
				continue;
			}

			__half sum(0.0);

			for (uint32_t oblk_idx = tid; oblk_idx < oblk_tot; oblk_idx += nr_thpf) {
				uint32_t n, oh, ow;
				n = div_mod(div_mod(oblk_idx, oblk_w, ow), oblk_h, oh) * BATCH_UNROLL;
				oh += oh_lo;
				ow += ow_lo;
				uint32_t ih = oh * SH - PH + fh,
						 iw = ow * SW - PW + fw,
						 soff = ic * IH * IW + ih * IW + iw + n * SRC_BATCH_STRIDE,
						 doff = (ic * CHL_MUL + chl_mul) * OH * OW + oh * OW + ow + n * DST_BATCH_STRIDE;
#pragma unroll
				for (uint32_t i = 0; i < BATCH_UNROLL; ++ i) {
					if (!i || n + i < N) {
						sum = fma(src[soff], dst_grad[doff], sum);
					}
					soff += SRC_BATCH_STRIDE;
					doff += DST_BATCH_STRIDE;
				}
			}
            if (!offset)
                sum2.x = sum;
            if (offset)
                sum2.y = sum;
		}
	}

    if (nr_thpf == 1) {
        flt_grad[OUT_IDX] = sum2.x;
        if (OUT_IDX != LAST_IDX)
            flt_grad[OUT_IDX + 1] = sum2.y;
    } else {
        extern __shared__ uint8_t shared_storage[];
        __half2* thread_sum = reinterpret_cast<__half2*>(shared_storage);
        thread_sum += THREADID_X * nr_thpf / 2;
        thread_sum[tid] = sum2;
#pragma unroll
        for (uint32_t i = nr_thpf / 2; i; i >>= 1) {
            bool cond = nr_thpf >= i * 2 && tid < i;
            if (i >= WARP_SIZE) {
                __syncthreads();
            } else {
                cub::WARP_SYNC(0xffffffff);
            }
            if (cond) {
                __half2 one = {1.0, 1.0};
                __half2 v0 = thread_sum[tid], v1 = fma2(v0, one, thread_sum[tid + i]);
                thread_sum[tid] = v1;
            }
        }

        if (!tid) {
            flt_grad[OUT_IDX] = thread_sum[0].x;
            if (OUT_IDX != LAST_IDX)
                flt_grad[OUT_IDX + 1] = thread_sum[0].y;
        }
    }
}
#endif

#define GET_KERN(func, type)                                    \
    FixFunction<type> f_struct;                                 \
    switch (_p) {                                               \
        case 1 << 10:                                           \
            f_struct.f = func<type, 1 << 10>;                   \
            break;                                              \
        case 1 << 9:                                            \
            f_struct.f = func<type, 1 << 9>;                    \
            break;                                              \
        case 1 << 8:                                            \
            f_struct.f = func<type, 1 << 8>;                    \
            break;                                              \
        case 1 << 7:                                            \
            f_struct.f = func<type, 1 << 7>;                    \
            break;                                              \
        case 1 << 6:                                            \
            f_struct.f = func<type, 1 << 6>;                    \
            break;                                              \
        case 1 << 5:                                            \
            f_struct.f = func<type, 1 << 5>;                    \
            break;                                              \
        case 1 << 4:                                            \
            f_struct.f = func<type, 1 << 4>;                    \
            break;                                              \
        case 1 << 3:                                            \
            f_struct.f = func<type, 1 << 3>;                    \
            break;                                              \
        case 1 << 2:                                            \
            f_struct.f = func<type, 1 << 2>;                    \
            break;                                              \
        case 1 << 1:                                            \
            f_struct.f = func<type, 1 << 1>;                    \
            break;                                              \
        case 1 << 0:                                            \
            f_struct.f = func<type, 1 << 0>;                    \
            break;                                              \
        default:                                                \
            megdnn_assert(false, "DO NOT IMP CASE FUNCTION!!"); \
    }                                                           \
    return f_struct;

template <typename T>
struct FixFunction {
    void (*f)(T*, const T*, const T*, Param);
};

template <typename T>
FixFunction<T> get_kern(const uint32_t& _p);

template <>
FixFunction<float> get_kern<float>(const uint32_t& _p) {
    GET_KERN(kern_bwd_filter_float, float);
}

#if CUDA_VERSION >= 9000
template <>
FixFunction<__half> get_kern<__half>(const uint32_t& _p) {
    GET_KERN(kern_bwd_filter_hf, __half);
}
#endif

template <>
FixFunction<dt_float16> get_kern<dt_float16>(const uint32_t& _p) {
    GET_KERN(kern_bwd_filter_float, dt_float16);
}

#undef GET_KERN
}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace convolution {
namespace chanwise {
template <typename T>
void run_bwd_filter(T *filter_grad, const T *src, const T *dst_grad,
		const Param &param, cudaStream_t stream) {
	void (*kern)(T*, const T*, const T*, Param) = NULL;
	uint32_t                                           
		nr_thread = query_blocksize_for_kernel(get_kern<T>(1024).f),
		nr_thpf = std::min(nr_thread,                  
        	std::max<uint32_t>(                    
				1,                                 
				param.out_h * param.out_w * param.batch /
				(BATCH_UNROLL * 16)));
	// find nearest power-of-2 of nr_thpf
	do {
#define CK(_n) \
		if (nr_thpf >= _n) { \
			kern = get_kern<T>(_n).f; \
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
		--nr_flt_per_blk;
	megdnn_assert(nr_flt_per_blk);

	int nr_block = DIVUP(
		param.flt_h * param.flt_w * param.src_chl * param.chl_mul,
		nr_flt_per_blk);
	nr_thread = nr_flt_per_blk * nr_thpf;
	uint32_t shared = nr_thread * 2 * sizeof(T);
	kern <<< nr_block, nr_thread, shared, stream >>> (
		filter_grad, src, dst_grad, param);
	after_kernel_launch();
}

template void run_bwd_filter(float*, const float*, const float*, const Param&,
                             cudaStream_t);

#if CUDA_VERSION >= 9000
template void run_bwd_filter(__half*, const __half*, const __half*, const Param&,
                             cudaStream_t);
#endif

template void run_bwd_filter(dt_float16*, const dt_float16*, const dt_float16*,
                             const Param&, cudaStream_t);

} // namespace chanwise
} // namespace convolution
} // namespace cuda
} // namespace megdnn


// vim: syntax=cuda.doxygen

