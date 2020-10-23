/**
 * \file dnn/src/cuda/dct/dct_channel_select.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megcore_cdefs.h"
#include "src/cuda/dct/dct_channel_select.cuh"
#include "src/cuda/error_info.cuh"

namespace megdnn {
namespace cuda {

template <typename T>
struct CudaPostProcess;

template <>
struct CudaPostProcess<float> {
    CudaPostProcess(float){};
    static inline __device__ float func(float val) { return val; }
};

template <>
struct CudaPostProcess<int8_t> {
    CudaDTypeParamImpl<dt_qint8> m_type_cvt;
    CudaPostProcess(float scale) { m_type_cvt.inv_scale = 1.f / scale; };
    inline __device__ int8_t func(float val) {
        return m_type_cvt.quantize(val).as_int8();
    }
};

template <uint32_t format>
struct ChannelBlockHelper;
template <>
struct ChannelBlockHelper<dct::DctLayoutFormat::NCHW4> {
    static constexpr int channel_block = 4;
};
template <>
struct ChannelBlockHelper<dct::DctLayoutFormat::NCHW> {
    static constexpr int channel_block = 1;
};

namespace dct {
namespace {
inline __device__ void load_row(float (&row_cache)[8], const uint8_t* src) {
    int2 row = *((int2*)src);
    row_cache[0] = (float)(((uchar4*)&(row.x))->x);
    row_cache[1] = (float)(((uchar4*)&(row.x))->y);
    row_cache[2] = (float)(((uchar4*)&(row.x))->z);
    row_cache[3] = (float)(((uchar4*)&(row.x))->w);
    row_cache[4] = (float)(((uchar4*)&(row.y))->x);
    row_cache[5] = (float)(((uchar4*)&(row.y))->y);
    row_cache[6] = (float)(((uchar4*)&(row.y))->z);
    row_cache[7] = (float)(((uchar4*)&(row.y))->w);
}

inline __device__ void fast_dct_1d_internel(float& src0, float& src1,
                                            float& src2, float& src3,
                                            float& src4, float& src5,
                                            float& src6, float& src7) {
    constexpr float rsqrt_8 = 0.3535533905932737f;  //!< rsqrt_8 = sqrt(1 / 8)
    constexpr float a = 1.387039845322148f;  //!< a = sqrt2 * cos(pi * 1 / 16)
    constexpr float b = 1.306562964876377f;  //!< b = sqrt2 * cos(pi * 2 / 16)
    constexpr float c = 1.175875602419359f;  //!< c = sqrt2 * cos(pi * 3 / 16)
    constexpr float d = 0.785694958387102f;  //!< d = sqrt2 * cos(pi * 5 / 16)
    constexpr float e = 0.541196100146197f;  //!< e = sqrt2 * cos(pi * 6 / 16)
    constexpr float f = 0.275899379282943f;  //!< f = sqrt2 * cos(pi * 7 / 16)

    const float add_0_7 = src0 + src7;
    const float add_1_6 = src1 + src6;
    const float add_2_5 = src2 + src5;
    const float add_3_4 = src3 + src4;
    const float sub_0_7 = src0 - src7;
    const float sub_6_1 = src6 - src1;
    const float sub_2_5 = src2 - src5;
    const float sub_4_3 = src4 - src3;

    const float add_0_7_3_4 = add_0_7 + add_3_4;
    const float add_1_6_2_5 = add_1_6 + add_2_5;
    const float add_0_7_sub_3_4 = add_0_7 - add_3_4;
    const float add_1_6_sub_2_5 = add_1_6 - add_2_5;

    src0 = rsqrt_8 * (add_0_7_3_4 + add_1_6_2_5);
    src2 = rsqrt_8 * (b * add_0_7_sub_3_4 + e * add_1_6_sub_2_5);
    src4 = rsqrt_8 * (add_0_7_3_4 - add_1_6_2_5);
    src6 = rsqrt_8 * (e * add_0_7_sub_3_4 - b * add_1_6_sub_2_5);

    src1 = rsqrt_8 * (a * sub_0_7 - c * sub_6_1 + d * sub_2_5 - f * sub_4_3);
    src3 = rsqrt_8 * (c * sub_0_7 + f * sub_6_1 - a * sub_2_5 + d * sub_4_3);
    src5 = rsqrt_8 * (d * sub_0_7 + a * sub_6_1 + f * sub_2_5 - c * sub_4_3);
    src7 = rsqrt_8 * (f * sub_0_7 + d * sub_6_1 + c * sub_2_5 + a * sub_4_3);
}

inline __device__ void fast_dct_1d(float (&src)[8]) {
    fast_dct_1d_internel(src[0], src[1], src[2], src[3], src[4], src[5], src[6],
                         src[7]);
}

inline __device__ void fast_dct_1d_col(float (&src)[8][8], const int col) {
    fast_dct_1d_internel(src[0][col], src[1][col], src[2][col], src[3][col],
                         src[4][col], src[5][col], src[6][col], src[7][col]);
}
enum class MaskType {
    NO_MASK = 0,
    USER_DEFINE_MASK = 1,
    FIX_32_MASK = 2,
    MASK_END
};
template <const int dct_block, const int block_oh, const int block_ow,
          uint32_t format, MaskType mask_type, typename DstDtype, typename T2>
struct StoreMask;

template <const int dct_block, const int block_oh, const int block_ow,
          typename T2>
struct StoreMask<dct_block, block_oh, block_ow, DctLayoutFormat::NCHW,
                 MaskType::USER_DEFINE_MASK, float, T2> {
    static inline __device__ void func(
            const float (&thread_cache)[dct_block][dct_block], float* dst_tid,
            const int oc_stride, int channel_idx, const int* mask_offset,
            const int* mask_val, CudaPostProcess<T2>& quant_param,
            megcore::AsyncErrorInfo* error_info, void* error_tracker) {
        __shared__ float shared[dct_block][dct_block][block_oh][block_ow];
#pragma unroll
        for (int i = 0; i < dct_block; ++i)
#pragma unroll
            for (int j = 0; j < dct_block; ++j) {
                shared[i][j][threadIdx.y][threadIdx.x] = thread_cache[i][j];
            }
        const int store_channel_offset = mask_offset[channel_idx];
        const int nr_store_channel =
                mask_offset[channel_idx + 1] - store_channel_offset;
        if (nr_store_channel < 0) {
            set_async_error_info(error_info, error_tracker,
                                 "nchw sub mask len must > 0");
        }
        for (int store_channel_idx = 0; store_channel_idx < nr_store_channel;
             ++store_channel_idx) {
            const int index =
                    mask_val[store_channel_offset + store_channel_idx];
            dst_tid[store_channel_idx * oc_stride] =
                    shared[index / dct_block][index % dct_block][threadIdx.y]
                          [threadIdx.x];
        }
    }
};

template <const int dct_block, const int block_oh, const int block_ow,
          typename T2>
struct StoreMask<dct_block, block_oh, block_ow, DctLayoutFormat::NCHW4,
                 MaskType::USER_DEFINE_MASK, int8_t, T2> {
    static inline __device__ void func(
            const float (&thread_cache)[dct_block][dct_block], int8_t* dst_tid,
            const int oc_stride, int channel_idx, const int* mask_offset,
            const int* mask_val, CudaPostProcess<T2>& quant_param,
            megcore::AsyncErrorInfo* error_info, void* error_tracker) {
        //! nchw4 channel_block is 4
        constexpr int channel_block =
                ChannelBlockHelper<DctLayoutFormat::NCHW4>::channel_block;
        __shared__ float shared[dct_block][dct_block][block_oh][block_ow];
#pragma unroll
        for (int i = 0; i < dct_block; ++i)
#pragma unroll
            for (int j = 0; j < dct_block; ++j) {
                shared[i][j][threadIdx.y][threadIdx.x] = thread_cache[i][j];
            }
        const int store_channel_offset = mask_offset[channel_idx];
        const int nr_store_channel =
                mask_offset[channel_idx + 1] - store_channel_offset;
        if (nr_store_channel % 4 != 0 || nr_store_channel < 0) {
            set_async_error_info(error_info, error_tracker,
                                 "nchw4 sub_mask_len mod 4 should be 0 and "
                                 "sub_mask_len must > 0");
        }
        for (int store_channel_idx = 0; store_channel_idx < nr_store_channel;
             store_channel_idx += channel_block) {
            const int index0 =
                    mask_val[store_channel_offset + store_channel_idx];
            const int index1 =
                    mask_val[store_channel_offset + store_channel_idx + 1];
            const int index2 =
                    mask_val[store_channel_offset + store_channel_idx + 2];
            const int index3 =
                    mask_val[store_channel_offset + store_channel_idx + 3];
            const int store_c4_idx = store_channel_idx / channel_block;
            *(char4*)(&dst_tid[store_c4_idx * channel_block * oc_stride]) = {
                    quant_param.func(
                            shared[index0 / dct_block][index0 % dct_block]
                                  [threadIdx.y][threadIdx.x]),
                    quant_param.func(
                            shared[index1 / dct_block][index1 % dct_block]
                                  [threadIdx.y][threadIdx.x]),
                    quant_param.func(
                            shared[index2 / dct_block][index2 % dct_block]
                                  [threadIdx.y][threadIdx.x]),
                    quant_param.func(
                            shared[index3 / dct_block][index3 % dct_block]
                                  [threadIdx.y][threadIdx.x])};
        }
    }
};

template <const int dct_block, const int block_oh, const int block_ow,
          uint32_t format, typename DstDtype, typename T2>
struct StoreMask<dct_block, block_oh, block_ow, format, MaskType::NO_MASK,
                 DstDtype, T2> {
    static inline __device__ void func(
            const float (&thread_cache)[dct_block][dct_block],
            DstDtype* dst_tid, const int oc_stride, int channel_idx,
            const int* mask_offset, const int* mask_val,
            CudaPostProcess<T2>& quant_param,
            megcore::AsyncErrorInfo* error_info, void* error_tracker) {
        constexpr int channel_block = ChannelBlockHelper<format>::channel_block;
#pragma unroll
        for (int i = 0; i < dct_block; i++) {
#pragma unroll
            for (int j = 0; j < dct_block; j++) {
                dst_tid[(i * dct_block + j) / channel_block * channel_block *
                                oc_stride +
                        (i * dct_block + j) % channel_block] =
                        quant_param.func(thread_cache[i][j]);
            }
        }
    }
};

template <const int dct_block, const int block_oh, const int block_ow,
          typename T2>
struct StoreMask<dct_block, block_oh, block_ow, DctLayoutFormat::NCHW,
                 MaskType::FIX_32_MASK, float, T2> {
    static inline __device__ void func(
            const float (&thread_cache)[dct_block][dct_block], float* dst_tid,
            const int oc_stride, int channel_idx, const int* mask_offset,
            const int* mask_val, CudaPostProcess<T2>& quant_param,
            megcore::AsyncErrorInfo* error_info, void* error_tracker) {
#define STORE(store_index, index)      \
    dst_tid[store_index * oc_stride] = \
            thread_cache[index / dct_block][index % dct_block]

        STORE(0, 0);
        STORE(1, 1);
        STORE(2, 8);
        STORE(3, 16);
        STORE(4, 9);
        STORE(5, 2);
        STORE(6, 3);
        STORE(7, 10);

        if (channel_idx == 0) {
            STORE(8, 17);
            STORE(9, 24);
            STORE(10, 32);
            STORE(11, 25);
            STORE(12, 18);
            STORE(13, 11);
            STORE(14, 4);
            STORE(15, 5);
        }
#undef STORE
    }
};

template <const int dct_block, const int block_oh, const int block_ow,
          typename T2>
struct StoreMask<dct_block, block_oh, block_ow, DctLayoutFormat::NCHW4,
                 MaskType::FIX_32_MASK, int8_t, T2> {
    static inline __device__ void func(
            const float (&thread_cache)[dct_block][dct_block], int8_t* dst_tid,
            const int oc_stride, int channel_idx, const int* mask_offset,
            const int* mask_val, CudaPostProcess<T2>& quant_param,
            megcore::AsyncErrorInfo* error_info, void* error_tracker) {
#define STORE(store_index, index0, index1, index2, index3)                 \
    *(char4*)(&dst_tid[store_index * oc_stride]) = {                       \
            quant_param.func(                                              \
                    thread_cache[index0 / dct_block][index0 % dct_block]), \
            quant_param.func(                                              \
                    thread_cache[index1 / dct_block][index1 % dct_block]), \
            quant_param.func(                                              \
                    thread_cache[index2 / dct_block][index2 % dct_block]), \
            quant_param.func(                                              \
                    thread_cache[index3 / dct_block][index3 % dct_block])}

        STORE(0, 0, 1, 8, 16);
        STORE(4, 9, 2, 3, 10);
        if (channel_idx == 0) {
            STORE(8, 17, 24, 32, 25);
            STORE(12, 18, 11, 4, 5);
        }
#undef STORE
    }
};

template <const int dct_block, MaskType mask_type, const int ker_block_h,
          const int ker_block_w, uint32_t format, typename DstDtype,
          typename T2>
__global__ void kern_dct(const uint8_t* src, DstDtype* dst, const int n,
                         const int c, const int h, const int w, const int oh,
                         const int ow, const int oc_stride, const int oc,
                         const int* mask_offset, const int* mask_val,
                         CudaPostProcess<T2> quant_param,
                         megcore::AsyncErrorInfo* error_info,
                         void* error_tracker) {
    constexpr int block_oh = ker_block_h / dct_block;
    constexpr int block_ow = ker_block_w / dct_block;
    const int channel_stride = h * w;
    const int oc_idx = blockIdx.z % c;
    const int oh_idx = blockIdx.y * block_oh + threadIdx.y;
    const int ow_idx = blockIdx.x * block_ow + threadIdx.x;
    float thread_cache[dct_block][dct_block];
    const uint8_t* src_tid =
            src + blockIdx.z * channel_stride +
            (blockIdx.y * ker_block_h + threadIdx.y * dct_block) * w +
            (blockIdx.x * ker_block_w + threadIdx.x * dct_block);
    const int inner_channel_offset =
            (oh_idx * ow + ow_idx) * ChannelBlockHelper<format>::channel_block;

    DstDtype* dst_tid =
            dst + blockIdx.z * channel_stride + inner_channel_offset;
    if (mask_type != MaskType::NO_MASK) {
        const int batch_idx = blockIdx.z / c;
        const int batch_stride = oc_stride * oc;
        int out_channel_offset = 0;
        if (mask_type == MaskType::FIX_32_MASK) {
            //! trick out_channel_offset = {0, 16, 24}[oc_idx]; oc_idx = 0, 1, 2
            out_channel_offset = 16 * oc_idx - 8 * (oc_idx >> 1);
        } else {
            out_channel_offset = mask_offset[oc_idx];
        }
        dst_tid = dst + batch_idx * batch_stride +
                  out_channel_offset * oc_stride + inner_channel_offset;
    }

    if (oh_idx < oh && ow_idx < ow) {
        load_row(thread_cache[0], src_tid + 0 * w);
        load_row(thread_cache[1], src_tid + 1 * w);
        load_row(thread_cache[2], src_tid + 2 * w);
        load_row(thread_cache[3], src_tid + 3 * w);
        load_row(thread_cache[4], src_tid + 4 * w);
        load_row(thread_cache[5], src_tid + 5 * w);
        load_row(thread_cache[6], src_tid + 6 * w);
        load_row(thread_cache[7], src_tid + 7 * w);

        //! TMP = A @ C.T
        fast_dct_1d(thread_cache[0]);
        fast_dct_1d(thread_cache[1]);
        fast_dct_1d(thread_cache[2]);
        fast_dct_1d(thread_cache[3]);
        fast_dct_1d(thread_cache[4]);
        fast_dct_1d(thread_cache[5]);
        fast_dct_1d(thread_cache[6]);
        fast_dct_1d(thread_cache[7]);

        //! TMP = C @ TMP
        fast_dct_1d_col(thread_cache, 0);
        fast_dct_1d_col(thread_cache, 1);
        fast_dct_1d_col(thread_cache, 2);
        fast_dct_1d_col(thread_cache, 3);
        fast_dct_1d_col(thread_cache, 4);
        fast_dct_1d_col(thread_cache, 5);
        fast_dct_1d_col(thread_cache, 6);
        fast_dct_1d_col(thread_cache, 7);

        StoreMask<dct_block, block_oh, block_ow, format, mask_type, DstDtype,
                  T2>::func(thread_cache, dst_tid, oc_stride, oc_idx,
                            mask_offset, mask_val, quant_param, error_info,
                            error_tracker);
    }
}

}  // namespace

template <int dct_block, uint32_t format, typename DstDtype>
void call_kern_dct(const uint8_t* d_src, DstDtype* d_dst, const int n,
                   const int c, const int h, const int w, const int oc,
                   bool fix_32_mask, const int* mask_offset,
                   const int* mask_val, cudaStream_t stream,
                   megcore::AsyncErrorInfo* error_info, void* error_tracker,
                   float scale) {
    constexpr int ker_block_h = 32;
    constexpr int ker_block_w = 256;
    const int oh = h / dct_block;
    const int ow = w / dct_block;
    const int oc_stride = oh * ow;
    const dim3 block_dim(DIVUP(w, ker_block_w), DIVUP(h, ker_block_h), n * c);
    const dim3 thread_dim(DIVUP(ker_block_w, dct_block),
                          DIVUP(ker_block_h, dct_block));
    auto cuda_dtype_param = CudaPostProcess<DstDtype>(scale);
    if (fix_32_mask) {
        kern_dct<dct_block, MaskType::FIX_32_MASK, ker_block_h, ker_block_w,
                 format><<<block_dim, thread_dim, 0, stream>>>(
                d_src, d_dst, n, c, h, w, oh, ow, oc_stride, oc, mask_offset,
                mask_val, cuda_dtype_param, error_info, error_tracker);
    } else if (mask_offset && mask_val) {
        kern_dct<dct_block, MaskType::USER_DEFINE_MASK, ker_block_h,
                 ker_block_w, format><<<block_dim, thread_dim, 0, stream>>>(
                d_src, d_dst, n, c, h, w, oh, ow, oc_stride, oc, mask_offset,
                mask_val, cuda_dtype_param, error_info, error_tracker);
    } else {
        kern_dct<dct_block, MaskType::NO_MASK, ker_block_h, ker_block_w, format>
                <<<block_dim, thread_dim, 0, stream>>>(
                        d_src, d_dst, n, c, h, w, oh, ow, oc_stride, oc,
                        mask_offset, mask_val, cuda_dtype_param, error_info,
                        error_tracker);
    }
}

template void call_kern_dct<8, DctLayoutFormat::NCHW, float>(
        const uint8_t* d_src, float* d_dst, const int n, const int c,
        const int h, const int w, const int oc, bool fix_32_mask,
        const int* mask_offset, const int* mask_val, cudaStream_t stream,
        megcore::AsyncErrorInfo* error_info, void* error_tracker, float scale);

template void call_kern_dct<8, DctLayoutFormat::NCHW4, int8_t>(
        const uint8_t* d_src, int8_t* d_dst, const int n, const int c,
        const int h, const int w, const int oc, bool fix_32_mask,
        const int* mask_offset, const int* mask_val, cudaStream_t stream,
        megcore::AsyncErrorInfo* error_info, void* error_tracker, float scale);

}  // namespace dct

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen