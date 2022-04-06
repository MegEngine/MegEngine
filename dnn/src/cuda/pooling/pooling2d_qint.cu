/**
 * \file dnn/src/cuda/pooling/pooling2d_qint.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "./pooling2d_qint.cuh"
#include "src/common/opr_param_defs_enumv.cuh"
#include "src/cuda/kernel_common/diagnostic_prologue.cuh"
#include "src/cuda/query_blocksize.cuh"

using namespace megdnn;
using namespace cuda;
using namespace pooling2d;

namespace {
__device__ __forceinline__ int pack_int8_to_int8x4(
        int8_t x, int8_t y, int8_t z, int8_t w) {
    int ix = static_cast<int>(x), iy = static_cast<int>(y), iz = static_cast<int>(z),
        iw = static_cast<int>(w);

    asm volatile("prmt.b32 %0, %0, %1, 0x1140;" : "+r"(ix) : "r"(iy));
    asm volatile("prmt.b32 %0, %0, %1, 0x1140;" : "+r"(iz) : "r"(iw));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(ix) : "r"(iz));
    return ix;
}

template <int regs, int dtype_bits, typename OutDtype>
__device__ __forceinline__ OutDtype pack_int8(int8_t (&x)[regs]);

template <>
__device__ __forceinline__ int pack_int8<4, 8, int>(int8_t (&x)[4]) {
    return pack_int8_to_int8x4(x[0], x[1], x[2], x[3]);
}

template <>
__device__ __forceinline__ int2 pack_int8<8, 8, int2>(int8_t (&x)[8]) {
    int8_t x0[4]{x[0], x[1], x[2], x[3]};
    int8_t x1[4]{x[4], x[5], x[6], x[7]};
    return ::make_int2(pack_int8<4, 8, int>(x0), pack_int8<4, 8, int>(x1));
}

template <>
__device__ __forceinline__ int4 pack_int8<16, 8, int4>(int8_t (&x)[16]) {
    int8_t x0[4]{x[0], x[1], x[2], x[3]};
    int8_t x1[4]{x[4], x[5], x[6], x[7]};
    int8_t x2[4]{x[8], x[9], x[10], x[11]};
    int8_t x3[4]{x[12], x[13], x[14], x[15]};
    return ::make_int4(
            pack_int8<4, 8, int>(x0), pack_int8<4, 8, int>(x1),
            pack_int8<4, 8, int>(x2), pack_int8<4, 8, int>(x3));
}

__device__ __forceinline__ int8_t pack_int8_to_int4x2(int8_t x0, int8_t x1) {
    return (x0 & 0xf) | (x1 << 4);
}
template <>
__device__ __forceinline__ int pack_int8<8, 4, int>(int8_t (&x)[8]) {
    int8_t x0 = pack_int8_to_int4x2(x[0], x[1]);
    int8_t x1 = pack_int8_to_int4x2(x[2], x[3]);
    int8_t x2 = pack_int8_to_int4x2(x[4], x[5]);
    int8_t x3 = pack_int8_to_int4x2(x[6], x[7]);
    return pack_int8_to_int8x4(x0, x1, x2, x3);
}

template <>
__device__ __forceinline__ int4 pack_int8<32, 4, int4>(int8_t (&x)[32]) {
    int8_t x0[8]{x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]};
    int8_t x1[8]{x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]};
    int8_t x2[8]{x[16], x[17], x[18], x[19], x[20], x[21], x[22], x[23]};
    int8_t x3[8]{x[24], x[25], x[26], x[27], x[28], x[29], x[30], x[31]};
    return ::make_int4(
            pack_int8<8, 4, int>(x0), pack_int8<8, 4, int>(x1),
            pack_int8<8, 4, int>(x2), pack_int8<8, 4, int>(x3));
}

template <typename Dtype>
struct TypeTrait;

template <>
struct TypeTrait<int8_t> {
    static constexpr int bit_width = 8;
    static constexpr int mask = 0xff;
    static constexpr int8_t min = -128;
    static constexpr int elem_per_32bit = 32 / bit_width;
    static constexpr int shift_fix_sign = 0;
    static constexpr bool need_zero_pad = false;
};

template <>
struct TypeTrait<dt_qint4> {
    static constexpr int bit_width = 4;
    static constexpr int mask = 0xf;
    static constexpr int8_t min = -8;
    static constexpr int elem_per_32bit = 32 / bit_width;
    static constexpr int shift_fix_sign = 4;
    static constexpr bool need_zero_pad = false;
};
template <>
struct TypeTrait<dt_quint4> {
    static constexpr int bit_width = 4;
    static constexpr int mask = 0xf;
    static constexpr int8_t min = 0;
    static constexpr int elem_per_32bit = 32 / bit_width;
    static constexpr int shift_fix_sign = 0;
    static constexpr bool need_zero_pad = true;
};

template <typename src_type, typename _feed_type>
struct MaxPooler {
    using feed_type = _feed_type;
    static constexpr int bit_width = TypeTrait<src_type>::bit_width;
    static constexpr int nr_results = sizeof(feed_type) * 8 / bit_width;
    static constexpr int elem_per_32bit = TypeTrait<src_type>::elem_per_32bit;
    static constexpr int shift_fix_sign = TypeTrait<src_type>::shift_fix_sign;
    int8_t res[nr_results];

    __device__ MaxPooler(int, int) {}
    __device__ __forceinline__ void init() {
#pragma unroll
        for (int i = 0; i < nr_results; ++i) {
            res[i] = TypeTrait<src_type>::min;
        }
    }
    __device__ __forceinline__ void feed(int x, int idx = 0) {
        constexpr int unroll_n = sizeof(int) * 8 / bit_width;
#pragma unroll
        for (int i = 0; i < unroll_n; i++) {
            int8_t temp = ((x >> (i * bit_width)) & TypeTrait<src_type>::mask)
                       << shift_fix_sign;
            temp = temp >> shift_fix_sign;
            res[idx + i] = res[idx + i] > temp ? res[idx + i] : temp;
        }
    }
    __device__ __forceinline__ void feed(int2 x) {
        feed(x.x, 0 * elem_per_32bit);
        feed(x.y, 1 * elem_per_32bit);
    }
    __device__ __forceinline__ void feed(int4 x) {
        feed(x.x, 0 * elem_per_32bit);
        feed(x.y, 1 * elem_per_32bit);
        feed(x.z, 2 * elem_per_32bit);
        feed(x.w, 3 * elem_per_32bit);
    }
    __device__ __forceinline__ feed_type get_ans() {
        feed_type ans;
        ans = pack_int8<nr_results, bit_width, feed_type>(res);
        return ans;
    }
};

template <typename src_type, typename _feed_type, typename inter_type>
struct MeanIncludeRoundedPooler {
    using feed_type = _feed_type;
    static constexpr int bit_width = TypeTrait<src_type>::bit_width;
    static constexpr int nr_results = sizeof(feed_type) * 8 / bit_width;
    static constexpr int elem_per_32bit = TypeTrait<src_type>::elem_per_32bit;
    static constexpr int shift_fix_sign = TypeTrait<src_type>::shift_fix_sign;
    static constexpr bool need_zero_pad = TypeTrait<src_type>::need_zero_pad;

    int32_t res[nr_results];
    const int count;
    const float fi_count;
    int real_fi_count;
    const int zero_pad;

    __device__ MeanIncludeRoundedPooler(int count, int zero_point)
            : count{count}, fi_count{1.f / count}, zero_pad{zero_point} {}

    __device__ __forceinline__ void init() {
#pragma unroll
        for (int i = 0; i < nr_results; ++i) {
            res[i] = 0;
        }
        if (need_zero_pad) {
            real_fi_count = 0;
        }
    }
    __device__ __forceinline__ void feed(int x, int idx) {
        constexpr int unroll_n = sizeof(int) * 8 / bit_width;
#pragma unroll
        for (int i = 0; i < unroll_n; i++) {
            int8_t temp = ((x >> (i * bit_width)) & TypeTrait<src_type>::mask)
                       << shift_fix_sign;
            temp = temp >> shift_fix_sign;
            res[idx + i] += static_cast<int32_t>(temp);
        }
    }
    __device__ __forceinline__ void feed(int x) {
        feed(x, 0);
        if (need_zero_pad) {
            real_fi_count++;
        }
    }
    __device__ __forceinline__ void feed(int2 x) {
        feed(x.x, 0 * elem_per_32bit);
        feed(x.y, 1 * elem_per_32bit);
        if (need_zero_pad) {
            real_fi_count++;
        }
    }
    __device__ __forceinline__ void feed(int4 x) {
        feed(x.x, 0 * elem_per_32bit);
        feed(x.y, 1 * elem_per_32bit);
        feed(x.z, 2 * elem_per_32bit);
        feed(x.w, 3 * elem_per_32bit);
        if (need_zero_pad) {
            real_fi_count++;
        }
    }
    __device__ __forceinline__ feed_type get_ans() {
        feed_type ans;
        int8_t out_res[nr_results];
#pragma unroll
        for (int i = 0; i < nr_results; i++) {
            float f32_res = roundf(static_cast<float>(res[i]) * fi_count);
            if (need_zero_pad) {
                f32_res =
                        roundf((static_cast<float>(res[i]) +
                                (count - real_fi_count) * zero_pad) *
                               fi_count);
            }
            int i8_res;
            asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(i8_res) : "f"(f32_res));
            out_res[i] = i8_res;
        }
        ans = pack_int8<nr_results, bit_width, feed_type>(out_res);
        return ans;
    }
};

template <typename src_type, typename _feed_type, typename inter_type>
struct MeanExcludeRoundedPooler {
    using feed_type = _feed_type;
    static constexpr int bit_width = TypeTrait<src_type>::bit_width;
    static constexpr int nr_results = sizeof(feed_type) * 8 / bit_width;
    static constexpr int elem_per_32bit = TypeTrait<src_type>::elem_per_32bit;
    static constexpr int shift_fix_sign = TypeTrait<src_type>::shift_fix_sign;
    int32_t res[nr_results];
    int count;
    __device__ MeanExcludeRoundedPooler(int, int) {}

    __device__ __forceinline__ void init() {
#pragma unroll
        for (int i = 0; i < nr_results; ++i) {
            res[i] = 0;
        }
        count = 0;
    }
    __device__ __forceinline__ void feed(int x, int idx) {
        constexpr int unroll_n = sizeof(int) * 8 / bit_width;
#pragma unroll
        for (int i = 0; i < unroll_n; i++) {
            int8_t temp = ((x >> (i * bit_width)) & TypeTrait<src_type>::mask)
                       << shift_fix_sign;
            temp = temp >> shift_fix_sign;
            res[idx + i] += static_cast<int32_t>(temp);
        }
    }
    __device__ __forceinline__ void feed(int x) {
        feed(x, 0);
        count++;
    }

    __device__ __forceinline__ void feed(int2 x) {
        feed(x.x, 0 * elem_per_32bit);
        feed(x.y, 1 * elem_per_32bit);
        count++;
    }
    __device__ __forceinline__ void feed(int4 x) {
        feed(x.x, 0 * elem_per_32bit);
        feed(x.y, 1 * elem_per_32bit);
        feed(x.z, 2 * elem_per_32bit);
        feed(x.w, 3 * elem_per_32bit);
        count++;
    }
    __device__ __forceinline__ feed_type get_ans() {
        feed_type ans;
        int8_t out_res[nr_results];
#pragma unroll
        for (int i = 0; i < nr_results; i++) {
            float f32_res = roundf(static_cast<float>(res[i]) / count);
            int i8_res;
            asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(i8_res) : "f"(f32_res));
            out_res[i] = i8_res;
        }
        ans = pack_int8<nr_results, bit_width, feed_type>(out_res);
        return ans;
    }
};

template <typename Pooler>
__global__ void pooling2d_device_template_int8_cdiv4hwn4(
        const int8_t* __restrict__ src, int8_t* __restrict__ dst, Param param) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    using ldg_type = typename Pooler::feed_type;
    static int constexpr pack_size = 4;
    static int constexpr ldg_width = sizeof(ldg_type) / sizeof(int32_t);
    const int batch = (bidy * blockDim.x + tidx) * ldg_width;
    const int packed_ch = bidz * blockDim.y + tidy;
    const int npack = param.n * pack_size;
    if (batch >= param.n || packed_ch >= param.c / pack_size)
        return;

    const int ho = bidx / param.wo;
    const int wo = bidx - param.wo * ho;
    const int input_pixels = param.hi * param.wi;
    const int output_pixels = param.ho * param.wo;
    const int8_t* __restrict__ g_src_ptr =
            src + batch * pack_size + packed_ch * input_pixels * npack;
    int8_t* __restrict__ g_dst_ptr = dst + batch * pack_size +
                                     packed_ch * output_pixels * npack +
                                     (ho * param.wo + wo) * npack;

    Pooler pooler(param.window_h * param.window_w, 0);
    pooler.init();
    for (int fh = 0; fh < param.window_h; fh++) {
        uint32_t ih = ho * param.sh + fh - param.ph;
        for (int fw = 0; fw < param.window_w; fw++) {
            uint32_t iw = wo * param.sw + fw - param.pw;
            if (ih < param.hi && iw < param.wi) {
                const int8_t* __restrict__ cur_src_ptr =
                        g_src_ptr + (ih * param.wi + iw) * npack;
                ldg_type sval = __ldg(reinterpret_cast<const ldg_type*>(cur_src_ptr));
                pooler.feed(sval);
            }
        }
    }
    ldg_type res = pooler.get_ans();
    *(reinterpret_cast<ldg_type*>(g_dst_ptr)) = res;
}

template <typename Pooler, int pack_size, int pack_byte, int ldg_width_assert = 4>
__global__ void pooling2d_device_template_nchwc(
        const int8_t* __restrict__ src, int8_t* __restrict__ dst, Param param,
        int zero_point) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    using ldg_type = typename Pooler::feed_type;
    static int constexpr ldg_width = sizeof(ldg_type) / sizeof(int32_t);
    static int constexpr ldg_width_bytes = sizeof(ldg_type);
    static int constexpr section = pack_byte / sizeof(ldg_type);
    MEGDNN_STATIC_ASSERT(
            ldg_width == ldg_width_assert,
            "pooling2d (NCHW64) kernel must use 128bit width ldg instruction")
    const int c_packed = param.c / pack_size;
    const int batch = tid / (param.ho * param.wo * c_packed * section);
    const int batch_residual = tid - batch * param.ho * param.wo * c_packed * section;
    const int oc = batch_residual / (param.ho * param.wo * section);
    const int oc_residual = batch_residual - oc * param.ho * param.wo * section;
    const int oh = oc_residual / (param.wo * section);
    const int oh_residual = (oc_residual - oh * param.wo * section);
    const int ow = oh_residual / section;
    const int sec = oh_residual - ow * section;
    if (batch >= param.n || oc >= c_packed || oh >= param.ho || ow >= param.wo)
        return;

    const int in_batch_stride = param.hi * param.wi * param.c * pack_byte / pack_size;
    const int out_batch_stride = param.ho * param.wo * param.c * pack_byte / pack_size;
    const int in_channel_stride = param.hi * param.wi * pack_byte;
    const int out_channel_stride = param.ho * param.wo * pack_byte;
    const int8_t* __restrict__ g_src_ptr =
            src +
            (batch * in_batch_stride + oc * in_channel_stride + sec * ldg_width_bytes);
    int8_t* __restrict__ g_dst_ptr =
            dst + (batch * out_batch_stride + oc * out_channel_stride +
                   (oh * param.wo + ow) * pack_byte + sec * ldg_width_bytes);

    Pooler pooler(param.window_h * param.window_w, zero_point);
    pooler.init();
    for (int fh = 0; fh < param.window_h; fh++) {
        uint32_t ih = oh * param.sh + fh - param.ph;
        for (int fw = 0; fw < param.window_w; fw++) {
            uint32_t iw = ow * param.sw + fw - param.pw;
            if (ih < param.hi && iw < param.wi) {
                const int8_t* __restrict__ cur_src_ptr =
                        g_src_ptr + (ih * param.wi + iw) * pack_byte;
                ldg_type sval = __ldg(reinterpret_cast<const ldg_type*>(cur_src_ptr));
                pooler.feed(sval);
            }
        }
    }
    ldg_type res = pooler.get_ans();
    *(reinterpret_cast<ldg_type*>(g_dst_ptr)) = res;
}

template <typename Pooler, int pack_size, int pack_byte, int ldg_width_assert = 4>
__global__ void pooling2d_device_template_nhwc(
        const int8_t* __restrict__ src, int8_t* __restrict__ dst, Param param,
        int zero_point) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    using ldg_type = typename Pooler::feed_type;
    static int constexpr ldg_width = sizeof(ldg_type) / sizeof(int32_t);
    static int constexpr ldg_width_bytes = sizeof(ldg_type);
    MEGDNN_STATIC_ASSERT(
            ldg_width == ldg_width_assert,
            "pooling2d (NHWC) kernel must ldg_width == ldg_width_assert")
    const int c_packed = param.c / pack_size;
    const int batch = tid / (param.ho * param.wo * c_packed);
    const int batch_residual = tid - batch * param.ho * param.wo * c_packed;
    const int oh = batch_residual / (param.wo * c_packed);
    const int oh_residual = batch_residual - oh * param.wo * c_packed;
    const int ow = oh_residual / c_packed;
    const int ow_residual = oh_residual - ow * c_packed;
    const int sec = ow_residual;
    if (batch >= param.n || oh >= param.ho || ow >= param.wo)
        return;

    const int in_batch_stride = param.hi * param.wi * param.c * pack_byte / pack_size;
    const int out_batch_stride = param.ho * param.wo * param.c * pack_byte / pack_size;
    const int w_stride = param.c * pack_byte / pack_size;
    const int8_t* __restrict__ g_src_ptr =
            src + (batch * in_batch_stride + sec * ldg_width_bytes);
    int8_t* __restrict__ g_dst_ptr =
            dst + (batch * out_batch_stride + (oh * param.wo + ow) * w_stride +
                   sec * ldg_width_bytes);

    Pooler pooler(param.window_h * param.window_w, zero_point);
    pooler.init();
    for (int fh = 0; fh < param.window_h; fh++) {
        uint32_t ih = oh * param.sh + fh - param.ph;
        for (int fw = 0; fw < param.window_w; fw++) {
            uint32_t iw = ow * param.sw + fw - param.pw;
            if (ih < param.hi && iw < param.wi) {
                const int8_t* __restrict__ cur_src_ptr =
                        g_src_ptr + (ih * param.wi + iw) * w_stride;
                ldg_type sval = __ldg(reinterpret_cast<const ldg_type*>(cur_src_ptr));
                pooler.feed(sval);
            }
        }
    }
    ldg_type res = pooler.get_ans();
    *(reinterpret_cast<ldg_type*>(g_dst_ptr)) = res;
}

};  // namespace

void megdnn::cuda::pooling2d::do_pooling2d_int8_cdiv4hwn4(
        const int8_t* d_src, int8_t* d_dst, const Param& param, cudaStream_t stream,
        uint32_t mode) {
    using Mode = megdnn::param_enumv::Pooling::Mode;
    void (*kern)(const int8_t* __restrict__, int8_t* __restrict__, Param param);
    uint32_t vthreads_x = 0, vthreads_y = param.c / 4;
#define dispatch_pooling_mode(_feed_type)                                   \
    switch (mode) {                                                         \
        case Mode::MAX:                                                     \
            kern = pooling2d_device_template_int8_cdiv4hwn4<                \
                    MaxPooler<int8_t, _feed_type>>;                         \
            break;                                                          \
        case Mode::AVERAGE:                                                 \
            kern = pooling2d_device_template_int8_cdiv4hwn4<                \
                    MeanIncludeRoundedPooler<int8_t, _feed_type, int32_t>>; \
            break;                                                          \
        case Mode::AVERAGE_COUNT_EXCLUDE_PADDING:                           \
            kern = pooling2d_device_template_int8_cdiv4hwn4<                \
                    MeanExcludeRoundedPooler<int8_t, _feed_type, int32_t>>; \
            break;                                                          \
        default:                                                            \
            megdnn_assert(false, "invalid pooling mode");                   \
    }
    if (param.n % 4 == 0) {
        dispatch_pooling_mode(int4);
        vthreads_x = param.n / 4;
    } else if (param.n % 2 == 0) {
        dispatch_pooling_mode(int2);
        vthreads_x = param.n / 2;
    } else {
        dispatch_pooling_mode(int32_t);
        vthreads_x = param.n;
    }
#undef dispatch_pooling_mode
    constexpr uint32_t threads_x = 16;
    uint32_t nr_threads = query_blocksize_for_kernel(kern);
    uint32_t nr_threads_x = std::min(threads_x, vthreads_x),
             nr_threads_y = std::min(nr_threads / nr_threads_x, vthreads_y);
    uint32_t nr_blocks_x = param.ho * param.wo,
             nr_blocks_y = DIVUP(vthreads_x, nr_threads_x),
             nr_blocks_z = DIVUP(vthreads_y, nr_threads_y);
    dim3 threads{nr_threads_x, nr_threads_y, 1};
    dim3 blocks{nr_blocks_x, nr_blocks_y, nr_blocks_z};
    kern<<<blocks, threads, 0, stream>>>(d_src, d_dst, param);
    after_kernel_launch();
}

void megdnn::cuda::pooling2d::do_pooling2d_int8_ncdiv4hw4(
        const int8_t* d_src, int8_t* d_dst, const Param& param, cudaStream_t stream,
        uint32_t mode, bool /* uint_case */, int zero_point) {
    using Mode = megdnn::param_enumv::Pooling::Mode;
    void (*kern)(
            const int8_t* __restrict__, int8_t* __restrict__, Param param,
            int zero_point);
    constexpr int ldg_byte = 4;
    constexpr int elem_per_byte = 1;
    constexpr int pack_size = 4;
    constexpr int pack_byte = pack_size / elem_per_byte;
    constexpr int elem_per_thread = ldg_byte * elem_per_byte;
    constexpr int ldg_assert_width = ldg_byte / sizeof(int32_t);
    uint32_t vthreads = param.n * param.c * param.ho * param.wo / elem_per_thread;
    switch (mode) {
        case Mode::MAX:
            kern = pooling2d_device_template_nchwc<
                    MaxPooler<int8_t, int32_t>, pack_size, pack_byte, ldg_assert_width>;
            break;
        case Mode::AVERAGE:
            kern = pooling2d_device_template_nchwc<
                    MeanIncludeRoundedPooler<int8_t, int32_t, int32_t>, pack_size,
                    pack_byte, ldg_assert_width>;
            break;
        case Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
            kern = pooling2d_device_template_nchwc<
                    MeanExcludeRoundedPooler<int8_t, int32_t, int32_t>, pack_size,
                    pack_byte, ldg_assert_width>;
            break;
        default:
            megdnn_assert(false, "invalid pooling mode");
    }
    uint32_t nr_threads = query_blocksize_for_kernel(kern);
    nr_threads = std::min(nr_threads, vthreads);
    uint32_t nr_blocks = DIVUP(vthreads, nr_threads);
    kern<<<nr_blocks, nr_threads, 0, stream>>>(d_src, d_dst, param, zero_point);
    after_kernel_launch();
}

void megdnn::cuda::pooling2d::do_pooling2d_int8_ncdiv32hw32(
        const int8_t* d_src, int8_t* d_dst, const Param& param, cudaStream_t stream,
        uint32_t mode, bool /* uint_case */, int zero_point) {
    using Mode = megdnn::param_enumv::Pooling::Mode;
    void (*kern)(
            const int8_t* __restrict__, int8_t* __restrict__, Param param,
            int zero_point);
    constexpr int ldg_byte = 16;
    constexpr int elem_per_byte = 1;
    constexpr int pack_size = 32;
    constexpr int pack_byte = pack_size / elem_per_byte;
    constexpr int elem_per_thread = ldg_byte * elem_per_byte;
    uint32_t vthreads = param.n * param.c * param.ho * param.wo / elem_per_thread;
    switch (mode) {
        case Mode::MAX:
            kern = pooling2d_device_template_nchwc<
                    MaxPooler<int8_t, int4>, pack_size, pack_byte>;
            break;
        case Mode::AVERAGE:
            kern = pooling2d_device_template_nchwc<
                    MeanIncludeRoundedPooler<int8_t, int4, int32_t>, pack_size,
                    pack_byte>;
            break;
        case Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
            kern = pooling2d_device_template_nchwc<
                    MeanExcludeRoundedPooler<int8_t, int4, int32_t>, pack_size,
                    pack_byte>;
            break;
        default:
            megdnn_assert(false, "invalid pooling mode");
    }
    uint32_t nr_threads = query_blocksize_for_kernel(kern);
    nr_threads = std::min(nr_threads, vthreads);
    uint32_t nr_blocks = DIVUP(vthreads, nr_threads);
    kern<<<nr_blocks, nr_threads, 0, stream>>>(d_src, d_dst, param, zero_point);
    after_kernel_launch();
}

void megdnn::cuda::pooling2d::do_pooling2d_int4_ncdiv64hw64(
        const int8_t* d_src, int8_t* d_dst, const Param& param, cudaStream_t stream,
        uint32_t mode, bool uint_case, int zero_point) {
    using Mode = megdnn::param_enumv::Pooling::Mode;
    void (*kern)(
            const int8_t* __restrict__, int8_t* __restrict__, Param param,
            int zero_point);
    constexpr int ldg_byte = 16;
    constexpr int elem_per_byte = 2;
    constexpr int pack_size = 64;
    constexpr int pack_byte = pack_size / elem_per_byte;
    constexpr int elem_per_thread = ldg_byte * elem_per_byte;
    uint32_t vthreads = param.n * param.c * param.ho * param.wo / elem_per_thread;
    if (uint_case) {
        switch (mode) {
            case Mode::MAX:
                kern = pooling2d_device_template_nchwc<
                        MaxPooler<dt_quint4, int4>, pack_size, pack_byte>;
                break;
            case Mode::AVERAGE:
                kern = pooling2d_device_template_nchwc<
                        MeanIncludeRoundedPooler<dt_quint4, int4, int32_t>, pack_size,
                        pack_byte>;
                break;
            case Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
                kern = pooling2d_device_template_nchwc<
                        MeanExcludeRoundedPooler<dt_quint4, int4, int32_t>, pack_size,
                        pack_byte>;
                break;
            default:
                megdnn_assert(false, "invalid pooling mode");
        }

    } else {
        switch (mode) {
            case Mode::MAX:
                kern = pooling2d_device_template_nchwc<
                        MaxPooler<dt_qint4, int4>, pack_size, pack_byte>;
                break;
            case Mode::AVERAGE:
                kern = pooling2d_device_template_nchwc<
                        MeanIncludeRoundedPooler<dt_qint4, int4, int32_t>, pack_size,
                        pack_byte>;
                break;
            case Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
                kern = pooling2d_device_template_nchwc<
                        MeanExcludeRoundedPooler<dt_qint4, int4, int32_t>, pack_size,
                        pack_byte>;
                break;
            default:
                megdnn_assert(false, "invalid pooling mode");
        }
    }
    uint32_t nr_threads = query_blocksize_for_kernel(kern);
    nr_threads = std::min(nr_threads, vthreads);
    uint32_t nr_blocks = DIVUP(vthreads, nr_threads);
    kern<<<nr_blocks, nr_threads, 0, stream>>>(d_src, d_dst, param, zero_point);
    after_kernel_launch();
}

void megdnn::cuda::pooling2d::do_pooling2d_int4_nhwc(
        const int8_t* d_src, int8_t* d_dst, const Param& param, cudaStream_t stream,
        uint32_t mode, bool uint_case, int zero_point) {
    using Mode = megdnn::param_enumv::Pooling::Mode;
    void (*kern)(
            const int8_t* __restrict__, int8_t* __restrict__, Param param,
            int zero_point);

    megdnn_assert(param.c % 8 == 0);
    constexpr int ldg_byte = 4;
    constexpr int elem_per_byte = 2;
    constexpr int ldg_width_assert = 1;
    constexpr int pack_size = ldg_byte * elem_per_byte;
    constexpr int pack_byte = pack_size / elem_per_byte;
    constexpr int elem_per_thread = ldg_byte * elem_per_byte;
    uint32_t vthreads = param.n * param.c * param.ho * param.wo / elem_per_thread;
    if (uint_case) {
        switch (mode) {
            case Mode::MAX:
                kern = pooling2d_device_template_nhwc<
                        MaxPooler<dt_quint4, int32_t>, pack_size, pack_byte,
                        ldg_width_assert>;
                break;
            case Mode::AVERAGE:
                kern = pooling2d_device_template_nhwc<
                        MeanIncludeRoundedPooler<dt_quint4, int32_t, int32_t>,
                        pack_size, pack_byte, ldg_width_assert>;
                break;
            case Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
                kern = pooling2d_device_template_nhwc<
                        MeanExcludeRoundedPooler<dt_quint4, int32_t, int32_t>,
                        pack_size, pack_byte, ldg_width_assert>;
                break;
            default:
                megdnn_assert(false, "invalid pooling mode");
        }

    } else {
        switch (mode) {
            case Mode::MAX:
                kern = pooling2d_device_template_nhwc<
                        MaxPooler<dt_qint4, int32_t>, pack_size, pack_byte,
                        ldg_width_assert>;
                break;
            case Mode::AVERAGE:
                kern = pooling2d_device_template_nhwc<
                        MeanIncludeRoundedPooler<dt_qint4, int32_t, int32_t>, pack_size,
                        pack_byte, ldg_width_assert>;
                break;
            case Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
                kern = pooling2d_device_template_nhwc<
                        MeanExcludeRoundedPooler<dt_qint4, int32_t, int32_t>, pack_size,
                        pack_byte, ldg_width_assert>;
                break;
            default:
                megdnn_assert(false, "invalid pooling mode");
        }
    }
    uint32_t nr_threads = query_blocksize_for_kernel(kern);
    nr_threads = std::min(nr_threads, vthreads);
    uint32_t nr_blocks = DIVUP(vthreads, nr_threads);
    kern<<<nr_blocks, nr_threads, 0, stream>>>(d_src, d_dst, param, zero_point);
    after_kernel_launch();
}

#include "src/cuda/kernel_common/diagnostic_epilogue.cuh"
// vim: syntax=cuda.doxygen
