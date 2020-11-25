/**
 * \file dnn/src/cuda/pooling/pooling2d_int8_cdiv4hwn4.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./pooling2d_int8.cuh"
#include "src/common/opr_param_defs_enumv.cuh"
#include "src/cuda/query_blocksize.cuh"

using namespace megdnn;
using namespace cuda;
using namespace pooling2d;

namespace {
// common macros
#define FEED1 Base::feed(x, 0);
#define FEED2           \
    Base::feed(x.x, 0); \
    Base::feed(x.y, 4);
#define FEED4           \
    FEED2;              \
    Base::feed(x.z, 8); \
    Base::feed(x.w, 12);

#define ANS1(cb) cb(Base::res[0], Base::res[1], Base::res[2], Base::res[3], i1);

#define ANS2(cb) \
    ANS1(cb);    \
    cb(Base::res[4], Base::res[5], Base::res[6], Base::res[7], i2);

#define ANS4(cb)                                                      \
    ANS2(cb);                                                         \
    cb(Base::res[8], Base::res[9], Base::res[10], Base::res[11], i3); \
    cb(Base::res[12], Base::res[13], Base::res[14], Base::res[15], i4);

__device__ __forceinline__ int pack_int8_to_int8x4(int8_t x, int8_t y, int8_t z,
                                                   int8_t w) {
    int ix = static_cast<int>(x), iy = static_cast<int>(y),
        iz = static_cast<int>(z), iw = static_cast<int>(w);

    asm volatile("prmt.b32 %0, %0, %1, 0x1140;" : "+r"(ix) : "r"(iy));
    asm volatile("prmt.b32 %0, %0, %1, 0x1140;" : "+r"(iz) : "r"(iw));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(ix) : "r"(iz));
    return ix;
}

template <typename src_type, typename feed_type>
struct MaxPoolerBase;

template <typename feed_type>
struct MaxPoolerBase<int8_t, feed_type> {
    static constexpr int nr_results = sizeof(feed_type) / sizeof(int8_t);
    int8_t res[nr_results];

    __device__ MaxPoolerBase(int) {}
    __device__ __forceinline__ void init() {
#pragma unroll
        for (int i = 0; i < nr_results; ++i) {
            res[i] = -128;
        }
    }
    __device__ __forceinline__ void feed(int32_t x, int idx) {
        int8_t ix = (x & 0xff);
        int8_t iy = ((x >> 8) & 0xff);
        int8_t iz = ((x >> 16) & 0xff);
        int8_t iw = ((x >> 24) & 0xff);
        res[idx] = res[idx] > ix ? res[idx] : ix;
        res[idx + 1] = res[idx + 1] > iy ? res[idx + 1] : iy;
        res[idx + 2] = res[idx + 2] > iz ? res[idx + 2] : iz;
        res[idx + 3] = res[idx + 3] > iw ? res[idx + 3] : iw;
    }
};

template <typename src_type, typename feed_type>
struct MaxPooler;

#define SPEC_WITH_FEED_TYPE(_feed_type) \
    template <>                         \
    struct MaxPooler<int8_t, _feed_type> : MaxPoolerBase<int8_t, _feed_type>

#define COMMON_DEFS(_feed_type)                     \
    using feed_type = _feed_type;                   \
    using Base = MaxPoolerBase<int8_t, _feed_type>; \
    using MaxPoolerBase<int8_t, _feed_type>::MaxPoolerBase;

#define cb(_x, _y, _z, _w, _ret) \
    { _ret = pack_int8_to_int8x4(_x, _y, _z, _w); }

SPEC_WITH_FEED_TYPE(int32_t) {
    COMMON_DEFS(int32_t);
    __device__ __forceinline__ void feed(int32_t x) { FEED1; }

    __device__ __forceinline__ int get_ans() {
        int i1;
        ANS1(cb);
        return i1;
    }
};

SPEC_WITH_FEED_TYPE(int2) {
    COMMON_DEFS(int2);
    __device__ __forceinline__ void feed(int2 x) { FEED2; }
    __device__ __forceinline__ int2 get_ans() {
        int i1, i2;
        ANS2(cb);
        return ::make_int2(i1, i2);
    }
};

SPEC_WITH_FEED_TYPE(int4) {
    COMMON_DEFS(int4);
    __device__ __forceinline__ void feed(int4 x) { FEED4; }

    __device__ __forceinline__ int4 get_ans() {
        int i1, i2, i3, i4;
        ANS4(cb);
        return ::make_int4(i1, i2, i3, i4);
    }
};

#undef cb
#undef COMMON_DEFS
#undef SPEC_WITH_FEED_TYPE

template <typename src_type, typename feed_type, typename inter_type>
struct MeanIncludeRoundedPoolerBase;

template <typename feed_type>
struct MeanIncludeRoundedPoolerBase<int8_t, feed_type, int32_t> {
    static constexpr int nr_results = sizeof(feed_type) / sizeof(int8_t);
    int32_t res[nr_results];
    const int count;
    const float fi_count;

    __device__ MeanIncludeRoundedPoolerBase(int count)
            : count{count}, fi_count{1.f / count} {}
    __device__ __forceinline__ void init() {
#pragma unroll
        for (int i = 0; i < nr_results; ++i) {
            res[i] = 0;
        }
    }

    __device__ __forceinline__ void feed(int32_t x, int idx) {
        int8_t ix = (x & 0xff);
        int8_t iy = ((x >> 8) & 0xff);
        int8_t iz = ((x >> 16) & 0xff);
        int8_t iw = ((x >> 24) & 0xff);
        res[idx] += static_cast<int32_t>(ix);
        res[idx + 1] += static_cast<int32_t>(iy);
        res[idx + 2] += static_cast<int32_t>(iz);
        res[idx + 3] += static_cast<int32_t>(iw);
    }
};

template <typename src_type, typename feed_type, typename inter_type>
struct MeanIncludeRoundedPooler;

#define SPEC_WITH_FEED_TYPE(_feed_type)                          \
    template <>                                                  \
    struct MeanIncludeRoundedPooler<int8_t, _feed_type, int32_t> \
            : MeanIncludeRoundedPoolerBase<int8_t, _feed_type, int32_t>

#define COMMON_DEFS(_feed_type)                                             \
    using feed_type = _feed_type;                                           \
    using Base = MeanIncludeRoundedPoolerBase<int8_t, _feed_type, int32_t>; \
    using MeanIncludeRoundedPoolerBase<int8_t, _feed_type,                  \
                                       int32_t>::MeanIncludeRoundedPoolerBase;

#define cb(_x, _y, _z, _w, _ret)                                          \
    {                                                                     \
        float fx = roundf(static_cast<float>(_x) * Base::fi_count);       \
        float fy = roundf(static_cast<float>(_y) * Base::fi_count);       \
        float fz = roundf(static_cast<float>(_z) * Base::fi_count);       \
        float fw = roundf(static_cast<float>(_w) * Base::fi_count);       \
        _ret = transform_float4_to_int8x4(::make_float4(fx, fy, fz, fw)); \
    }

SPEC_WITH_FEED_TYPE(int32_t) {
    COMMON_DEFS(int32_t);
    __device__ __forceinline__ void feed(int32_t x) { FEED1; }

    __device__ __forceinline__ int get_ans() {
        int i1;
        ANS1(cb);
        return i1;
    }
};

SPEC_WITH_FEED_TYPE(int2) {
    COMMON_DEFS(int2);
    __device__ __forceinline__ void feed(int2 x) { FEED2; }
    __device__ __forceinline__ int2 get_ans() {
        int i1, i2;
        ANS2(cb);
        return ::make_int2(i1, i2);
    }
};

SPEC_WITH_FEED_TYPE(int4) {
    COMMON_DEFS(int4);
    __device__ __forceinline__ void feed(int4 x) { FEED4; }

    __device__ __forceinline__ int4 get_ans() {
        int i1, i2, i3, i4;
        ANS4(cb);
        return ::make_int4(i1, i2, i3, i4);
    }
};

#undef cb
#undef COMMON_DEFS
#undef SPEC_WITH_FEED_TYPE

template <typename src_type, typename feed_type, typename inter_type>
struct MeanExcludeRoundedPoolerBase;

template <typename feed_type>
struct MeanExcludeRoundedPoolerBase<int8_t, feed_type, int32_t> {
    static const int nr_results = sizeof(feed_type) / sizeof(int8_t);
    int32_t res[nr_results];
    int count;

    __device__ MeanExcludeRoundedPoolerBase(int /* count */) {}
    __device__ __forceinline__ void init() {
#pragma unroll
        for (int i = 0; i < nr_results; ++i) {
            res[i] = 0;
        }
        count = 0;
    }

    __device__ __forceinline__ void feed(int32_t x, int idx) {
        int8_t ix = (x & 0xff);
        int8_t iy = ((x >> 8) & 0xff);
        int8_t iz = ((x >> 16) & 0xff);
        int8_t iw = ((x >> 24) & 0xff);
        res[idx] += static_cast<int32_t>(ix);
        res[idx + 1] += static_cast<int32_t>(iy);
        res[idx + 2] += static_cast<int32_t>(iz);
        res[idx + 3] += static_cast<int32_t>(iw);
    }
};

template <typename src_type, typename feed_type, typename inter_type>
struct MeanExcludeRoundedPooler;

#define SPEC_WITH_FEED_TYPE(_feed_type)                          \
    template <>                                                  \
    struct MeanExcludeRoundedPooler<int8_t, _feed_type, int32_t> \
            : MeanExcludeRoundedPoolerBase<int8_t, _feed_type, int32_t>

#define COMMON_DEFS(_feed_type)                                             \
    using feed_type = _feed_type;                                           \
    using Base = MeanExcludeRoundedPoolerBase<int8_t, _feed_type, int32_t>; \
    using MeanExcludeRoundedPoolerBase<int8_t, _feed_type,                  \
                                       int32_t>::MeanExcludeRoundedPoolerBase;

#define cb(_x, _y, _z, _w, _ret)                                          \
    {                                                                     \
        float fx = roundf(static_cast<float>(_x) / Base::count);          \
        float fy = roundf(static_cast<float>(_y) / Base::count);          \
        float fz = roundf(static_cast<float>(_z) / Base::count);          \
        float fw = roundf(static_cast<float>(_w) / Base::count);          \
        _ret = transform_float4_to_int8x4(::make_float4(fx, fy, fz, fw)); \
    }

SPEC_WITH_FEED_TYPE(int32_t) {
    COMMON_DEFS(int32_t);
    __device__ __forceinline__ void feed(int32_t x) {
        FEED1;
        count++;
    }

    __device__ __forceinline__ int get_ans() {
        int i1;
        ANS1(cb);
        return i1;
    }
};

SPEC_WITH_FEED_TYPE(int2) {
    COMMON_DEFS(int2);
    __device__ __forceinline__ void feed(int2 x) {
        FEED2;
        count++;
    }
    __device__ __forceinline__ int2 get_ans() {
        int i1, i2;
        ANS2(cb);
        return ::make_int2(i1, i2);
    }
};

SPEC_WITH_FEED_TYPE(int4) {
    COMMON_DEFS(int4);
    __device__ __forceinline__ void feed(int4 x) {
        FEED4;
        count++;
    }

    __device__ __forceinline__ int4 get_ans() {
        int i1, i2, i3, i4;
        ANS4(cb);
        return ::make_int4(i1, i2, i3, i4);
    }
};

#undef cb
#undef COMMON_DEFS
#undef SPEC_WITH_FEED_TYPE

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

    Pooler pooler(param.window_h * param.window_w);
    pooler.init();
    for (int fh = 0; fh < param.window_h; fh++) {
        uint32_t ih = ho * param.sh + fh - param.ph;
        for (int fw = 0; fw < param.window_w; fw++) {
            uint32_t iw = wo * param.sw + fw - param.pw;
            if (ih < param.hi && iw < param.wi) {
                const int8_t* __restrict__ cur_src_ptr =
                        g_src_ptr + (ih * param.wi + iw) * npack;
                ldg_type sval =
                        __ldg(reinterpret_cast<const ldg_type*>(cur_src_ptr));
                pooler.feed(sval);
            }
        }
    }
    ldg_type res = pooler.get_ans();
    *(reinterpret_cast<ldg_type*>(g_dst_ptr)) = res;
}

template <typename Pooler>
__global__ void pooling2d_device_template_int8_ncdiv4hw4(
        const int8_t* __restrict__ src, int8_t* __restrict__ dst, Param param) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    using ldg_type = typename Pooler::feed_type;
    static int constexpr pack_size = 4;
    static int constexpr ldg_width = sizeof(ldg_type) / sizeof(int32_t);
    MEGDNN_STATIC_ASSERT(
            ldg_width == 1,
            "pooling2d (NCHW4) kernel must use 32bit width ldg instruction");
    const int wo_ldg = param.wo / ldg_width;
    const int c_packed = param.c / pack_size;
    const int batch = tid / (param.ho * wo_ldg * c_packed);
    const int chw = tid - batch * param.ho * wo_ldg * c_packed;
    const int oc_packed = chw / (param.ho * wo_ldg);
    const int hw = chw - oc_packed * param.ho * wo_ldg;
    const int oh = hw / wo_ldg;
    const int ow = (hw - wo_ldg * oh) * ldg_width;
    if (batch >= param.n || oc_packed >= c_packed || oh >= param.ho ||
        ow >= param.wo)
        return;

    const int in_batch_stride = param.hi * param.wi * param.c;
    const int out_batch_stride = param.ho * param.wo * param.c;
    const int in_channel_stride = param.hi * param.wi * pack_size;
    const int out_channel_stride = param.ho * param.wo * pack_size;
    const int8_t* __restrict__ g_src_ptr =
            src + batch * in_batch_stride + oc_packed * in_channel_stride;
    int8_t* __restrict__ g_dst_ptr = dst + batch * out_batch_stride +
                                     oc_packed * out_channel_stride +
                                     (oh * param.wo + ow) * pack_size;

    Pooler pooler(param.window_h * param.window_w);
    pooler.init();
    for (int fh = 0; fh < param.window_h; fh++) {
        uint32_t ih = oh * param.sh + fh - param.ph;
        for (int fw = 0; fw < param.window_w; fw++) {
            uint32_t iw = ow * param.sw + fw - param.pw;
            if (ih < param.hi && iw < param.wi) {
                const int8_t* __restrict__ cur_src_ptr =
                        g_src_ptr + (ih * param.wi + iw) * pack_size;
                ldg_type sval = __ldg(reinterpret_cast<const ldg_type*>(cur_src_ptr));
                pooler.feed(sval);
            }
        }
    }
    ldg_type res = pooler.get_ans();
    *(reinterpret_cast<ldg_type*>(g_dst_ptr)) = res;
}

template <typename Pooler>
__global__ void pooling2d_device_template_int8_ncdiv32hw32(
        const int8_t* __restrict__ src, int8_t* __restrict__ dst, Param param) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    using ldg_type = typename Pooler::feed_type;
    static int constexpr pack_size = 32;
    static int constexpr ldg_width = sizeof(ldg_type) / sizeof(int32_t);
    static int constexpr ldg_width_bytes = sizeof(ldg_type);
    static int constexpr section = pack_size / sizeof(ldg_type);
    MEGDNN_STATIC_ASSERT(
            ldg_width == 4,
            "pooling2d (NCHW32) kernel must use 128bit width ldg instruction");
    const int c_packed = param.c / pack_size;
    const int batch = tid / (param.ho * param.wo * c_packed * section);
    const int batch_residual =
            tid - batch * param.ho * param.wo * c_packed * section;
    const int oc = batch_residual / (param.ho * param.wo * section);
    const int oc_residual = batch_residual - oc * param.ho * param.wo * section;
    const int oh = oc_residual / (param.wo * section);
    const int oh_residual = (oc_residual - oh * param.wo * section);
    const int ow = oh_residual / section;
    const int sec = oh_residual - ow * section;
    if (batch >= param.n || oc >= c_packed || oh >= param.ho || ow >= param.wo)
        return;

    const int in_batch_stride = param.hi * param.wi * param.c;
    const int out_batch_stride = param.ho * param.wo * param.c;
    const int in_channel_stride = param.hi * param.wi * pack_size;
    const int out_channel_stride = param.ho * param.wo * pack_size;
    const int8_t* __restrict__ g_src_ptr = src + batch * in_batch_stride +
                                           oc * in_channel_stride +
                                           sec * ldg_width_bytes;
    int8_t* __restrict__ g_dst_ptr =
            dst + batch * out_batch_stride + oc * out_channel_stride +
            (oh * param.wo + ow) * pack_size + sec * ldg_width_bytes;

    Pooler pooler(param.window_h * param.window_w);
    pooler.init();
    for (int fh = 0; fh < param.window_h; fh++) {
        uint32_t ih = oh * param.sh + fh - param.ph;
        for (int fw = 0; fw < param.window_w; fw++) {
            uint32_t iw = ow * param.sw + fw - param.pw;
            if (ih < param.hi && iw < param.wi) {
                const int8_t* __restrict__ cur_src_ptr =
                        g_src_ptr + (ih * param.wi + iw) * pack_size;
                ldg_type sval =
                        __ldg(reinterpret_cast<const ldg_type*>(cur_src_ptr));
                pooler.feed(sval);
            }
        }
    }
    ldg_type res = pooler.get_ans();
    *(reinterpret_cast<ldg_type*>(g_dst_ptr)) = res;
}

};  // namespace

void megdnn::cuda::pooling2d::do_pooling2d_int8_cdiv4hwn4(const int8_t* d_src,
                                                          int8_t* d_dst,
                                                          const Param& param,
                                                          cudaStream_t stream,
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

void megdnn::cuda::pooling2d::do_pooling2d_int8_ncdiv4hw4(const int8_t* d_src,
                                                          int8_t* d_dst,
                                                          const Param& param,
                                                          cudaStream_t stream,
                                                          uint32_t mode) {
    using Mode = megdnn::param_enumv::Pooling::Mode;
    void (*kern)(const int8_t* __restrict__, int8_t* __restrict__, Param param);
    uint32_t vthreads = param.n * param.c * param.ho * param.wo / 4;
    switch (mode) {
        case Mode::MAX:
            kern = pooling2d_device_template_int8_ncdiv4hw4<
                    MaxPooler<int8_t, int32_t>>;
            break;
        case Mode::AVERAGE:
            kern = pooling2d_device_template_int8_ncdiv4hw4<
                    MeanIncludeRoundedPooler<int8_t, int32_t, int32_t>>;
            break;
        case Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
            kern = pooling2d_device_template_int8_ncdiv4hw4<
                    MeanExcludeRoundedPooler<int8_t, int32_t, int32_t>>;
            break;
        default:
            megdnn_assert(false, "invalid pooling mode");
    }
    uint32_t nr_threads = query_blocksize_for_kernel(kern);
    nr_threads = std::min(nr_threads, vthreads);
    uint32_t nr_blocks = DIVUP(vthreads, nr_threads);
    kern<<<nr_blocks, nr_threads, 0, stream>>>(d_src, d_dst, param);
    after_kernel_launch();
}

void megdnn::cuda::pooling2d::do_pooling2d_int8_ncdiv32hw32(const int8_t* d_src,
                                                            int8_t* d_dst,
                                                            const Param& param,
                                                            cudaStream_t stream,
                                                            uint32_t mode) {
    using Mode = megdnn::param_enumv::Pooling::Mode;
    void (*kern)(const int8_t* __restrict__, int8_t* __restrict__, Param param);
    uint32_t vthreads = param.n * param.c * param.ho * param.wo / 16;
    switch (mode) {
        case Mode::MAX:
            kern = pooling2d_device_template_int8_ncdiv32hw32<
                    MaxPooler<int8_t, int4>>;
            break;
        case Mode::AVERAGE:
            kern = pooling2d_device_template_int8_ncdiv32hw32<
                    MeanIncludeRoundedPooler<int8_t, int4, int32_t>>;
            break;
        case Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
            kern = pooling2d_device_template_int8_ncdiv32hw32<
                    MeanExcludeRoundedPooler<int8_t, int4, int32_t>>;
            break;
        default:
            megdnn_assert(false, "invalid pooling mode");
    }
    uint32_t nr_threads = query_blocksize_for_kernel(kern);
    nr_threads = std::min(nr_threads, vthreads);
    uint32_t nr_blocks = DIVUP(vthreads, nr_threads);
    kern<<<nr_blocks, nr_threads, 0, stream>>>(d_src, d_dst, param);
    after_kernel_launch();
}

#undef FEED1
#undef FEED2
#undef FEED3
#undef ANS1
#undef ANS2
#undef ANS4

// vim: syntax=cuda.doxygen
