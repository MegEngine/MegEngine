/**
 * \file dnn/src/cuda/relayout_format/relayout_format.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/int_fastdiv.cuh"
#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/relayout_format/relayout_format.cuh"
#include "src/cuda/integer_subbyte_utils.cuh"
#include "src/cuda/memory_utils.cuh"
using namespace megdnn;
using namespace cuda;
using namespace integer_subbyte;

namespace {

template <typename SrcType, typename DstType, bool same_scale>
struct CudaPostProcess;

template <>
struct CudaPostProcess<dtype::Uint8, dtype::QuantizedS8, true> {
    CudaPostProcess(float, uint8_t, float, uint8_t){};
    inline __device__ int8_t operator()(uint8_t val) { return val - 128; }
};

template <>
struct CudaPostProcess<dtype::Uint8, dtype::QuantizedS8, false> {
    CudaDTypeParamImpl<dt_qint8> m_dst_type_cvt;
    CudaPostProcess(float, uint8_t, float dst_scale, uint8_t) {
        m_dst_type_cvt = CudaDTypeParamImpl<dt_qint8>(dst_scale);
    };
    inline __device__ int8_t operator()(uint8_t val) {
        return m_dst_type_cvt.quantize((float)val - 128.f).as_int8();
    }
};

template <>
struct CudaPostProcess<dtype::Quantized8Asymm, dtype::QuantizedS8, false> {
    CudaDTypeParamImpl<dt_qint8> m_dst_type_cvt;
    CudaDTypeParamImpl<dt_quint8> m_src_type_cvt;
    CudaPostProcess(float src_scale, uint8_t src_zero_point, float dst_scale,
                    uint8_t) {
        m_dst_type_cvt = CudaDTypeParamImpl<dt_qint8>(dst_scale);
        m_src_type_cvt =
                CudaDTypeParamImpl<dt_quint8>(src_scale, src_zero_point);
    };
    inline __device__ int8_t operator()(uint8_t val) {
        float med_var = m_src_type_cvt.dequantize(dt_quint8(val));
        return m_dst_type_cvt.quantize(med_var).as_int8();
    }
};

template <>
struct CudaPostProcess<dtype::Quantized8Asymm, dtype::QuantizedS8, true> {
    uint8_t m_src_zero_point = 0;
    CudaPostProcess(float, uint8_t src_zero_point, float, uint8_t) {
        m_src_zero_point = src_zero_point;
    };
    inline __device__ int8_t operator()(uint8_t val) {
        return val - m_src_zero_point;
    }
};

template <>
struct CudaPostProcess<dtype::QuantizedS8, dtype::QuantizedS8, false> {
    CudaDTypeParamImpl<dt_qint8> m_dst_type_cvt;
    CudaDTypeParamImpl<dt_qint8> m_src_type_cvt;
    CudaPostProcess(float src_scale, uint8_t, float dst_scale, uint8_t) {
        m_dst_type_cvt = CudaDTypeParamImpl<dt_qint8>(dst_scale);
        m_src_type_cvt = CudaDTypeParamImpl<dt_qint8>(src_scale);
    };
    inline __device__ int8_t operator()(int8_t val) {
        float med_var = m_src_type_cvt.dequantize(dt_qint8(val));
        return m_dst_type_cvt.quantize(med_var).as_int8();
    }
};

template <>
struct CudaPostProcess<dtype::QuantizedS8, dtype::QuantizedS8, true> {
    CudaPostProcess(){};
    CudaPostProcess(float, uint8_t, float, uint8_t){};
    inline __device__ int8_t operator()(int8_t val) { return val; }
};

template <>
struct CudaPostProcess<dtype::QuantizedS32, dtype::QuantizedS32, false> {
    CudaDTypeParamImpl<dt_qint32> m_dst_type_cvt;
    CudaDTypeParamImpl<dt_qint32> m_src_type_cvt;
    CudaPostProcess(float src_scale, int, float dst_scale, int) {
        m_dst_type_cvt = CudaDTypeParamImpl<dt_qint32>(dst_scale);
        m_src_type_cvt = CudaDTypeParamImpl<dt_qint32>(src_scale);
    };
    inline __device__ int operator()(int val) {
        float med_var = m_src_type_cvt.dequantize(dt_qint32(val));
        return m_dst_type_cvt.quantize(med_var).as_int32();
    }
};
template <>
struct CudaPostProcess<dtype::QuantizedS32, dtype::QuantizedS32, true> {
    CudaPostProcess(float, int, float, int){};
    inline __device__ int operator()(int val) { return val; }
};

template <>
struct CudaPostProcess<dtype::QuantizedS4, dtype::QuantizedS4, false> {
    using SrcType = dtype::QuantizedS4;
    using DstType = dtype::QuantizedS4;
    CudaDTypeParamImpl<dt_qint4> m_dst_type_cvt;
    CudaDTypeParamImpl<dt_qint4> m_src_type_cvt;
    CudaPostProcess(float src_scale, uint8_t, float dst_scale, uint8_t) {
        m_dst_type_cvt = CudaDTypeParamImpl<dt_qint4>(dst_scale);
        m_src_type_cvt = CudaDTypeParamImpl<dt_qint4>(src_scale);
    }
    inline __device__ int8_t operator()(int8_t val) {
        float intermediate = m_src_type_cvt.dequantize(dt_qint4(val));
        return m_dst_type_cvt.quantize(intermediate).as_int8();
    }
};

template <>
struct CudaPostProcess<dtype::QuantizedS4, dtype::QuantizedS4, true> {
    using SrcType = dtype::QuantizedS4;
    using DstType = dtype::QuantizedS4;
    CudaPostProcess(float, uint8_t, float, uint8_t){};
    inline __device__ int8_t operator()(int8_t val) { return val; }
};

template <>
struct CudaPostProcess<dtype::Quantized4Asymm, dtype::Quantized4Asymm, false> {
    using SrcType = dtype::Quantized4Asymm;
    using DstType = dtype::Quantized4Asymm;
    CudaDTypeParamImpl<dt_quint4> m_dst_type_cvt;
    CudaDTypeParamImpl<dt_quint4> m_src_type_cvt;
    CudaPostProcess(float src_scale, uint8_t src_zero_point, float dst_scale,
                    uint8_t dst_zero_point) {
        m_dst_type_cvt =
                CudaDTypeParamImpl<dt_quint4>(dst_scale, dst_zero_point);
        m_src_type_cvt =
                CudaDTypeParamImpl<dt_quint4>(src_scale, src_zero_point);
    };
    inline __device__ uint8_t operator()(uint8_t val) {
        float intermediate = m_src_type_cvt.dequantize(dt_quint4(val));
        return m_dst_type_cvt.quantize(intermediate).as_uint8();
    }
};

template <>
struct CudaPostProcess<dtype::Quantized4Asymm, dtype::Quantized4Asymm, true> {
    using SrcType = dtype::Quantized4Asymm;
    using DstType = dtype::Quantized4Asymm;
    uint8_t m_src_zero_point = 0;
    uint8_t m_dst_zero_point = 0;
    CudaPostProcess(float, uint8_t src_zero_point, float,
                    uint8_t dst_zero_point) {
        m_src_zero_point = src_zero_point;
        m_dst_zero_point = dst_zero_point;
    };
    inline __device__ uint8_t operator()(uint8_t val) {
        int result = val - m_src_zero_point + m_dst_zero_point;
        result = result >= 0 ? result : 0;
        result = result < 16 ? result : 15;
        return static_cast<uint8_t>(result);
    }
};

template <typename cype, int pack_w, typename enable = void>
struct DTypeRWHelper;
template <typename ctype>
struct DTypeRWHelper<
        ctype, 1,
        typename std::enable_if<std::is_same<ctype, dt_qint8>::value ||
                                std::is_same<ctype, dt_quint8>::value ||
                                std::is_same<ctype, dt_uint8>::value>::type> {
    using InnerDtype = char;
    using DstDtype = char4;
};

template <typename ctype>
struct DTypeRWHelper<
        ctype, 4,
        typename std::enable_if<std::is_same<ctype, dt_qint8>::value ||
                                std::is_same<ctype, dt_quint8>::value ||
                                std::is_same<ctype, dt_uint8>::value>::type> {
    using InnerDtype = char4;
    using DstDtype = char4;
};

template <>
struct DTypeRWHelper<dt_qint32, 1> {
    using InnerDtype = int;
    using DstDtype = int4;
};

template <>
struct DTypeRWHelper<dt_qint32, 4> {
    using InnerDtype = int4;
    using DstDtype = int4;
};
    
template <typename ctype>
struct DTypeRWHelper<
        ctype, 2,
        typename std::enable_if<std::is_same<ctype, dt_qint4>::value ||
                                std::is_same<ctype, dt_quint4>::value>::type> {
    using InnerDtype = char;
    using DstDtype = array_wrapper<uint8_t, 32>;
};

template <typename ctype>
struct DTypeRWHelper<
        ctype, 8,
        typename std::enable_if<std::is_same<ctype, dt_qint4>::value ||
                                std::is_same<ctype, dt_quint4>::value>::type> {
    using InnerDtype = unsigned;
    using DstDtype = array_wrapper<uint8_t, 32>;
};

template <int pack_w, int pack_c, typename SrcType, typename DnnSrcType,
          typename DnnDstType, bool same_scale>
struct Translayout {
    using InnerDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   pack_w>::InnerDtype;
    using DstDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   pack_w>::DstDtype;
    static inline __device__ void trans(DstDtype (&dst_width)[pack_w],
                                        InnerDtype (&read_channel)[pack_c],
                                        const char zero_point);
};

template <typename SrcType, typename DnnSrcType, typename DnnDstType,
          bool same_scale>
struct Translayout<1, 4, SrcType, DnnSrcType, DnnDstType, same_scale> {
    using InnerDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   1>::InnerDtype;
    using DstDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   1>::DstDtype;
    static inline __device__ void trans(
            DstDtype (&dst_width)[1], InnerDtype (&read_channel)[4],
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        dst_width[0].x = post_process(read_channel[0]);
        dst_width[0].y = post_process(read_channel[1]);
        dst_width[0].z = post_process(read_channel[2]);
        dst_width[0].w = post_process(read_channel[3]);
    }
};

template <typename SrcType, typename DnnSrcType, typename DnnDstType,
          bool same_scale>
struct Translayout<4, 4, SrcType, DnnSrcType, DnnDstType, same_scale> {
    using InnerDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   4>::InnerDtype;
    using DstDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   4>::DstDtype;
    static inline __device__ void trans(
            DstDtype (&dst_width)[4], InnerDtype (&read_channel)[4],
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        dst_width[0].x = post_process(read_channel[0].x);
        dst_width[0].y = post_process(read_channel[1].x);
        dst_width[0].z = post_process(read_channel[2].x);
        dst_width[0].w = post_process(read_channel[3].x);

        dst_width[1].x = post_process(read_channel[0].y);
        dst_width[1].y = post_process(read_channel[1].y);
        dst_width[1].z = post_process(read_channel[2].y);
        dst_width[1].w = post_process(read_channel[3].y);

        dst_width[2].x = post_process(read_channel[0].z);
        dst_width[2].y = post_process(read_channel[1].z);
        dst_width[2].z = post_process(read_channel[2].z);
        dst_width[2].w = post_process(read_channel[3].z);

        dst_width[3].x = post_process(read_channel[0].w);
        dst_width[3].y = post_process(read_channel[1].w);
        dst_width[3].z = post_process(read_channel[2].w);
        dst_width[3].w = post_process(read_channel[3].w);
    }
};

#define pack_channel(_idx)                                        \
    transform_int8_to_int4x8(post_process(intermediate[0][_idx]), \
                             post_process(intermediate[1][_idx]), \
                             post_process(intermediate[2][_idx]), \
                             post_process(intermediate[3][_idx]), \
                             post_process(intermediate[4][_idx]), \
                             post_process(intermediate[5][_idx]), \
                             post_process(intermediate[6][_idx]), \
                             post_process(intermediate[7][_idx]));
template <typename SrcType, bool same_scale>
struct Translayout<2, 64, SrcType, dtype::QuantizedS4, dtype::QuantizedS4,
                   same_scale> {
    using DnnSrcType = dtype::QuantizedS4;
    using DnnDstType = dtype::QuantizedS4;
    using InnerDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   2>::InnerDtype;
    using DstDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   2>::DstDtype;
    static inline __device__ void trans(
            DstDtype (&dst_width)[2], InnerDtype (&read_channel)[64],
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        int intermediate[8][2];
        int* dst_frag = reinterpret_cast<int*>(dst_width);
#pragma unroll
        for (int i = 0; i < 64; i += 8) {
            transform_int4x2_to_int8(
                    intermediate[0],
                    reinterpret_cast<uint8_t&>(read_channel[i + 0]));
            transform_int4x2_to_int8(
                    intermediate[1],
                    reinterpret_cast<uint8_t&>(read_channel[i + 1]));
            transform_int4x2_to_int8(
                    intermediate[2],
                    reinterpret_cast<uint8_t&>(read_channel[i + 2]));
            transform_int4x2_to_int8(
                    intermediate[3],
                    reinterpret_cast<uint8_t&>(read_channel[i + 3]));
            transform_int4x2_to_int8(
                    intermediate[4],
                    reinterpret_cast<uint8_t&>(read_channel[i + 4]));
            transform_int4x2_to_int8(
                    intermediate[5],
                    reinterpret_cast<uint8_t&>(read_channel[i + 5]));
            transform_int4x2_to_int8(
                    intermediate[6],
                    reinterpret_cast<uint8_t&>(read_channel[i + 6]));
            transform_int4x2_to_int8(
                    intermediate[7],
                    reinterpret_cast<uint8_t&>(read_channel[i + 7]));

            int frag_idx = i / 8;
            dst_frag[0 * 8 + frag_idx] = pack_channel(0);
            dst_frag[1 * 8 + frag_idx] = pack_channel(1);
        }
    }
    using Fragment = array_wrapper<SrcType, 64>;
    static inline __device__ void trans(
            Fragment& dst, Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process) {
        trans(reinterpret_cast<DstDtype(&)[2]>(dst),
              reinterpret_cast<InnerDtype(&)[64]>(src), post_process, 0);
    }
};

template <typename SrcType, bool same_scale>
struct Translayout<8, 64, SrcType, dtype::QuantizedS4, dtype::QuantizedS4,
                   same_scale> {
    using DnnSrcType = dtype::QuantizedS4;
    using DnnDstType = dtype::QuantizedS4;
    using InnerDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   8>::InnerDtype;
    using DstDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   8>::DstDtype;
    static inline __device__ void trans(
            DstDtype (&dst_width)[8], InnerDtype (&read_channel)[64],
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        int intermediate[8][8];
        int* dst_frag = reinterpret_cast<int*>(dst_width);
#pragma unroll
        for (int i = 0; i < 64; i += 8) {
            transform_int4x8_to_int8(intermediate[0], read_channel[i + 0]);
            transform_int4x8_to_int8(intermediate[1], read_channel[i + 1]);
            transform_int4x8_to_int8(intermediate[2], read_channel[i + 2]);
            transform_int4x8_to_int8(intermediate[3], read_channel[i + 3]);
            transform_int4x8_to_int8(intermediate[4], read_channel[i + 4]);
            transform_int4x8_to_int8(intermediate[5], read_channel[i + 5]);
            transform_int4x8_to_int8(intermediate[6], read_channel[i + 6]);
            transform_int4x8_to_int8(intermediate[7], read_channel[i + 7]);
            int frag_idx = i / 8;
            dst_frag[0 * 8 + frag_idx] = pack_channel(0);
            dst_frag[1 * 8 + frag_idx] = pack_channel(1);
            dst_frag[2 * 8 + frag_idx] = pack_channel(2);
            dst_frag[3 * 8 + frag_idx] = pack_channel(3);
            dst_frag[4 * 8 + frag_idx] = pack_channel(4);
            dst_frag[5 * 8 + frag_idx] = pack_channel(5);
            dst_frag[6 * 8 + frag_idx] = pack_channel(6);
            dst_frag[7 * 8 + frag_idx] = pack_channel(7);
        }
    }
    using Fragment = array_wrapper<unsigned, 64>;
    static inline __device__ void trans(
            Fragment& dst, Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process) {
        trans(reinterpret_cast<DstDtype(&)[8]>(dst),
              reinterpret_cast<InnerDtype(&)[64]>(src), post_process, 0);
    }
};
#undef pack_channel

#define pack_channel(_idx)                                         \
    transform_int8_to_uint4x8(post_process(intermediate[0][_idx]), \
                              post_process(intermediate[1][_idx]), \
                              post_process(intermediate[2][_idx]), \
                              post_process(intermediate[3][_idx]), \
                              post_process(intermediate[4][_idx]), \
                              post_process(intermediate[5][_idx]), \
                              post_process(intermediate[6][_idx]), \
                              post_process(intermediate[7][_idx]));
template <typename SrcType, bool same_scale>
struct Translayout<2, 64, SrcType, dtype::Quantized4Asymm,
                   dtype::Quantized4Asymm, same_scale> {
    using DnnSrcType = dtype::Quantized4Asymm;
    using DnnDstType = dtype::Quantized4Asymm;
    using InnerDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   2>::InnerDtype;
    using DstDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   2>::DstDtype;
    static inline __device__ void trans(
            DstDtype (&dst_width)[2], InnerDtype (&read_channel)[64],
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        int intermediate[8][2];
        int* dst_frag = reinterpret_cast<int*>(dst_width);
#pragma unroll
        for (int i = 0; i < 64; i += 8) {
            transform_uint4x2_to_int8(
                    intermediate[0],
                    reinterpret_cast<uint8_t&>(read_channel[i + 0]));
            transform_uint4x2_to_int8(
                    intermediate[1],
                    reinterpret_cast<uint8_t&>(read_channel[i + 1]));
            transform_uint4x2_to_int8(
                    intermediate[2],
                    reinterpret_cast<uint8_t&>(read_channel[i + 2]));
            transform_uint4x2_to_int8(
                    intermediate[3],
                    reinterpret_cast<uint8_t&>(read_channel[i + 3]));
            transform_uint4x2_to_int8(
                    intermediate[4],
                    reinterpret_cast<uint8_t&>(read_channel[i + 4]));
            transform_uint4x2_to_int8(
                    intermediate[5],
                    reinterpret_cast<uint8_t&>(read_channel[i + 5]));
            transform_uint4x2_to_int8(
                    intermediate[6],
                    reinterpret_cast<uint8_t&>(read_channel[i + 6]));
            transform_uint4x2_to_int8(
                    intermediate[7],
                    reinterpret_cast<uint8_t&>(read_channel[i + 7]));

            int frag_idx = i / 8;
            dst_frag[0 * 8 + frag_idx] = pack_channel(0);
            dst_frag[1 * 8 + frag_idx] = pack_channel(1);
        }
    }
    using Fragment = array_wrapper<SrcType, 64>;
    static inline __device__ void trans(
            Fragment& dst, Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process) {
        trans(reinterpret_cast<DstDtype(&)[2]>(dst),
              reinterpret_cast<InnerDtype(&)[64]>(src), post_process, 0);
    }
};

template <typename SrcType, bool same_scale>
struct Translayout<8, 64, SrcType, dtype::Quantized4Asymm,
                   dtype::Quantized4Asymm, same_scale> {
    using DnnSrcType = dtype::Quantized4Asymm;
    using DnnDstType = dtype::Quantized4Asymm;
    using InnerDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   8>::InnerDtype;
    using DstDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   8>::DstDtype;
    static inline __device__ void trans(
            DstDtype (&dst_width)[8], InnerDtype (&read_channel)[64],
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        int intermediate[8][8];
        int* dst_frag = reinterpret_cast<int*>(dst_width);
#pragma unroll
        for (int i = 0; i < 64; i += 8) {
            transform_uint4x8_to_int8(intermediate[0], read_channel[i + 0]);
            transform_uint4x8_to_int8(intermediate[1], read_channel[i + 1]);
            transform_uint4x8_to_int8(intermediate[2], read_channel[i + 2]);
            transform_uint4x8_to_int8(intermediate[3], read_channel[i + 3]);
            transform_uint4x8_to_int8(intermediate[4], read_channel[i + 4]);
            transform_uint4x8_to_int8(intermediate[5], read_channel[i + 5]);
            transform_uint4x8_to_int8(intermediate[6], read_channel[i + 6]);
            transform_uint4x8_to_int8(intermediate[7], read_channel[i + 7]);
            int frag_idx = i / 8;
            dst_frag[0 * 8 + frag_idx] = pack_channel(0);
            dst_frag[1 * 8 + frag_idx] = pack_channel(1);
            dst_frag[2 * 8 + frag_idx] = pack_channel(2);
            dst_frag[3 * 8 + frag_idx] = pack_channel(3);
            dst_frag[4 * 8 + frag_idx] = pack_channel(4);
            dst_frag[5 * 8 + frag_idx] = pack_channel(5);
            dst_frag[6 * 8 + frag_idx] = pack_channel(6);
            dst_frag[7 * 8 + frag_idx] = pack_channel(7);
        }
    }
    using Fragment = array_wrapper<unsigned, 64>;
    static inline __device__ void trans(
            Fragment& dst, Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process) {
        trans(reinterpret_cast<DstDtype(&)[8]>(dst),
              reinterpret_cast<InnerDtype(&)[64]>(src), post_process, 0);
    }
};
#undef pack_channel

#define pack(_idx)                                                \
    transform_int8_to_int4x8(post_process(intermediate[0][_idx]), \
                             post_process(intermediate[1][_idx]), \
                             post_process(intermediate[2][_idx]), \
                             post_process(intermediate[3][_idx]), \
                             post_process(intermediate[4][_idx]), \
                             post_process(intermediate[5][_idx]), \
                             post_process(intermediate[6][_idx]), \
                             post_process(intermediate[7][_idx]));
template <typename SrcType, bool same_scale>
struct Translayout<64, 8, SrcType, dtype::QuantizedS4, dtype::QuantizedS4,
                   same_scale> {
    using DnnSrcType = dtype::QuantizedS4;
    using DnnDstType = dtype::QuantizedS4;
    static constexpr int row = 8;
    static constexpr int col = 64;
    static constexpr int size_nbits = 4;
    static constexpr int col_in_type = col * size_nbits / (8 * sizeof(SrcType));
    static constexpr int elements_in_type = row * col_in_type;
    static constexpr int inc_col = 8;
    static constexpr int inc_col_in_type =
            inc_col * size_nbits / (8 * sizeof(SrcType));
    using Fragment = array_wrapper<SrcType, elements_in_type>;
    static MEGDNN_DEVICE __forceinline__ void trans(
            Fragment& dst, const Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process) {
        int intermediate[8][8];
        int* dst_frag = reinterpret_cast<int*>(&dst);
#pragma unroll
        for (int j = 0; j < col_in_type; j += inc_col_in_type) {
            transform_int4x8_to_int8(
                    intermediate[0],
                    reinterpret_cast<const int&>(src[0 * col_in_type + j]));
            transform_int4x8_to_int8(
                    intermediate[1],
                    reinterpret_cast<const int&>(src[1 * col_in_type + j]));
            transform_int4x8_to_int8(
                    intermediate[2],
                    reinterpret_cast<const int&>(src[2 * col_in_type + j]));
            transform_int4x8_to_int8(
                    intermediate[3],
                    reinterpret_cast<const int&>(src[3 * col_in_type + j]));
            transform_int4x8_to_int8(
                    intermediate[4],
                    reinterpret_cast<const int&>(src[4 * col_in_type + j]));
            transform_int4x8_to_int8(
                    intermediate[5],
                    reinterpret_cast<const int&>(src[5 * col_in_type + j]));
            transform_int4x8_to_int8(
                    intermediate[6],
                    reinterpret_cast<const int&>(src[6 * col_in_type + j]));
            transform_int4x8_to_int8(
                    intermediate[7],
                    reinterpret_cast<const int&>(src[7 * col_in_type + j]));
            dst_frag[(j / inc_col_in_type) * 8 + 0] = pack(0);
            dst_frag[(j / inc_col_in_type) * 8 + 1] = pack(1);
            dst_frag[(j / inc_col_in_type) * 8 + 2] = pack(2);
            dst_frag[(j / inc_col_in_type) * 8 + 3] = pack(3);
            dst_frag[(j / inc_col_in_type) * 8 + 4] = pack(4);
            dst_frag[(j / inc_col_in_type) * 8 + 5] = pack(5);
            dst_frag[(j / inc_col_in_type) * 8 + 6] = pack(6);
            dst_frag[(j / inc_col_in_type) * 8 + 7] = pack(7);
        }
    }
};
#undef pack

#define pack(_idx)                                 \
    ((post_process(intermediate[0][_idx]) & 0xf) | \
     (post_process(intermediate[1][_idx]) << 4))
template <typename SrcType, bool same_scale>
struct Translayout<64, 2, SrcType, dtype::QuantizedS4, dtype::QuantizedS4,
                   same_scale> {
    using DnnSrcType = dtype::QuantizedS4;
    using DnnDstType = dtype::QuantizedS4;
    static constexpr int row = 2;
    static constexpr int col = 64;
    static constexpr int size_nbits = 4;
    static constexpr int col_in_type = col * size_nbits / (8 * sizeof(SrcType));
    static constexpr int elements_in_type = row * col_in_type;
    static constexpr int inc_col = 8;
    static constexpr int inc_col_in_type =
            inc_col * size_nbits / (8 * sizeof(SrcType));
    using Fragment = array_wrapper<SrcType, elements_in_type>;
    static MEGDNN_DEVICE __forceinline__ void trans(
            Fragment& dst, const Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process) {
        int intermediate[2][8];
        uint8_t* dst_frag = reinterpret_cast<uint8_t*>(&dst);
#pragma unroll
        for (int j = 0; j < col_in_type; j += inc_col_in_type) {
            transform_int4x8_to_int8(
                    intermediate[0],
                    reinterpret_cast<const int&>(src[0 * col_in_type + j]));
            transform_int4x8_to_int8(
                    intermediate[1],
                    reinterpret_cast<const int&>(src[1 * col_in_type + j]));
            dst_frag[(j / inc_col_in_type) * 8 + 0] = pack(0);
            dst_frag[(j / inc_col_in_type) * 8 + 1] = pack(1);
            dst_frag[(j / inc_col_in_type) * 8 + 2] = pack(2);
            dst_frag[(j / inc_col_in_type) * 8 + 3] = pack(3);
            dst_frag[(j / inc_col_in_type) * 8 + 4] = pack(4);
            dst_frag[(j / inc_col_in_type) * 8 + 5] = pack(5);
            dst_frag[(j / inc_col_in_type) * 8 + 6] = pack(6);
            dst_frag[(j / inc_col_in_type) * 8 + 7] = pack(7);
        }
    }
};
#undef pack

#define pack(_idx)                                                 \
    transform_int8_to_uint4x8(post_process(intermediate[0][_idx]), \
                              post_process(intermediate[1][_idx]), \
                              post_process(intermediate[2][_idx]), \
                              post_process(intermediate[3][_idx]), \
                              post_process(intermediate[4][_idx]), \
                              post_process(intermediate[5][_idx]), \
                              post_process(intermediate[6][_idx]), \
                              post_process(intermediate[7][_idx]));
template <typename SrcType, bool same_scale>
struct Translayout<64, 8, SrcType, dtype::Quantized4Asymm,
                   dtype::Quantized4Asymm, same_scale> {
    using DnnSrcType = dtype::Quantized4Asymm;
    using DnnDstType = dtype::Quantized4Asymm;
    static constexpr int row = 8;
    static constexpr int col = 64;
    static constexpr int size_nbits = 4;
    static constexpr int col_in_type = col * size_nbits / (8 * sizeof(SrcType));
    static constexpr int elements_in_type = row * col_in_type;
    static constexpr int inc_col = 8;
    static constexpr int inc_col_in_type =
            inc_col * size_nbits / (8 * sizeof(SrcType));
    using Fragment = array_wrapper<SrcType, elements_in_type>;
    static MEGDNN_DEVICE __forceinline__ void trans(
            Fragment& dst, const Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process) {
        int intermediate[8][8];
        int* dst_frag = reinterpret_cast<int*>(&dst);
#pragma unroll
        for (int j = 0; j < col_in_type; j += inc_col_in_type) {
            transform_uint4x8_to_int8(
                    intermediate[0],
                    reinterpret_cast<const int&>(src[0 * col_in_type + j]));
            transform_uint4x8_to_int8(
                    intermediate[1],
                    reinterpret_cast<const int&>(src[1 * col_in_type + j]));
            transform_uint4x8_to_int8(
                    intermediate[2],
                    reinterpret_cast<const int&>(src[2 * col_in_type + j]));
            transform_uint4x8_to_int8(
                    intermediate[3],
                    reinterpret_cast<const int&>(src[3 * col_in_type + j]));
            transform_uint4x8_to_int8(
                    intermediate[4],
                    reinterpret_cast<const int&>(src[4 * col_in_type + j]));
            transform_uint4x8_to_int8(
                    intermediate[5],
                    reinterpret_cast<const int&>(src[5 * col_in_type + j]));
            transform_uint4x8_to_int8(
                    intermediate[6],
                    reinterpret_cast<const int&>(src[6 * col_in_type + j]));
            transform_uint4x8_to_int8(
                    intermediate[7],
                    reinterpret_cast<const int&>(src[7 * col_in_type + j]));
            dst_frag[(j / inc_col_in_type) * 8 + 0] = pack(0);
            dst_frag[(j / inc_col_in_type) * 8 + 1] = pack(1);
            dst_frag[(j / inc_col_in_type) * 8 + 2] = pack(2);
            dst_frag[(j / inc_col_in_type) * 8 + 3] = pack(3);
            dst_frag[(j / inc_col_in_type) * 8 + 4] = pack(4);
            dst_frag[(j / inc_col_in_type) * 8 + 5] = pack(5);
            dst_frag[(j / inc_col_in_type) * 8 + 6] = pack(6);
            dst_frag[(j / inc_col_in_type) * 8 + 7] = pack(7);
        }
    }
};
#undef pack

#define pack(_idx)                         \
    (post_process(intermediate[0][_idx]) | \
     (post_process(intermediate[1][_idx]) << 4))
template <typename SrcType, bool same_scale>
struct Translayout<64, 2, SrcType, dtype::Quantized4Asymm,
                   dtype::Quantized4Asymm, same_scale> {
    using DnnSrcType = dtype::Quantized4Asymm;
    using DnnDstType = dtype::Quantized4Asymm;
    static constexpr int row = 2;
    static constexpr int col = 64;
    static constexpr int size_nbits = 4;
    static constexpr int col_in_type = col * size_nbits / (8 * sizeof(SrcType));
    static constexpr int elements_in_type = row * col_in_type;
    static constexpr int inc_col = 8;
    static constexpr int inc_col_in_type =
            inc_col * size_nbits / (8 * sizeof(SrcType));
    using Fragment = array_wrapper<SrcType, elements_in_type>;
    static MEGDNN_DEVICE __forceinline__ void trans(
            Fragment& dst, const Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process) {
        int intermediate[2][8];
        uint8_t* dst_frag = reinterpret_cast<uint8_t*>(&dst);
#pragma unroll
        for (int j = 0; j < col_in_type; j += inc_col_in_type) {
            transform_uint4x8_to_int8(
                    intermediate[0],
                    reinterpret_cast<const int&>(src[0 * col_in_type + j]));
            transform_uint4x8_to_int8(
                    intermediate[1],
                    reinterpret_cast<const int&>(src[1 * col_in_type + j]));
            dst_frag[(j / inc_col_in_type) * 8 + 0] = pack(0);
            dst_frag[(j / inc_col_in_type) * 8 + 1] = pack(1);
            dst_frag[(j / inc_col_in_type) * 8 + 2] = pack(2);
            dst_frag[(j / inc_col_in_type) * 8 + 3] = pack(3);
            dst_frag[(j / inc_col_in_type) * 8 + 4] = pack(4);
            dst_frag[(j / inc_col_in_type) * 8 + 5] = pack(5);
            dst_frag[(j / inc_col_in_type) * 8 + 6] = pack(6);
            dst_frag[(j / inc_col_in_type) * 8 + 7] = pack(7);
        }
    }
};
#undef pack

template <typename DstType>
inline __device__ DstType make_zero_pad(const uint8_t zero_point) {
    return zero_point;
}

template <>
inline __device__ char4 make_zero_pad<char4>(const uint8_t zero_point) {
    char izp = reinterpret_cast<const char&>(zero_point);
    return {izp, izp, izp, izp};
}

template <>
inline __device__ int4 make_zero_pad<int4>(const uint8_t zero_point) {
    return {zero_point, zero_point, zero_point, zero_point};
}

template <int size_nbits>
inline __device__ int make_zero(int zero_point);

template <>
inline __device__ int make_zero<4>(int zero_point) {
    return transform_int8_to_uint4x8(zero_point, zero_point, zero_point,
                                     zero_point, zero_point, zero_point,
                                     zero_point, zero_point);
}

template <typename DstDtype>
inline __device__ void write_helper(DstDtype* ptr, DstDtype val) {
    *ptr = val;
}

template <>
inline __device__ void write_helper<char4>(char4* ptr, char4 val) {
    int32_t* rel_ptr = (int32_t*)ptr;
    *rel_ptr = *(int32_t*)(&val);
}

template <>
inline __device__ void write_helper<array_wrapper<uint8_t, 32>>(
        array_wrapper<uint8_t, 32>* ptr, array_wrapper<uint8_t, 32> val) {
    uint4 const* data = reinterpret_cast<uint4 const*>(&val);
    void* ptr_ = reinterpret_cast<void*>(ptr);
    asm volatile(
            "{\n"
            " st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
            " st.global.v4.u32 [%5], {%6, %7, %8, %9};\n"
            "}\n"
            :
            : "l"(ptr_), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z),
              "r"(data[0].w), "l"(((uint8_t*)ptr_) + 16), "r"(data[1].x),
              "r"(data[1].y), "r"(data[1].z), "r"(data[1].w));
}

template <bool with_pad, int pack_w, int pack_c, bool same_scale, bool all_pad,
          typename SrcType, typename DstType, typename DnnSrcType,
          typename DnnDstType>
struct RelayoutKern {
    using InnerDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   pack_w>::InnerDtype;
    using DstDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   pack_w>::DstDtype;
    static inline __device__ void write(DstDtype* dst_ptr,
                                        DstDtype (&dst_width)[pack_w]) {
        DstDtype* dst_inner_ptr = (DstDtype*)dst_ptr;
#pragma unroll
        for (int iw_idx = 0; iw_idx < pack_w; ++iw_idx) {
            write_helper(dst_inner_ptr + iw_idx, dst_width[iw_idx]);
        }
    }

    static inline __device__ void read(const SrcType* src_ptr,
                                       InnerDtype (&read_channel)[pack_c],
                                       const int ic_stride) {
#pragma unroll
        for (int ic_idx = 0; ic_idx < pack_c; ++ic_idx) {
            read_channel[ic_idx] = *(InnerDtype*)(src_ptr + ic_idx * ic_stride);
        }
    }

    static inline __device__ void read_with_pad(
            const SrcType* src_ptr, InnerDtype (&read_channel)[pack_c],
            const int ic_stride, const int remain_ic,
            const InnerDtype zero_point) {
#pragma unroll
        for (int ic_idx = 0; ic_idx < pack_c; ++ic_idx) {
            read_channel[ic_idx] =
                    ic_idx < remain_ic
                            ? *(InnerDtype*)(src_ptr + ic_idx * ic_stride)
                            : zero_point;
        }
    }

    static inline __device__ void fake_read(const SrcType* src_ptr,
                                            InnerDtype (&read_channel)[pack_c],
                                            const int ic_stride,
                                            const int remain_ic,
                                            const InnerDtype zero_point) {
#pragma unroll
        for (int ic_idx = 0; ic_idx < pack_c; ++ic_idx) {
            read_channel[ic_idx] = zero_point;
        }
    }

    static inline __device__ void core_relayout_kern(
            const SrcType* src, DstType* dst, const int ic_stride,
            const int remain_ic,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const uint8_t zero_point) {
        InnerDtype read_channel[pack_c];
        if (all_pad) {
            const InnerDtype zero_pad = make_zero_pad<InnerDtype>(zero_point);
            fake_read(src, read_channel, ic_stride, remain_ic, zero_pad);
        } else {
            if (with_pad) {
                const InnerDtype zero_pad =
                        make_zero_pad<InnerDtype>(zero_point);
                read_with_pad(src, read_channel, ic_stride, remain_ic,
                              zero_pad);
            } else {
                read(src, read_channel, ic_stride);
            }
        }
        DstDtype dst_width[pack_w];
        Translayout<pack_w, pack_c, SrcType, DnnSrcType, DnnDstType,
                    same_scale>::trans(dst_width, read_channel, post_process,
                                       zero_point);
        write(reinterpret_cast<DstDtype*>(dst), dst_width);
    }
};

template <int pack_w, int pack_c, bool same_scale, typename SrcType,
          typename DstType, typename DnnSrcType, typename DnnDstType,
          int size_nbits = 8>
__global__ void kern_nchw_nchwx(
        const SrcType* src, DstType* dst, int in_n, int ic, int ihw,
        int n_stride_src, int ic_stride, int n_stride_dst, int oc_stride,
        CudaPostProcess<DnnSrcType, DnnDstType, same_scale> post_process,
        const uint8_t zero_point, const int group, const int ocpg) {
    static constexpr int size_src_type = sizeof(SrcType);
    static constexpr int size_dst_type = sizeof(DstType);
#ifndef MEGDNN_COMMA
#define MEGDNN_COMMA ,
#endif
    MEGDNN_STATIC_ASSERT(std::is_same<SrcType MEGDNN_COMMA DstType>::value,
                         "Currently this kernel only support accessing tensor "
                         "src and dst in same data type.");
    n_stride_src /= size_src_type;
    ic_stride /= size_src_type;
    n_stride_dst /= size_dst_type;
    oc_stride /= size_dst_type;

    const int n_idx = blockIdx.y;
    const int ihw_block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ihw_offset =
            ihw_block_idx * pack_w;
    const int ihw_offset_in_type =
            ihw_offset * size_nbits / (8 * size_src_type);
    if (ihw_offset < ihw) {
        const int src_offset_base = n_idx * n_stride_src + ihw_offset_in_type;
        const int dst_offset_base =
                n_idx * n_stride_dst + ihw_offset_in_type * pack_c;
        if (n_idx < in_n) {
            const int icpg = ic / group;
            const int ic_block = icpg / pack_c;
            const int remain_ic = icpg % pack_c;
            const int src_group_stride = icpg * ic_stride;
            const int dst_group_stride = (ocpg / pack_c) * oc_stride;
            for (int g_idx = 0; g_idx < group; ++g_idx) {
                const int src_offset =
                        src_offset_base + g_idx * src_group_stride;
                const int dst_offset =
                        dst_offset_base + g_idx * dst_group_stride;
                for (int ic_blk_idx = 0; ic_blk_idx < ic_block; ++ic_blk_idx) {
                    const int ic_offset = ic_blk_idx * pack_c * ic_stride;
                    const int oc_offset = ic_blk_idx * oc_stride;
                    RelayoutKern<false, pack_w, pack_c, same_scale, false,
                                 SrcType, DstType, DnnSrcType, DnnDstType>::
                            core_relayout_kern(src + src_offset + ic_offset,
                                               dst + dst_offset + oc_offset,
                                               ic_stride, remain_ic,
                                               post_process, zero_point);
                }

                if (remain_ic > 0) {
                    const int ic_offset = ic_block * pack_c * ic_stride;
                    const int oc_offset = ic_block * oc_stride;
                    RelayoutKern<true, pack_w, pack_c, same_scale, false,
                                 SrcType, DstType, DnnSrcType, DnnDstType>::
                            core_relayout_kern(src + src_offset + ic_offset,
                                               dst + dst_offset + oc_offset,
                                               ic_stride, remain_ic,
                                               post_process, zero_point);
                }
            }
        } else {
            //! pad n
            const int ic_full_block = group * ocpg / pack_c;
            for (int ic_blk_idx = 0; ic_blk_idx < ic_full_block; ++ic_blk_idx) {
                RelayoutKern<false, pack_w, pack_c, same_scale, true, SrcType,
                             DstType, DnnSrcType, DnnDstType>::
                        core_relayout_kern(src + src_offset_base,
                                           dst + dst_offset_base, ic_stride, 0,
                                           post_process, zero_point);
            }
        }
    }
}

__global__ void kern_nchw4_nchw(const int8_t* src, int8_t* dst, int n, int ic,
                                int oc, int oh, int ow, int group) {
    constexpr int pack_w = 1;
    constexpr int pack_ic = 4;
    const int n_idx = blockIdx.y;
    const int hw_block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int hw_offset = hw_block_idx * pack_w;
    const int hw = oh * ow;
    const int n_stride_src = ic * hw;
    const int n_stride_dst = oc * hw;
    const int c_stride = hw;

    if (hw_offset < hw) {
        const int icpg = ic / group;
        const int ocpg = oc / group;
        const int src_group_stride = icpg * c_stride;
        const int dst_group_stride = ocpg * c_stride;
        for (int g_idx = 0; g_idx < group; ++g_idx) {
            const int oc_block = ocpg / pack_ic;
            const int remain_oc = ocpg % pack_ic;
            const int src_offset_base = n_idx * n_stride_src +
                                        g_idx * src_group_stride +
                                        hw_offset * pack_ic;
            const int dst_offset_base =
                    n_idx * n_stride_dst + g_idx * dst_group_stride + hw_offset;

            for (int ic_blk_idx = 0; ic_blk_idx < oc_block; ++ic_blk_idx) {
                const int oc_offset = ic_blk_idx * pack_ic * c_stride;
                char4 temp = *(char4*)(src + src_offset_base + oc_offset);
                dst[dst_offset_base + oc_offset + 0 * c_stride] = temp.x;
                dst[dst_offset_base + oc_offset + 1 * c_stride] = temp.y;
                dst[dst_offset_base + oc_offset + 2 * c_stride] = temp.z;
                dst[dst_offset_base + oc_offset + 3 * c_stride] = temp.w;
            }

            if (remain_oc > 0) {
                const int oc_offset = oc_block * pack_ic * c_stride;
                char4 temp = *(char4*)(src + src_offset_base + oc_offset);
                dst[dst_offset_base + oc_offset + 0 * c_stride] = temp.x;
                if (remain_oc > 1) {
                    dst[dst_offset_base + oc_offset + 1 * c_stride] = temp.y;
                }
                if (remain_oc > 2) {
                    dst[dst_offset_base + oc_offset + 2 * c_stride] = temp.z;
                }
            }
        }
    }
}

__global__ void kern_nchw_nchw4_weight(
        const char* src, char* dst, int in_oc, int ic, int ihw,
        int oc_stride_src, int ic_stride, int oc_stride_dst,
        int group_stride_src, int group_stride_dst, const char zero_point,
        CudaPostProcess<dtype::QuantizedS8, dtype::QuantizedS8, true>
                post_process) {
    typedef char SrcType;
    typedef char DstType;
    typedef dtype::QuantizedS8 DnnSrcType;
    typedef dtype::QuantizedS8 DnnDstType;
    constexpr int pack_c = 4;
    constexpr int pack_w = 1;
    constexpr bool same_scale = true;

    const int group_idx = blockIdx.z;
    const int oc_idx = blockIdx.y;
    const int ihw_block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ihw_offset = ihw_block_idx * pack_w;

    if (ihw_offset < ihw) {
        const int ic_block = ic / pack_c;
        const int remain_ic = ic % pack_c;
        const int src_offset_base = group_idx * group_stride_src +
                                    oc_idx * oc_stride_src + ihw_offset;
        const int dst_offset_base = group_idx * group_stride_dst +
                                    oc_idx * oc_stride_dst +
                                    ihw_offset * pack_c;
        if (oc_idx < in_oc) {
            for (int ic_blk_idx = 0; ic_blk_idx < ic_block; ++ic_blk_idx) {
                const int ic_offset = ic_blk_idx * pack_c * ic_stride;
                RelayoutKern<false, pack_w, pack_c, same_scale, false, SrcType,
                             DstType, DnnSrcType, DnnDstType>::
                        core_relayout_kern(src + src_offset_base + ic_offset,
                                           dst + dst_offset_base + ic_offset,
                                           ic_stride, remain_ic, post_process,
                                           zero_point);
            }

            if (remain_ic > 0) {
                const int ic_offset = ic_block * pack_c * ic_stride;
                RelayoutKern<true, pack_w, pack_c, same_scale, false, SrcType,
                             DstType, DnnSrcType, DnnDstType>::
                        core_relayout_kern(src + src_offset_base + ic_offset,
                                           dst + dst_offset_base + ic_offset,
                                           ic_stride, remain_ic, post_process,
                                           zero_point);
            }
        } else {
            //! pad oc per group
            const int ic_full_block = (ic + pack_c - 1) / pack_c;
            for (int ic_blk_idx = 0; ic_blk_idx < ic_full_block; ++ic_blk_idx) {
                const int ic_offset = ic_blk_idx * pack_c * ic_stride;
                RelayoutKern<false, pack_w, pack_c, same_scale, true, SrcType,
                             DstType, DnnSrcType, DnnDstType>::
                        core_relayout_kern(src + src_offset_base + ic_offset,
                                           dst + dst_offset_base + ic_offset,
                                           ic_stride, remain_ic, post_process,
                                           zero_point);
            }
        }
    }
}

template <typename Type_, int pack_size_, int chan_blk_, int width_,
          int size_nbits_>
class TensorIteratorOverChannel {
public:
    using Type = Type_;
    static constexpr int pack_size = pack_size_;
    static constexpr int chan_blk = chan_blk_;
    static constexpr int width = width_;
    static constexpr int size_nbits = size_nbits_;
    static constexpr int elements_in_type =
            chan_blk * width * size_nbits / (8 * sizeof(Type));
    static constexpr int lane_size_in_type =
            (width * pack_size * size_nbits) / (8 * sizeof(Type));
    static constexpr int pack_size_in_type =
            (pack_size * size_nbits) >= (8 * sizeof(Type))
                    ? (pack_size * size_nbits / (8 * sizeof(Type)))
                    : (width * pack_size * size_nbits / (8 * sizeof(Type)));
    static constexpr int pack_size_in_byte = pack_size_in_type * sizeof(Type);
    using AccessType = array_wrapper<Type, pack_size_in_type>;
    using Fragment = array_wrapper<Type, elements_in_type>;

    MEGDNN_HOST TensorIteratorOverChannel()
            : pointer{nullptr}, chan_stride_in_elements{0}, channel{0} {}
    MEGDNN_HOST TensorIteratorOverChannel(Type* pointer_,
                                          int chan_stride_in_elements_,
                                          int channel_, int, int)
            : pointer{pointer_},
              chan_stride_in_elements{chan_stride_in_elements_},
              channel{channel_} {}

    MEGDNN_DEVICE __forceinline__ void initialize(int c_idx, int hw_idx) {
        pointer += (c_idx / pack_size) * chan_stride_in_elements +
                   hw_idx * pack_size * size_nbits / (8 * sizeof(Type));
        channel -= c_idx;
    }

    MEGDNN_DEVICE __forceinline__ void add_pointer_offset(
            size_t offset_in_type) {
        pointer += offset_in_type;
    }

    MEGDNN_DEVICE __forceinline__ void load(Fragment& frag, int zero_point) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        Type* pointer_ = pointer;
#pragma unroll
        for (int i = 0; i < chan_blk; i += pack_size) {
#pragma unroll
            for (int j = 0; j < lane_size_in_type / pack_size_in_type; j++) {
                int frag_idx = i / pack_size *
                                       (lane_size_in_type / pack_size_in_type) +
                               j;
                bool guard = i < channel;
                memory::global_load<AccessType, pack_size_in_byte>(
                        frag_ptr[frag_idx],
                        reinterpret_cast<void*>(pointer_ +
                                                j * pack_size_in_type),
                        guard, zero_point);
            }
            pointer_ += chan_stride_in_elements;
        }
    }

    MEGDNN_DEVICE __forceinline__ void store(const Fragment& frag) {
        const AccessType* frag_ptr = reinterpret_cast<const AccessType*>(&frag);
        Type* pointer_ = pointer;
#pragma unroll
        for (int i = 0; i < chan_blk; i += pack_size) {
#pragma unroll
            for (int j = 0; j < lane_size_in_type / pack_size_in_type; j++) {
                int frag_idx = i / pack_size *
                                       (lane_size_in_type / pack_size_in_type) +
                               j;
                bool guard = i < channel;
                memory::global_store<AccessType, pack_size_in_byte>(
                        frag_ptr[frag_idx],
                        reinterpret_cast<void*>(pointer_ +
                                                j * pack_size_in_type),
                        guard);
            }
            pointer_ += chan_stride_in_elements;
        }
    }


    MEGDNN_DEVICE __forceinline__ void advance() {
        pointer += (chan_blk / pack_size) * chan_stride_in_elements;
        channel -= chan_blk;
    }

private:
    Type* pointer;
    int chan_stride_in_elements;
    int channel;
};

template <typename Type_, int pack_size_, int chan_blk_, int width_,
          int size_nbits_>
class MaskedTensorIteratorOverChannel {
public:
    using Type = Type_;
    static constexpr int pack_size = pack_size_;
    static constexpr int chan_blk = chan_blk_;
    static constexpr int width = width_;
    static constexpr int size_nbits = size_nbits_;
    static constexpr int elements_in_type =
            chan_blk * width * size_nbits / (8 * sizeof(Type));
    static constexpr int lane_size_in_type =
            (width * pack_size * size_nbits) / (8 * sizeof(Type));
    static constexpr int pack_size_in_type =
            (pack_size * size_nbits) >= (8 * sizeof(Type))
                    ? (pack_size * size_nbits / (8 * sizeof(Type)))
                    : (width * pack_size * size_nbits / (8 * sizeof(Type)));
    static constexpr int pack_size_in_byte = pack_size_in_type * sizeof(Type);
    static constexpr int accesses = elements_in_type / pack_size_in_type;
    static constexpr int mask_size = (accesses + 32 - 1) / 32;
    using AccessType = array_wrapper<Type, pack_size_in_type>;
    using Fragment = array_wrapper<Type, elements_in_type>;

    MEGDNN_HOST MaskedTensorIteratorOverChannel()
            : pointer{nullptr},
              chan_stride_in_elements{0},
              channel{0} {}
    MEGDNN_HOST MaskedTensorIteratorOverChannel(
            Type* pointer_, int chan_stride_in_elements_, int channel_,
            int bound_, int div_)
            : pointer{pointer_},
              chan_stride_in_elements{chan_stride_in_elements_},
              channel{channel_},
              bound{bound_},
              div{uint32_t(div_)} {}

    MEGDNN_DEVICE __forceinline__ void initialize(int c_idx, int hw_idx) {
        pointer += (c_idx / pack_size) * chan_stride_in_elements;
        channel -= c_idx;
#pragma unroll
        for (int i = 0; i < mask_size; ++i) {
            mask[i] = 0;
        }
#pragma unroll
        for (int i = 0; i < chan_blk; i += pack_size) {
#pragma unroll
            for (int j = 0; j < lane_size_in_type / pack_size_in_type; j++) {
                int offset = hw_idx + j;
                int h = (int)((uint32_t)(offset) / div);
                int w = (int)((uint32_t)(offset) % div);
                bool guard = (i < channel) && (w < bound);
                int index = (i / pack_size) *
                                    (lane_size_in_type / pack_size_in_type) +
                            j;
                int mask_index = (index >> 5);
                int mask_shift = (index & 0x1f);
                mask[mask_index] |= (guard << mask_shift);
                stride[j] = (h * bound + w) * pack_size * size_nbits /
                            (8 * sizeof(Type));
            }
        }
    }

    MEGDNN_DEVICE __forceinline__ void add_pointer_offset(size_t offset_in_type) {
        pointer += offset_in_type;
    }

    MEGDNN_DEVICE __forceinline__ void load(Fragment& frag, int zero_point) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        Type* pointer_ = pointer;
#pragma unroll
        for (int i = 0; i < chan_blk; i += pack_size) {
#pragma unroll
            for (int j = 0; j < lane_size_in_type / pack_size_in_type; j++) {
                int frag_idx = i / pack_size *
                                       (lane_size_in_type / pack_size_in_type) +
                               j;
                int mask_index = (frag_idx >> 5);
                int mask_shift = (frag_idx & 0x1f);
                bool guard = (mask[mask_index] & (1 << mask_shift));
                memory::global_load<AccessType, pack_size_in_byte>(
                        frag_ptr[frag_idx],
                        reinterpret_cast<void*>(pointer_ + stride[j]), guard,
                        zero_point);
            }
            pointer_ += chan_stride_in_elements;
        }
    }

    MEGDNN_DEVICE __forceinline__ void store(const Fragment& frag) {
        const AccessType* frag_ptr = reinterpret_cast<const AccessType*>(&frag);
        Type* pointer_ = pointer;
#pragma unroll
        for (int i = 0; i < chan_blk; i += pack_size) {
#pragma unroll
            for (int j = 0; j < lane_size_in_type / pack_size_in_type; j++) {
                int frag_idx = i / pack_size *
                                       (lane_size_in_type / pack_size_in_type) +
                               j;
                int mask_index = (frag_idx >> 5);
                int mask_shift = (frag_idx & 0x1f);
                bool guard = (mask[mask_index] & (1 << mask_shift));
                memory::global_store<AccessType, pack_size_in_byte>(
                        frag_ptr[frag_idx],
                        reinterpret_cast<void*>(pointer_ + stride[j]), guard);
            }
            pointer_ += chan_stride_in_elements;
        }
    }

    MEGDNN_DEVICE __forceinline__ void advance() {
        pointer += (chan_blk / pack_size) * chan_stride_in_elements;
        channel -= chan_blk;
    }

private:
    Type* pointer;
    int chan_stride_in_elements;
    int channel;
    int bound;
    Uint32Fastdiv div;
    uint32_t mask[mask_size];
    size_t stride[lane_size_in_type / pack_size_in_type];
};

template <bool padding_, typename Type_, int pack_size_, int chan_blk_,
          int width_, int size_nbits_>
struct TensorIteratorPolicy;
template <typename Type_, int pack_size_, int chan_blk_, int width_,
          int size_nbits_>
struct TensorIteratorPolicy<true, Type_, pack_size_, chan_blk_, width_,
                            size_nbits_> {
    using TensorIterator =
            MaskedTensorIteratorOverChannel<Type_, pack_size_, chan_blk_,
                                            width_, size_nbits_>;
};
template <typename Type_, int pack_size_, int chan_blk_, int width_,
          int size_nbits_>
struct TensorIteratorPolicy<false, Type_, pack_size_, chan_blk_, width_,
                            size_nbits_> {
    using TensorIterator =
            TensorIteratorOverChannel<Type_, pack_size_, chan_blk_, width_,
                                      size_nbits_>;
};

template <typename SrcIterator_, typename DstIterator_, typename Transpose_,
          typename CudaPostProcess_>
struct RelayoutProblem {
    using SrcIterator = SrcIterator_;
    using DstIterator = DstIterator_;
    using Transpose = Transpose_; 
    using CudaPostProcess = CudaPostProcess_;
    MEGDNN_STATIC_ASSERT(SrcIterator::chan_blk == DstIterator::chan_blk,
                         "channel block mismatch");
    MEGDNN_STATIC_ASSERT(SrcIterator::width == DstIterator::width,
                         "width block mismatch");
    MEGDNN_STATIC_ASSERT(SrcIterator::size_nbits == DstIterator::size_nbits,
                         "size in bits of elements mismatch");
    static constexpr int pack_chan = SrcIterator::chan_blk;
    static constexpr int pack_width = SrcIterator::width;
    using DnnSrcType = typename CudaPostProcess::SrcType;
    using DnnDstType = typename CudaPostProcess::DstType;
    struct Param {
        SrcIterator src_iterator;
        DstIterator dst_iterator;
        CudaPostProcess post_process;
        int n_stride_src;
        int n_stride_dst;
        int batch_size;
        int channels;
        int hw;
        int zero_point;
        MEGDNN_HOST MEGDNN_DEVICE Param(SrcIterator src_iterator_,
                                        DstIterator dst_iterator_,
                                        CudaPostProcess post_process_,
                                        int n_stride_src_, int n_stride_dst_,
                                        int batch_size_, int channels_, int hw_,
                                        int zero_point_)
                : src_iterator{src_iterator_},
                  dst_iterator{dst_iterator_},
                  post_process{post_process_},
                  n_stride_src{n_stride_src_},
                  n_stride_dst{n_stride_dst_},
                  batch_size{batch_size_},
                  channels{channels_},
                  hw{hw_},
                  zero_point{zero_point_} {}
    };
};

template <typename RelayoutProblem_>
__global__ void relayout_kern(typename RelayoutProblem_::Param param) {
    using SrcIterator = typename RelayoutProblem_::SrcIterator;
    using DstIterator = typename RelayoutProblem_::DstIterator;
    static constexpr int pack_chan = RelayoutProblem_::pack_chan;
    static constexpr int pack_width = RelayoutProblem_::pack_width;
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_offset = thread_idx * pack_width;
    const int hw_idx = (thread_offset % param.hw);
    const int nc_blks = thread_offset / param.hw;
    const int c_blks = (param.channels + pack_chan - 1) / pack_chan;
    const int n_idx = nc_blks / c_blks;
    const int c_blk_idx = nc_blks % c_blks;
    const int c_idx = c_blk_idx * pack_chan;
    if (n_idx < param.batch_size) {
        const int src_offset = n_idx * param.n_stride_src;
        const int dst_offset = n_idx * param.n_stride_dst;
        param.src_iterator.add_pointer_offset(src_offset);
        param.dst_iterator.add_pointer_offset(dst_offset);
        param.src_iterator.initialize(c_idx, hw_idx);
        param.dst_iterator.initialize(c_idx, hw_idx);
        typename SrcIterator::Fragment src_frag;
        typename DstIterator::Fragment dst_frag;
        int zp = make_zero<SrcIterator::size_nbits>(param.zero_point);
        param.src_iterator.load(src_frag, zp);
        RelayoutProblem_::Transpose::trans(
                reinterpret_cast<typename SrcIterator::Fragment&>(dst_frag),
                src_frag, param.post_process);
        param.dst_iterator.store(dst_frag);
    }
}
}  // namespace

void relayout_format::relayout_format_cuda_nchw_nchwx(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale, const float dst_scale,
        const uint8_t src_zero_point, const uint8_t dst_zero_point, int group) {
    auto&& stype = src.layout.dtype;
    auto&& dtype = dst.layout.dtype;
    auto& src_layout = src.layout;
    auto& dst_layout = dst.layout;
    // check pack size
    int pack_oc = std::numeric_limits<int>::min();
#define DEF(_pack_oc, _src_type, _dst_type)             \
    if (stype.enumv().ev == DTypeEnum::Ev::_src_type && \
        dtype.enumv().ev == DTypeEnum::Ev::_dst_type) { \
        pack_oc = _pack_oc;                             \
    }
    // clang-format off
    DEF(64, QuantizedS4, QuantizedS4)
    DEF(64, Quantized4Asymm, Quantized4Asymm)
    DEF(4, QuantizedS8, QuantizedS8)
    DEF(4, Uint8, QuantizedS8)
    DEF(4, Quantized8Asymm, QuantizedS8)
    DEF(4, QuantizedS32, QuantizedS32)
    // clang-format on
    megdnn_assert(pack_oc == 4 || pack_oc == 64,
                  "Unsupport pack size(pack_oc:%d, src:%s, dst:%s)", pack_oc,
                  stype.name(), dtype.name());
#undef DEF
    // no padding
    if (stype.enumv().ev != DTypeEnum::Ev::QuantizedS4 &&
        stype.enumv().ev != DTypeEnum::Ev::Quantized4Asymm) {
        const int in_n = src.layout[0];
        const int out_n = dst.layout[0];
        const int ic = src.layout[1];
        const int h = src.layout[2];
        const int w = src.layout[3];
        const int oc = dst.layout[1] * pack_oc;
        const int hw = h * w;
        const int ocpg = oc / group;
        // stride in byte
        const int n_stride_src = src_layout.dtype.size(src_layout.stride[0]);
        const int ic_stride = src_layout.dtype.size(src_layout.stride[1]);
        const int n_stride_dst = dst_layout.dtype.size(dst_layout.stride[0]);
        const int oc_stride = dst_layout.dtype.size(dst_layout.stride[1]);

        bool same_scale = src_scale == dst_scale;
#define DISPATCH_RAW(_same_scale, _pack_w, _pack_oc, _src_type, _dst_type,   \
                     _src_c_type, _dst_c_type, _size_nbits)                  \
    if (same_scale == _same_scale && hw % _pack_w == 0 &&                    \
        stype.enumv().ev == DTypeEnum::Ev::_src_type &&                      \
        dtype.enumv().ev == DTypeEnum::Ev::_dst_type) {                      \
        auto kernel =                                                        \
                kern_nchw_nchwx<_pack_w, _pack_oc, _same_scale, _src_c_type, \
                                _dst_c_type, dtype::_src_type,               \
                                dtype::_dst_type, _size_nbits>;              \
        int nr_threads = query_blocksize_for_kernel(kernel);                 \
        const dim3 block_dim(DIVUP(hw, nr_threads* _pack_w), out_n);         \
        const dim3 thread_dim(nr_threads);                                   \
        return kernel<<<block_dim, thread_dim, 0, stream>>>(                 \
                (_src_c_type*)src.raw_ptr, (_dst_c_type*)dst.raw_ptr, in_n,  \
                ic, hw, n_stride_src, ic_stride, n_stride_dst, oc_stride,    \
                CudaPostProcess<dtype::_src_type, dtype::_dst_type,          \
                                _same_scale>(src_scale, src_zero_point,      \
                                             dst_scale, dst_zero_point),     \
                src_zero_point, group, ocpg);                                \
    }
#define DISPATCH_INT(_src_type, _dst_type)                         \
    DISPATCH_RAW(true, 4, 4, _src_type, _dst_type, int, int, 32);  \
    DISPATCH_RAW(false, 4, 4, _src_type, _dst_type, int, int, 32); \
    DISPATCH_RAW(true, 1, 4, _src_type, _dst_type, int, int, 32);  \
    DISPATCH_RAW(false, 1, 4, _src_type, _dst_type, int, int, 32);
#define DISPATCH_BYTE(_src_type, _dst_type)                         \
    DISPATCH_RAW(true, 4, 4, _src_type, _dst_type, char, char, 8);  \
    DISPATCH_RAW(false, 4, 4, _src_type, _dst_type, char, char, 8); \
    DISPATCH_RAW(true, 1, 4, _src_type, _dst_type, char, char, 8);  \
    DISPATCH_RAW(false, 1, 4, _src_type, _dst_type, char, char, 8);
        DISPATCH_INT(QuantizedS32, QuantizedS32);
        DISPATCH_BYTE(Uint8, QuantizedS8);
        DISPATCH_BYTE(Quantized8Asymm, QuantizedS8);
        DISPATCH_BYTE(QuantizedS8, QuantizedS8);
#undef DISPATCH_BYTE
#undef DISPATCH_INT
#undef DISPATCH_RAW
        megdnn_assert(
                false,
                "Unsupported data type(src:%s, dst:%s) or image size(%dx%d).",
                stype.name(), dtype.name(), h, w);
    } else {
        megdnn_assert(src_layout.dtype.is_low_bit());
        int n = src.layout[0];
        int ic = src.layout[1];
        int oc = dst.layout[1] * pack_oc;
        int h = src.layout[2];
        // align to byte
        int w = src.layout[3];
        int w_pad = DIVUP(w, 2) * 2;
        int hw  = h * w_pad;
        int n_stride_src = src_layout.stride[0];
        int ic_stride = src_layout.stride[1];
        int n_stride_dst = dst_layout.stride[0];
        int oc_stride = dst_layout.stride[1];
        int problem_size = n * (oc / pack_oc) * hw;
        bool same_scale = src_scale == dst_scale;
        bool padding = w % 2 != 0;
#define DISPATCH_RAW(_padding, _same_scale, _pack_w, _pack_oc, _src_type,      \
                     _dst_type, _src_c_type, _dst_c_type, _size_nbits)         \
    if (padding == _padding && same_scale == _same_scale &&                    \
        hw % _pack_w == 0 && stype.enumv().ev == DTypeEnum::Ev::_src_type &&   \
        dtype.enumv().ev == DTypeEnum::Ev::_dst_type) {                        \
        using InnerDtype_ = typename DTypeRWHelper<                            \
                typename DTypeTrait<dtype::_src_type>::ctype,                  \
                _pack_w>::InnerDtype;                                          \
        using SrcIterator_ =                                                   \
                TensorIteratorOverChannel<InnerDtype_, 1, _pack_oc, _pack_w,   \
                                          _size_nbits>;                        \
        using DstIterator_ =                                                   \
                typename TensorIteratorPolicy<_padding, _dst_c_type, _pack_oc, \
                                              _pack_oc, _pack_w,               \
                                              _size_nbits>::TensorIterator;    \
        using CudaPostProcess_ =                                               \
                CudaPostProcess<dtype::_src_type, dtype::_dst_type,            \
                                _same_scale>;                                  \
        using Transpose_ =                                                     \
                Translayout<_pack_w, _pack_oc, _src_c_type, dtype::_src_type,  \
                            dtype::_dst_type, _same_scale>;                    \
        using RelayoutProblem_ =                                               \
                RelayoutProblem<SrcIterator_, DstIterator_, Transpose_,        \
                                CudaPostProcess_>;                             \
        n_stride_src = n_stride_src * _size_nbits / (8 * sizeof(InnerDtype_)); \
        ic_stride = ic_stride * _size_nbits / (8 * sizeof(InnerDtype_));       \
        n_stride_dst = n_stride_dst * _size_nbits / (8 * sizeof(_dst_c_type)); \
        oc_stride = oc_stride * _size_nbits / (8 * sizeof(_dst_c_type));       \
        typename RelayoutProblem_::Param param{                                \
                SrcIterator_{(InnerDtype_*)src.raw_ptr, ic_stride, ic, w,      \
                             w_pad},                                           \
                DstIterator_{(_dst_c_type*)dst.raw_ptr, oc_stride, oc, w,      \
                             w_pad},                                           \
                CudaPostProcess_{src_scale, src_zero_point, dst_scale,         \
                                 dst_zero_point},                              \
                n_stride_src,                                                  \
                n_stride_dst,                                                  \
                n,                                                             \
                oc,                                                            \
                hw,                                                            \
                src_zero_point};                                               \
        auto kernel = relayout_kern<RelayoutProblem_>;                         \
        int nr_threads = query_blocksize_for_kernel(kernel);                   \
        nr_threads = std::min(nr_threads, DIVUP(problem_size, _pack_w));       \
        const dim3 block_dim(DIVUP(problem_size, nr_threads* _pack_w));        \
        const dim3 thread_dim(nr_threads);                                     \
        return kernel<<<block_dim, thread_dim, 0, stream>>>(param);            \
    }
#define DISPATCH_4BITS(_src_type, _dst_type)                                \
    DISPATCH_RAW(true, true, 8, 64, _src_type, _dst_type, char, char, 4);   \
    DISPATCH_RAW(true, false, 8, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(true, true, 2, 64, _src_type, _dst_type, char, char, 4);   \
    DISPATCH_RAW(true, false, 2, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, true, 8, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, false, 8, 64, _src_type, _dst_type, char, char, 4); \
    DISPATCH_RAW(false, true, 2, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, false, 2, 64, _src_type, _dst_type, char, char, 4);
        DISPATCH_4BITS(QuantizedS4, QuantizedS4);
        DISPATCH_4BITS(Quantized4Asymm, Quantized4Asymm);
#undef DISPATCH_4BITS
#undef DISPATCH_RAW
        megdnn_assert(
                false,
                "Unsupported data type(src:%s, dst:%s) or image size(%dx%d).",
                stype.name(), dtype.name(), h, w);
    }
}

bool relayout_format::relayout_format_cuda_usable(
        const TensorLayout& src_layout, const TensorLayout& dst_layout) {
    bool is_all_continue =
            src_layout.is_contiguous() && dst_layout.is_contiguous();
    bool is_all_int32 =
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS32 &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS32);
    bool is_all_int8 =
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::Uint8 &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8) ||
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::Quantized8Asymm &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8) ||
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8 &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8);
    bool is_all_int4 =
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS4 &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS4) ||
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::Quantized4Asymm &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::Quantized4Asymm);
    return is_all_continue && (is_all_int32 || is_all_int8 || is_all_int4);
}

void relayout_format::relayout_format_cuda_nchwx_nchw(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale, const float dst_scale,
        const uint8_t src_zero_point, const uint8_t dst_zero_point) {
    auto&& stype = src.layout.dtype;
    auto&& dtype = dst.layout.dtype;
    auto& src_layout = src.layout;
    auto& dst_layout = dst.layout;
    // check pack size
    int pack_ic = std::numeric_limits<int>::min();
#define DEF(_pack_ic, _src_type, _dst_type)             \
    if (stype.enumv().ev == DTypeEnum::Ev::_src_type && \
        dtype.enumv().ev == DTypeEnum::Ev::_dst_type) { \
        pack_ic = _pack_ic;                             \
    }
    // clang-format off
    DEF(64, QuantizedS4, QuantizedS4)
    DEF(64, Quantized4Asymm, Quantized4Asymm)
    // clang-format on
    megdnn_assert(pack_ic == 64, "Unsupport pack size(pack_ic:%d)", pack_ic);
#undef DEF
    int n = src.layout[0];
    int ic = src.layout[1] * pack_ic;
    int h = src.layout[2];
    // align to byte
    int w = src.layout[3];
    int w_pad = DIVUP(w, 2) * 2;
    int hw = h * w_pad;
    int n_stride_src = src_layout.stride[0];
    int ic_stride = src_layout.stride[1];
    int n_stride_dst = dst_layout.stride[0];
    int oc_stride = dst_layout.stride[1];
    int problem_size = n * (ic / pack_ic) * hw;
    int oc = dst.layout[1];

    bool same_scale = src_scale == dst_scale;
    bool padding = w % 2 != 0;
#define DISPATCH_RAW(_padding, _same_scale, _pack_w, _pack_oc, _src_type,      \
                     _dst_type, _src_c_type, _dst_c_type, _size_nbits)         \
    if (padding == _padding && same_scale == _same_scale &&                    \
        hw % _pack_w == 0 && stype.enumv().ev == DTypeEnum::Ev::_src_type &&   \
        dtype.enumv().ev == DTypeEnum::Ev::_dst_type) {                        \
        using SrcIterator_ =                                                   \
                typename TensorIteratorPolicy<_padding, _src_c_type, _pack_oc, \
                                              _pack_oc, _pack_w,               \
                                              _size_nbits>::TensorIterator;    \
        using InnerDtype_ = typename DTypeRWHelper<                            \
                typename DTypeTrait<dtype::_src_type>::ctype,                  \
                _pack_w>::InnerDtype;                                          \
        using DstIterator_ =                                                   \
                TensorIteratorOverChannel<InnerDtype_, 1, _pack_oc, _pack_w,   \
                                          _size_nbits>;                        \
        using CudaPostProcess_ =                                               \
                CudaPostProcess<dtype::_src_type, dtype::_dst_type,            \
                                _same_scale>;                                  \
        using Transpose_ =                                                     \
                Translayout<_pack_oc, _pack_w, _src_c_type, dtype::_src_type,  \
                            dtype::_dst_type, _same_scale>;                    \
        using RelayoutProblem_ =                                               \
                RelayoutProblem<SrcIterator_, DstIterator_, Transpose_,        \
                                CudaPostProcess_>;                             \
        n_stride_src = n_stride_src * _size_nbits / (8 * sizeof(_src_c_type)); \
        ic_stride = ic_stride * _size_nbits / (8 * sizeof(_src_c_type));       \
        n_stride_dst = n_stride_dst * _size_nbits / (8 * sizeof(InnerDtype_)); \
        oc_stride = oc_stride * _size_nbits / (8 * sizeof(InnerDtype_));       \
        typename RelayoutProblem_::Param param{                                \
                SrcIterator_{(_src_c_type*)src.raw_ptr, ic_stride, ic, w,      \
                             w_pad},                                           \
                DstIterator_{(InnerDtype_*)dst.raw_ptr, oc_stride, oc, w,      \
                             w_pad},                                           \
                CudaPostProcess_{src_scale, src_zero_point, dst_scale,         \
                                 dst_zero_point},                              \
                n_stride_src,                                                  \
                n_stride_dst,                                                  \
                n,                                                             \
                ic,                                                            \
                hw,                                                            \
                src_zero_point};                                               \
        auto kernel = relayout_kern<RelayoutProblem_>;                         \
        int nr_threads = query_blocksize_for_kernel(kernel);                   \
        nr_threads = std::min(nr_threads, DIVUP(problem_size, _pack_w));       \
        const dim3 block_dim(DIVUP(problem_size, nr_threads* _pack_w));        \
        const dim3 thread_dim(nr_threads);                                     \
        return kernel<<<block_dim, thread_dim, 0, stream>>>(param);            \
    }
#define DISPATCH_4BITS(_src_type, _dst_type)                                \
    DISPATCH_RAW(true, true, 8, 64, _src_type, _dst_type, char, char, 4);   \
    DISPATCH_RAW(true, false, 8, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(true, true, 2, 64, _src_type, _dst_type, char, char, 4);   \
    DISPATCH_RAW(true, false, 2, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, true, 8, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, false, 8, 64, _src_type, _dst_type, char, char, 4); \
    DISPATCH_RAW(false, true, 2, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, false, 2, 64, _src_type, _dst_type, char, char, 4);
    DISPATCH_4BITS(QuantizedS4, QuantizedS4);
    DISPATCH_4BITS(Quantized4Asymm, Quantized4Asymm);
#undef DISPATCH_4BITS
#undef DISPATCH_RAW
    megdnn_assert(false,
                  "Unsupported data type(src:%s, dst:%s) or image size(%dx%d).",
                  stype.name(), dtype.name(), h, w);
}

void relayout_format::relayout_format_cuda_nchw4_nchw(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const int group) {
    constexpr int pack_w = 1;
    const int n = src.layout[0];
    const int ic = src.layout[1] * 4;
    const int h = src.layout[2];
    const int w = src.layout[3];
    const int oc = dst.layout[1];
    const int hw = h * w;
    int nr_threads = query_blocksize_for_kernel(kern_nchw4_nchw);
    const dim3 block_dim(DIVUP(hw, nr_threads * pack_w), n);
    const dim3 thread_dim(nr_threads);
    kern_nchw4_nchw<<<block_dim, thread_dim, 0, stream>>>(
            (int8_t*)src.raw_ptr, (int8_t*)dst.raw_ptr, n, ic, oc, h, w, group);
    after_kernel_launch();
}

void relayout_format::relayout_format_cuda_nchw_nchw4_weight(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream) {
    constexpr int pack_c = 4;
    const bool is_group = src.layout.ndim == 5;
    const int group = is_group ? src.layout[0] : 1;
    const int oc = is_group ? src.layout[1] : src.layout[0];
    const int ic = is_group ? src.layout[2] : src.layout[1];
    const int kh = is_group ? src.layout[3] : src.layout[2];
    const int kw = is_group ? src.layout[4] : src.layout[3];
    const int hw = kh * kw;
    const int oc_round = ROUNDUP(oc, pack_c);
    const int ic_round = ROUNDUP(ic, pack_c);
    const int ic_stride = hw;
    const int oc_stride_src = ic * ic_stride;
    const int oc_stride_dst = ic_round * ic_stride;
    const int group_stride_src = oc * oc_stride_src;
    const int group_stride_dst = oc_round * oc_stride_dst;

    int nr_threads = 32;
    const dim3 block_dim(DIVUP(hw, nr_threads), oc_round, group);
    const dim3 thread_dim(nr_threads);

    kern_nchw_nchw4_weight<<<block_dim, thread_dim, 0, stream>>>(
            (char*)src.raw_ptr, (char*)dst.raw_ptr, oc, ic, hw, oc_stride_src,
            ic_stride, oc_stride_dst, group_stride_src, group_stride_dst, 0,
            {});
    after_kernel_launch();
}
