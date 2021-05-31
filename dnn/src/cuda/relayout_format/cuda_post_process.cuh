/**
 * \file dnn/src/cuda/relayout_format/cuda_post_process.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "src/cuda/relayout_format/relayout_format.cuh"

namespace megdnn {
namespace cuda {
namespace relayout_format {
namespace internal {
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

}  // namespace internal
}  // namespace relayout_format
}  // namespace cuda
}  // namespace megdnn
