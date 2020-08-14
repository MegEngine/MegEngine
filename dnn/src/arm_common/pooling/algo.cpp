/**
 * \file dnn/src/arm_common/pooling/algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/arm_common/pooling/algo.h"
#include "megdnn/opr_param_defs.h"
#include "src/arm_common/pooling/do_max_pooling_3x3_s2x2_int8.h"
#include "src/arm_common/pooling/do_max_pooling_w2x2_s2x2.h"
#include "src/arm_common/pooling/do_max_pooling_w4x4_s2x2.h"
#include "src/arm_common/pooling/do_pooling_2x2_nchw44.h"
#include "src/arm_common/pooling/do_pooling_3x3_nchw44.h"
#include "src/arm_common/pooling/do_pooling_4x4_nchw44.h"
#include "src/arm_common/pooling/do_pooling_5x5_nchw44.h"

#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_pooling)

namespace megdnn {
namespace arm_common {

WorkspaceBundle get_bundle(const PoolingImpl::PoolingKernSizeParam& param) {
    megdnn_assert((param.src_type.category() == DTypeCategory::FLOAT ||
                   param.src_type.enumv() == DTypeEnum::QuantizedS8 ||
                   param.src_type.enumv() == DTypeEnum::Quantized8Asymm ||
                   param.src_type == dtype::Int8{}) &&
                  param.format == param::Pooling::Format::NCHW &&
                  (param.mode == param::Pooling::Mode::MAX ||
                   (param.mode == param::Pooling::Mode::AVERAGE &&
                    param.filter[0] == 3)) &&
                  param.filter[0] == param.filter[1] &&
                  (param.filter[0] == 3 || param.filter[1] == 5) &&
                  param.stride[0] == 2 && param.stride[1] == 2 &&
                  param.isz[0] >= 2 && param.isz[1] >= 2);
    //! max pooling nxn stride 2
    auto IW = param.isz[1];
    auto OW = param.osz[1];

    // In order to process odd size filter,
    // Firstly, Store a row of the input separately by odd and even numbers
    // Then process them, get a row of the outputs
    // We need to store n rows of results
    SmallVector<size_t> needed_mem;
    for (size_t i = 0; i < param.filter[0]; ++i)
        needed_mem.push_back(OW * param.src_type.size());
    needed_mem.push_back((IW + 1) / 2 * param.src_type.size());
    needed_mem.push_back((IW + 1) / 2 * param.src_type.size());
    WorkspaceBundle ws(nullptr, needed_mem, 16);
    return ws;
}

WorkspaceBundle get_bundle_nchw44(
        const PoolingImpl::PoolingKernSizeParam& param) {
    megdnn_assert((param.src_type.enumv() == DTypeEnum::QuantizedS8 ||
                   param.src_type.enumv() == DTypeEnum::Int8) &&
                  (param.format == param::Pooling::Format::NCHW44));
    auto IH = param.isz[0];
    auto IW = param.isz[1];
    auto PH = param.padding[0];
    auto PW = param.padding[1];
    size_t padding_size = 0;
    if ((PH != 0) || (PW != 0)) {
        padding_size = (IW + 2 * PW) * (IH + 2 * PH) * 4 * sizeof(int8_t);
    }
    return WorkspaceBundle(nullptr, {padding_size});
}

const int8_t* handle_padding(const int8_t* src, size_t IH, size_t IW,
                             size_t& IH2, size_t& IW2, size_t PH, size_t PW,
                             const WorkspaceBundle& ws, bool is_max_mode) {
    int8_t* sptr_base = nullptr;
    int8_t padding_value = is_max_mode ? INT8_MIN : 0;
    bool need_pad = ((PH != 0) || (PW != 0)) ? true : false;
    if (need_pad) {
        IH2 = IH + 2 * PH;
        IW2 = IW + 2 * PW;
        sptr_base = static_cast<int8_t*>(ws.get(0));
        memset(sptr_base, padding_value, sizeof(int8_t) * IH2 * IW2 * 4);
        rep(ih, IH) {
            std::memcpy(sptr_base + (ih + PH) * IW2 * 4 + PW * 4,
                        src + ih * IW * 4, sizeof(int8_t) * IW * 4);
        }
    } else {
        IH2 = IH;
        IW2 = IW;
    }
    return need_pad ? sptr_base : src;
}
bool PoolingImpl::AlgoFilterxModexStride1::usable(
        const PoolingKernSizeParam& param) const {
    auto SH = param.stride[0];
    auto SW = param.stride[1];
    auto FH = param.filter[0];
    auto FW = param.filter[1];

    bool avaible = (param.src_type.category() == DTypeCategory::FLOAT ||
                    param.src_type.category() == DTypeCategory::QUANTIZED) &&
                   param.format == Param::Format::NCHW && SH == 1 && SW == 1 &&
                   FH == FW && (FH == 2 || FH == 3);
    return avaible;
}

void PoolingImpl::AlgoFilterxModexStride1::exec(
        const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];
    auto FH = param.filter[0];

    void* src_ptr = param.src_ptr;
    void* dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(Pooler, NeonPooler, window, midout_type_id)              \
    MIDOUT_BEGIN(megdnn_arm_common_pooling, midout_iv(0),                      \
                 midout_iv(midout_type_id), Pooler::MIDOUT_CASE_NUM,           \
                 NeonPooler::MIDOUT_CASE_NUM, window) {                        \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr,               \
                    src_dtype = param.src_type](size_t index, size_t) {        \
            size_t n = index / C;                                              \
            size_t c = index % C;                                              \
            do_pooling_compact<                                                \
                    Pooler MEGDNN_COMMA NeonPooler MEGDNN_COMMA window>(       \
                    static_cast<const typename Pooler::ctype*>(src_ptr) +      \
                            n * C * IH * IW + c * IH * IW,                     \
                    static_cast<typename Pooler::ctype*>(dst_ptr) +            \
                            n * C * OH * OW + c * OH * OW,                     \
                    src_dtype, IH, IW, OH, OW, PH, PW);                        \
        };                                                                     \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                 \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, \
                run);                                                          \
    }                                                                          \
    MIDOUT_END()

#define DISPATCH_WINDOW(Pooler, NeonPooler, dtype, ctype, comp_type,    \
                        midout_type_id)                                 \
    switch (FH) {                                                       \
        case 2: {                                                       \
            using _Pooler = Pooler<4, dtype, ctype, comp_type>;         \
            using _NeonPooler = NeonPooler<4, dtype, ctype, comp_type>; \
            DISPATCH_FUNC(_Pooler, _NeonPooler, 2, midout_type_id);     \
            break;                                                      \
        }                                                               \
        case 3: {                                                       \
            using _Pooler = Pooler<9, dtype, ctype, comp_type>;         \
            using _NeonPooler = NeonPooler<9, dtype, ctype, comp_type>; \
            DISPATCH_FUNC(_Pooler, _NeonPooler, 3, midout_type_id);     \
            break;                                                      \
        }                                                               \
        default:                                                        \
            megdnn_assert(0, "unsupport pooling filter size");          \
            break;                                                      \
    }

#define DISPATCH_MODE(dtype, ctype, comp_type, midout_type_id)                 \
    switch (param.mode) {                                                      \
        case Mode::MAX:                                                        \
            DISPATCH_WINDOW(MaxPooler, NeonMaxPooler, dtype, ctype, comp_type, \
                            midout_type_id);                                   \
            break;                                                             \
        case Mode::AVERAGE:                                                    \
            DISPATCH_WINDOW(MeanInPooler, NeonMeanPooler, dtype, ctype,        \
                            comp_type, midout_type_id);                        \
            break;                                                             \
        default:                                                               \
            megdnn_assert(0, "unsupport pooling mode");                        \
            break;                                                             \
    }

    if (param.src_type == dtype::Float32{}) {
        DISPATCH_MODE(dt_float32, float, float, 0);
    } else if (param.src_type.enumv() == DTypeEnum::QuantizedS8) {
        DISPATCH_MODE(dt_qint8, int8_t, float, 1);
    } else if (param.src_type.enumv() == DTypeEnum::Quantized8Asymm) {
        DISPATCH_MODE(dt_quint8, uint8_t, float, 2);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    } else if (param.src_type == dtype::Float16{}) {
        DISPATCH_MODE(dt_float16, __fp16, __fp16, 3);
#endif
    }
#undef DISPATCH_FUNC
#undef DISPATCH_WINDOW
#undef DISPATCH_MODE
}
bool PoolingImpl::AlgoFilter2ModexStride2::usable(
        const PoolingKernSizeParam& param) const {
    auto SH = param.stride[0];
    auto SW = param.stride[1];
    auto FH = param.filter[0];
    auto FW = param.filter[1];

    bool avaible = (param.src_type.category() == DTypeCategory::FLOAT ||
                    param.src_type.category() == DTypeCategory::QUANTIZED) &&
                   param.format == Param::Format::NCHW && FH == FW &&
                   SH == SW && FH == 2 && SH == 2;
    return avaible;
}

void PoolingImpl::AlgoFilter2ModexStride2::exec(
        const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];

    void* src_ptr = param.src_ptr;
    void* dst_ptr = param.dst_ptr;
#define DISPATCH_FUNC(Pooler, mode, midout_type_id)                            \
    MIDOUT_BEGIN(megdnn_arm_common_pooling, midout_iv(1),                      \
                 midout_iv(midout_type_id), Pooler::MIDOUT_CASE_NUM) {         \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr,               \
                    src_dtype = param.src_type](size_t index, size_t) {        \
            size_t n = index / C;                                              \
            size_t c = index % C;                                              \
            do_pooling_2x2<Pooler MEGDNN_COMMA mode>(                          \
                    static_cast<const typename Pooler::ctype*>(src_ptr) +      \
                            n * C * IH * IW + c * IH * IW,                     \
                    static_cast<typename Pooler::ctype*>(dst_ptr) +            \
                            n * C * OH * OW + c * OH * OW,                     \
                    src_dtype, IH, IW, OH, OW, PH, PW);                        \
        };                                                                     \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                 \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, \
                run);                                                          \
    }                                                                          \
    MIDOUT_END()

#define DISPATCH_MODE(dtype, ctype, comp_type, midout_type_id)        \
    switch (param.mode) {                                             \
        case Mode::MAX: {                                             \
            using _Pooler = MaxPooler<4, dtype, ctype, comp_type>;    \
            DISPATCH_FUNC(_Pooler, Mode::MAX, midout_type_id);        \
            break;                                                    \
        }                                                             \
        case Mode::AVERAGE: {                                         \
            using _Pooler = MeanInPooler<4, dtype, ctype, comp_type>; \
            DISPATCH_FUNC(_Pooler, Mode::AVERAGE, midout_type_id);    \
            break;                                                    \
        }                                                             \
        default:                                                      \
            megdnn_assert(0, "unsupport pooling mode");               \
            break;                                                    \
    }

    if (param.src_type == dtype::Float32{}) {
        DISPATCH_MODE(dt_float32, float, float, 0);
    } else if (param.src_type.enumv() == DTypeEnum::QuantizedS8) {
        DISPATCH_MODE(dt_qint8, int8_t, float, 1);
    } else if (param.src_type.enumv() == DTypeEnum::Quantized8Asymm) {
        DISPATCH_MODE(dt_quint8, uint8_t, float, 2);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    } else if (param.src_type == dtype::Float16{}) {
        DISPATCH_MODE(dt_float16, __fp16, __fp16, 3);
#endif
    }
#undef DISPATCH_FUNC
#undef DISPATCH_PAD
#undef DISPATCH_MODE
}

bool PoolingImpl::AlgoFilter3MaxStride2::usable(
        const PoolingKernSizeParam& param) const {
    bool avaible = (param.src_type.category() == DTypeCategory::FLOAT ||
                    param.src_type.category() == DTypeCategory::QUANTIZED) &&
                   param.format == Param::Format::NCHW &&
                   param.mode == Mode::MAX && param.filter[0] == 3 &&
                   param.filter[1] == 3 && param.stride[0] == 2 &&
                   param.stride[1] == 2 && param.isz[0] >= 2 &&
                   param.isz[1] >= 2;
    return avaible;
}

void PoolingImpl::AlgoFilter3MaxStride2::exec(
        const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];

    void* src_ptr = param.src_ptr;
    void* dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(type, func, midout_type_id)                              \
    MIDOUT_BEGIN(megdnn_arm_common_pooling, midout_iv(2),                      \
                 midout_iv(midout_type_id)) {                                  \
        WorkspaceBundle wbundle = get_bundle(param);                           \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr,               \
                    wbundle = wbundle,                                         \
                    workspace_ptr = param.workspace<dt_byte>()](               \
                           size_t index, size_t thread_id) {                   \
            auto ws = wbundle;                                                 \
            ws.set(workspace_ptr + ws.total_size_in_bytes() * thread_id);      \
            size_t n = index / C;                                              \
            size_t c = index % C;                                              \
            do_max_pooling_3x3_s2x2_##func##_NEON(                             \
                    static_cast<const type*>(src_ptr) + n * C * IH * IW +      \
                            c * IH * IW,                                       \
                    static_cast<type*>(dst_ptr) + n * C * OH * OW +            \
                            c * OH * OW,                                       \
                    IH, IW, OH, OW, PH, PW, ws);                               \
        };                                                                     \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                 \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, \
                run);                                                          \
    }                                                                          \
    MIDOUT_END();

    if (param.src_type == dtype::Float32{}) {
        DISPATCH_FUNC(float, float, 0);
    } else if (param.src_type.enumv() == DTypeEnum::QuantizedS8) {
        DISPATCH_FUNC(int8_t, int8, 1);
    } else if (param.src_type.enumv() == DTypeEnum::Quantized8Asymm) {
        DISPATCH_FUNC(uint8_t, uint8, 2);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    } else if (param.src_type == dtype::Float16{}) {
        DISPATCH_FUNC(__fp16, float16, 3);
#endif
    }
#undef DISPATCH_FUNC
}
bool PoolingImpl::AlgoFilter3AverageStride2::usable(
        const PoolingKernSizeParam& param) const {
    bool avaible = (param.src_type.category() == DTypeCategory::FLOAT) &&
                   param.format == Param::Format::NCHW &&
                   param.mode == Mode::AVERAGE && param.filter[0] == 3 &&
                   param.filter[1] == 3 && param.stride[0] == 2 &&
                   param.stride[1] == 2 && param.isz[0] >= 2 &&
                   param.isz[1] >= 2;
    return avaible;
}

void PoolingImpl::AlgoFilter3AverageStride2::exec(
        const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];

    void* src_ptr = param.src_ptr;
    void* dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(type, MEGDNN_SIMD_WIDTH, midout_type_id)                 \
    MIDOUT_BEGIN(megdnn_arm_common_pooling, midout_iv(3),                      \
                 midout_iv(midout_type_id)) {                                  \
        WorkspaceBundle wbundle = get_bundle(param);                           \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr,               \
                    wbundle = wbundle,                                         \
                    workspace_ptr = param.workspace<dt_byte>()](               \
                           size_t index, size_t thread_id) {                   \
            auto ws = wbundle;                                                 \
            ws.set(workspace_ptr + ws.total_size_in_bytes() * thread_id);      \
            size_t n = index / C;                                              \
            size_t c = index % C;                                              \
            do_average_pooling_3x3_s2x2_NEON(                                  \
                    static_cast<const type*>(src_ptr) + n * C * IH * IW +      \
                            c * IH * IW,                                       \
                    static_cast<type*>(dst_ptr) + n * C * OH * OW +            \
                            c * OH * OW,                                       \
                    IH, IW, OH, OW, PH, PW, ws, MEGDNN_SIMD_WIDTH);            \
        };                                                                     \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                 \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, \
                run);                                                          \
    }                                                                          \
    MIDOUT_END();
    if (param.src_type == dtype::Float32{}) {
        DISPATCH_FUNC(dt_float32, 4, 0);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    } else if (param.src_type == dtype::Float16{}) {
        DISPATCH_FUNC(__fp16, 8, 1);
#endif
    }
#undef DISPATCH_FUNC
}
bool PoolingImpl::AlgoFilter4MaxStride2::usable(
        const PoolingKernSizeParam& param) const {
    auto SH = param.stride[0];
    auto SW = param.stride[1];
    auto FH = param.filter[0];
    auto FW = param.filter[1];
    auto OH = param.osz[0], OW = param.osz[1];

    bool avaible = (param.src_type.category() == DTypeCategory::FLOAT ||
                    param.src_type.category() == DTypeCategory::QUANTIZED) &&
                   param.format == Param::Format::NCHW &&
                   param.mode == Mode::MAX && FH == 4 && FW == 4 && SH == 2 &&
                   SW == 2 && OH >= 2 && OW >= 2;
    return avaible;
}

void PoolingImpl::AlgoFilter4MaxStride2::exec(
        const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];

    void* src_ptr = param.src_ptr;
    void* dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(type, func, midout_type_id)                              \
    MIDOUT_BEGIN(megdnn_arm_common_pooling, midout_iv(4),                      \
                 midout_iv(midout_type_id)) {                                  \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr,               \
                    src_dtype = param.src_type](size_t index, size_t) {        \
            size_t n = index / C;                                              \
            size_t c = index % C;                                              \
            do_max_pooling_w4x4_s2x2_##func##_NEON(                            \
                    static_cast<const type*>(src_ptr) + n * C * IH * IW +      \
                            c * IH * IW,                                       \
                    static_cast<type*>(dst_ptr) + n * C * OH * OW +            \
                            c * OH * OW,                                       \
                    src_dtype, IH, IW, OH, OW, PH, PW);                        \
        };                                                                     \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                 \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, \
                run);                                                          \
    }                                                                          \
    MIDOUT_END();

    if (param.src_type == dtype::Float32{}) {
        DISPATCH_FUNC(float, float, 0);
    } else if (param.src_type.enumv() == DTypeEnum::QuantizedS8) {
        DISPATCH_FUNC(int8_t, int8, 1);
    } else if (param.src_type.enumv() == DTypeEnum::Quantized8Asymm) {
        DISPATCH_FUNC(uint8_t, uint8, 2);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    } else if (param.src_type == dtype::Float16{}) {
        DISPATCH_FUNC(__fp16, float16, 3);
#endif
    }
#undef DISPATCH_FUNC
}
bool PoolingImpl::AlgoFilter5MaxStride2::usable(
        const PoolingKernSizeParam& param) const {
    auto SH = param.stride[0];
    auto SW = param.stride[1];
    auto FH = param.filter[0];
    auto FW = param.filter[1];
    auto OH = param.osz[0], OW = param.osz[1];

    bool avaible = (param.src_type.category() == DTypeCategory::FLOAT ||
                    param.src_type.category() == DTypeCategory::QUANTIZED) &&
                   param.format == Param::Format::NCHW &&
                   param.mode == Mode::MAX && FH == 5 && FW == 5 && SH == 2 &&
                   SW == 2 && OH >= 2 && OW >= 2;
    return avaible;
}

void PoolingImpl::AlgoFilter5MaxStride2::exec(
        const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];

    void* src_ptr = param.src_ptr;
    void* dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(dtype, type, midout_type_id, MEGDNN_SIMD_WIDTH)          \
    MIDOUT_BEGIN(megdnn_arm_common_pooling, midout_iv(5),                      \
                 midout_iv(midout_type_id)) {                                  \
        WorkspaceBundle wbundle = get_bundle(param);                           \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr,               \
                    wbundle = wbundle,                                         \
                    workspace_ptr = param.workspace<dt_byte>()](               \
                           size_t index, size_t thread_id) {                   \
            auto ws = wbundle;                                                 \
            ws.set(workspace_ptr + ws.total_size_in_bytes() * thread_id);      \
            size_t n = index / C;                                              \
            size_t c = index % C;                                              \
            do_max_pooling_w5x5_s2x2_NEON<dtype>(                              \
                    static_cast<const type*>(src_ptr) + n * C * IH * IW +      \
                            c * IH * IW,                                       \
                    static_cast<type*>(dst_ptr) + n * C * OH * OW +            \
                            c * OH * OW,                                       \
                    IH, IW, OH, OW, PH, PW, ws, MEGDNN_SIMD_WIDTH);            \
        };                                                                     \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                 \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, \
                run);                                                          \
    }                                                                          \
    MIDOUT_END();

    if (param.src_type == dtype::Float32{}) {
        DISPATCH_FUNC(dt_float32, float, 0, 4);
    } else if (param.src_type.enumv() == DTypeEnum::QuantizedS8) {
        DISPATCH_FUNC(dt_int8, int8_t, 1, 16);
    } else if (param.src_type.enumv() == DTypeEnum::Quantized8Asymm) {
        DISPATCH_FUNC(dt_uint8, uint8_t, 2, 16);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    } else if (param.src_type == dtype::Float16{}) {
        DISPATCH_FUNC(dt_float16, __fp16, 3, 8);
#endif
    }
#undef DISPATCH_FUNC
}

bool PoolingImpl::AlgoInt8Filter2MaxStride2::usable(
        const PoolingKernSizeParam& param) const {
    auto SH = param.stride[0];
    auto SW = param.stride[1];
    auto FH = param.filter[0];
    auto FW = param.filter[1];
    auto PH = param.padding[0];
    auto PW = param.padding[1];

    bool avaible = param.src_type == dtype::Int8() &&
                   param.format == Param::Format::NCHW &&
                   param.mode == Mode::MAX && SH == 2 && SW == 2 && PH == 0 &&
                   PW == 0 && FH == 2 && FW == 2;
    return avaible;
}

void PoolingImpl::AlgoInt8Filter2MaxStride2::exec(
        const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;

    auto src_ptr = param.src<dt_int8>();
    auto dst_ptr = param.dst<dt_int8>();

    MIDOUT_BEGIN(megdnn_arm_common_pooling, midout_iv(6)) {
        auto run = [C, IH, IW, OH, OW, src_ptr, dst_ptr](size_t index, size_t) {
            size_t n = index / C;
            size_t c = index % C;
            pooling_max_w2x2_s2x2(src_ptr + n * C * IH * IW + c * IH * IW,
                                  dst_ptr + n * C * OH * OW + c * OH * OW, 1, 1,
                                  IH, IW, OH, OW);
        };
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N * C,
                run);
    }
    MIDOUT_END();
}

bool PoolingImpl::AlgoInt8Filter3MaxStride2::usable(
        const PoolingKernSizeParam& param) const {
    auto SH = param.stride[0];
    auto SW = param.stride[1];
    auto FH = param.filter[0];
    auto FW = param.filter[1];
    auto IH = param.isz[0];
    auto IW = param.isz[1];

    bool avaible = param.src_type == dtype::Int8() &&
                   param.format == Param::Format::NCHW &&
                   param.mode == Mode::MAX && FH == 3 && FW == 3 && SH == 2 &&
                   SW == 2 && IH >= 2 && IW >= 2;
    return avaible;
}

void PoolingImpl::AlgoInt8Filter3MaxStride2::exec(
        const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];

    auto src_ptr = param.src<dt_int8>();
    auto dst_ptr = param.dst<dt_int8>();

    MIDOUT_BEGIN(megdnn_arm_common_pooling, midout_iv(7)) {
        WorkspaceBundle wbundle = get_bundle(param);
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr,
                    wbundle = wbundle,
                    workspace_ptr = param.workspace<dt_byte>()](
                           size_t index, size_t thread_id) {
            auto ws = wbundle;
            ws.set(workspace_ptr + thread_id * ws.total_size_in_bytes());
            size_t n = index / C;
            size_t c = index % C;
            do_max_pooling_3x3_s2x2_int8_NEON(
                    src_ptr + n * C * IH * IW + c * IH * IW,
                    dst_ptr + n * C * OH * OW + c * OH * OW, IH, IW, OH, OW, PH,
                    PW, ws);
        };
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N * C,
                run);
    }
    MIDOUT_END();
}

bool PoolingImpl::AlgoFilter3ModexStridexNCHW44::usable(
        const PoolingKernSizeParam& param) const {
    auto SH = param.stride[0];
    auto SW = param.stride[1];
    auto FH = param.filter[0];
    auto FW = param.filter[1];

    bool avaible = (param.src_type.enumv() == DTypeEnum::QuantizedS8 ||
                    param.src_type.enumv() == DTypeEnum::Int8) &&
                   param.format == Param::Format::NCHW44 &&
                   (param.mode == Mode::MAX || param.mode == Mode::AVERAGE) &&
                   FH == 3 && FW == 3 && SW == SH && (SH == 1 || SW == 2);
    //! Int8 not support average, because its round mode is different form
    //! qint8
    avaible &= !(param.src_type.enumv() == DTypeEnum::Int8 &&
                 param.mode == Mode::AVERAGE);
    return avaible;
}

void PoolingImpl::AlgoFilter3ModexStridexNCHW44::exec(
        const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];
    auto SW = param.stride[0];

    void* src_ptr = param.src_ptr;
    void* dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(type, func, i, mode)                                     \
    MIDOUT_BEGIN(megdnn_arm_common_pooling, midout_iv(8),                      \
                 midout_iv(#type #i##_hash)) {                                 \
        WorkspaceBundle wbundle = get_bundle_nchw44(param);                    \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr,               \
                    wbundle = wbundle,                                         \
                    workspace_ptr = param.workspace<dt_byte>()](               \
                           size_t index, size_t thread_id) {                   \
            auto ws = wbundle;                                                 \
            ws.set(workspace_ptr + ws.total_size_in_bytes() * thread_id);      \
            size_t n = index / C;                                              \
            size_t c = index % C;                                              \
            do_##mode##_pooling_3x3_stride##i##_##func##_nchw44_NEON(          \
                    static_cast<const type*>(src_ptr) + n * C * IH * IW * 4 +  \
                            c * IH * IW * 4,                                   \
                    static_cast<type*>(dst_ptr) + n * C * OH * OW * 4 +        \
                            c * OH * OW * 4,                                   \
                    IH, IW, OH, OW, PH, PW, ws);                               \
        };                                                                     \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                 \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, \
                run);                                                          \
    }                                                                          \
    MIDOUT_END();

#define DISPATCH_MODE(type, func, stride)                       \
    switch (param.mode) {                                       \
        case Mode::MAX: {                                       \
            DISPATCH_FUNC(type, func, stride, max);             \
            break;                                              \
        }                                                       \
        case Mode::AVERAGE: {                                   \
            DISPATCH_FUNC(type, func, stride, avg);             \
            break;                                              \
        }                                                       \
        default:                                                \
            megdnn_throw(ssprintf("Unsupport pooling mode %d",  \
                                  static_cast<int>(param.mode)) \
                                 .c_str());                     \
    }

#define DISPATCH_STRIDE(type, func)                                         \
    switch (SW) {                                                           \
        case 1: {                                                           \
            DISPATCH_MODE(type, func, 1);                                   \
            break;                                                          \
        }                                                                   \
        case 2: {                                                           \
            DISPATCH_MODE(type, func, 2);                                   \
            break;                                                          \
        }                                                                   \
        default:                                                            \
            megdnn_throw(ssprintf("Unsupport stride size %d", SW).c_str()); \
    }

    DISPATCH_STRIDE(int8_t, int8);

#undef DISPATCH_STRIDE
#undef DISPATCH_MODE
#undef DISPATCH_FUNC
}

bool PoolingImpl::AlgoFilter2ModexStridexNCHW44::usable(
        const PoolingKernSizeParam& param) const {
    auto SH = param.stride[0];
    auto SW = param.stride[1];
    auto FH = param.filter[0];
    auto FW = param.filter[1];

    bool avaible = (param.src_type.enumv() == DTypeEnum::QuantizedS8 ||
                    param.src_type.enumv() == DTypeEnum::Int8) &&
                   param.format == Param::Format::NCHW44 &&
                   (param.mode == Mode::MAX || param.mode == Mode::AVERAGE) &&
                   FH == 2 && FW == 2 && SH == SW && (SW == 1 || SW == 2);
    //! Int8 not support average, because its round mode is different form
    //! qint8
    avaible &= !(param.src_type.enumv() == DTypeEnum::Int8 &&
                 param.mode == Mode::AVERAGE);
    return avaible;
}

void PoolingImpl::AlgoFilter2ModexStridexNCHW44::exec(
        const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];
    auto SW = param.stride[0];

    void* src_ptr = param.src_ptr;
    void* dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(type, func, i, mode)                                     \
    MIDOUT_BEGIN(megdnn_arm_common_pooling, midout_iv(9),                      \
                 midout_iv(#func #i##_hash)) {                                 \
        WorkspaceBundle wbundle = get_bundle_nchw44(param);                    \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr,               \
                    wbundle = wbundle,                                         \
                    workspace_ptr = param.workspace<dt_byte>()](               \
                           size_t index, size_t thread_id) {                   \
            auto ws = wbundle;                                                 \
            ws.set(workspace_ptr + ws.total_size_in_bytes() * thread_id);      \
            size_t n = index / C;                                              \
            size_t c = index % C;                                              \
            do_##mode##_pooling_2x2_stride##i##_##func##_nchw44_NEON(          \
                    static_cast<const type*>(src_ptr) + n * C * IH * IW * 4 +  \
                            c * IH * IW * 4,                                   \
                    static_cast<type*>(dst_ptr) + n * C * OH * OW * 4 +        \
                            c * OH * OW * 4,                                   \
                    IH, IW, OH, OW, PH, PW, ws);                               \
        };                                                                     \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                 \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, \
                run);                                                          \
    }                                                                          \
    MIDOUT_END();

#define DISPATCH_MODE(type, func, stride)                       \
    switch (param.mode) {                                       \
        case Mode::MAX: {                                       \
            DISPATCH_FUNC(type, func, stride, max);             \
            break;                                              \
        }                                                       \
        case Mode::AVERAGE: {                                   \
            DISPATCH_FUNC(type, func, stride, avg);             \
            break;                                              \
        }                                                       \
        default:                                                \
            megdnn_throw(ssprintf("Unsupport pooling mode %d",  \
                                  static_cast<int>(param.mode)) \
                                 .c_str());                     \
    }

#define DISPATCH_STRIDE(type, func)                                         \
    switch (SW) {                                                           \
        case 1: {                                                           \
            DISPATCH_MODE(type, func, 1);                                   \
            break;                                                          \
        }                                                                   \
        case 2: {                                                           \
            DISPATCH_MODE(type, func, 2);                                   \
            break;                                                          \
        }                                                                   \
        default:                                                            \
            megdnn_throw(ssprintf("Unsupport stride size %d", SW).c_str()); \
    }

    DISPATCH_STRIDE(int8_t, int8);

#undef DISPATCH_STRIDE
#undef DISPATCH_MODE
#undef DISPATCH_FUNC
}

bool PoolingImpl::AlgoFilter4ModexStridexNCHW44::usable(
        const PoolingKernSizeParam& param) const {
    auto SH = param.stride[0];
    auto SW = param.stride[1];
    auto FH = param.filter[0];
    auto FW = param.filter[1];

    bool avaible = (param.src_type.enumv() == DTypeEnum::QuantizedS8 ||
                    param.src_type.enumv() == DTypeEnum::Int8) &&
                   param.format == Param::Format::NCHW44 &&
                   (param.mode == Mode::MAX || param.mode == Mode::AVERAGE) &&
                   FH == 4 && FW == 4 && SH == SW && (SW == 1 || SW == 2);

    //! Int8 not support average, because its round mode is different form
    //! qint8
    avaible &= !(param.src_type.enumv() == DTypeEnum::Int8 &&
                 param.mode == Mode::AVERAGE);
    return avaible;
}

void PoolingImpl::AlgoFilter4ModexStridexNCHW44::exec(
        const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];
    auto SW = param.stride[0];

    void* src_ptr = param.src_ptr;
    void* dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(type, func, i, mode)                                     \
    MIDOUT_BEGIN(megdnn_arm_common_pooling, midout_iv(10),                     \
                 midout_iv(#func #i##_hash)) {                                 \
        WorkspaceBundle wbundle = get_bundle_nchw44(param);                    \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr,               \
                    wbundle = wbundle,                                         \
                    workspace_ptr = param.workspace<dt_byte>()](               \
                           size_t index, size_t thread_id) {                   \
            auto ws = wbundle;                                                 \
            ws.set(workspace_ptr + ws.total_size_in_bytes() * thread_id);      \
            size_t n = index / C;                                              \
            size_t c = index % C;                                              \
            do_##mode##_pooling_4x4_stride##i##_##func##_nchw44_NEON(          \
                    static_cast<const type*>(src_ptr) + n * C * IH * IW * 4 +  \
                            c * IH * IW * 4,                                   \
                    static_cast<type*>(dst_ptr) + n * C * OH * OW * 4 +        \
                            c * OH * OW * 4,                                   \
                    IH, IW, OH, OW, PH, PW, ws);                               \
        };                                                                     \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                 \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, \
                run);                                                          \
    }                                                                          \
    MIDOUT_END();

#define DISPATCH_MODE(type, func, stride)                       \
    switch (param.mode) {                                       \
        case Mode::MAX: {                                       \
            DISPATCH_FUNC(type, func, stride, max);             \
            break;                                              \
        }                                                       \
        case Mode::AVERAGE: {                                   \
            DISPATCH_FUNC(type, func, stride, avg);             \
            break;                                              \
        }                                                       \
        default:                                                \
            megdnn_throw(ssprintf("Unsupport pooling mode %d",  \
                                  static_cast<int>(param.mode)) \
                                 .c_str());                     \
    }

#define DISPATCH_STRIDE(type, func)                                         \
    switch (SW) {                                                           \
        case 1: {                                                           \
            DISPATCH_MODE(type, func, 1);                                   \
            break;                                                          \
        }                                                                   \
        case 2: {                                                           \
            DISPATCH_MODE(type, func, 2);                                   \
            break;                                                          \
        }                                                                   \
        default:                                                            \
            megdnn_throw(ssprintf("Unsupport stride size %d", SW).c_str()); \
    }

    DISPATCH_STRIDE(int8_t, int8);

#undef DISPATCH_STRIDE
#undef DISPATCH_MODE
#undef DISPATCH_FUNC
}

bool PoolingImpl::AlgoFilter5ModexStridexNCHW44::usable(
        const PoolingKernSizeParam& param) const {
    auto SH = param.stride[0];
    auto SW = param.stride[1];
    auto FH = param.filter[0];
    auto FW = param.filter[1];

    bool avaible = (param.src_type.enumv() == DTypeEnum::QuantizedS8 ||
                    param.src_type.enumv() == DTypeEnum::Int8) &&
                   param.format == Param::Format::NCHW44 &&
                   (param.mode == Mode::MAX || param.mode == Mode::AVERAGE) &&
                   FH == 5 && FW == 5 && SH == SW && (SW == 1 || SW == 2);
    //! Int8 not support average, because its round mode is different form
    //! qint8
    avaible &= !(param.src_type.enumv() == DTypeEnum::Int8 &&
                 param.mode == Mode::AVERAGE);
    return avaible;
}

void PoolingImpl::AlgoFilter5ModexStridexNCHW44::exec(
        const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];
    auto SW = param.stride[0];

    void* src_ptr = param.src_ptr;
    void* dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(type, func, i, mode)                                     \
    MIDOUT_BEGIN(megdnn_arm_common_pooling, midout_iv(11),                     \
                 midout_iv(#func #i##_hash)) {                                 \
        WorkspaceBundle wbundle = get_bundle_nchw44(param);                    \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr,               \
                    wbundle = wbundle,                                         \
                    workspace_ptr = param.workspace<dt_byte>()](               \
                           size_t index, size_t thread_id) {                   \
            auto ws = wbundle;                                                 \
            ws.set(workspace_ptr + ws.total_size_in_bytes() * thread_id);      \
            size_t n = index / C;                                              \
            size_t c = index % C;                                              \
            do_##mode##_pooling_5x5_stride##i##_##func##_nchw44_NEON(          \
                    static_cast<const type*>(src_ptr) + n * C * IH * IW * 4 +  \
                            c * IH * IW * 4,                                   \
                    static_cast<type*>(dst_ptr) + n * C * OH * OW * 4 +        \
                            c * OH * OW * 4,                                   \
                    IH, IW, OH, OW, PH, PW, ws);                               \
        };                                                                     \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                 \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, \
                run);                                                          \
    }                                                                          \
    MIDOUT_END();

#define DISPATCH_MODE(type, func, stride)                       \
    switch (param.mode) {                                       \
        case Mode::MAX: {                                       \
            DISPATCH_FUNC(type, func, stride, max);             \
            break;                                              \
        }                                                       \
        case Mode::AVERAGE: {                                   \
            DISPATCH_FUNC(type, func, stride, avg);             \
            break;                                              \
        }                                                       \
        default:                                                \
            megdnn_throw(ssprintf("Unsupport pooling mode %d",  \
                                  static_cast<int>(param.mode)) \
                                 .c_str());                     \
    }

#define DISPATCH_STRIDE(type, func)                                         \
    switch (SW) {                                                           \
        case 1: {                                                           \
            DISPATCH_MODE(type, func, 1);                                   \
            break;                                                          \
        }                                                                   \
        case 2: {                                                           \
            DISPATCH_MODE(type, func, 2);                                   \
            break;                                                          \
        }                                                                   \
        default:                                                            \
            megdnn_throw(ssprintf("Unsupport stride size %d", SW).c_str()); \
    }

    DISPATCH_STRIDE(int8_t, int8);

#undef DISPATCH_STRIDE
#undef DISPATCH_MODE
#undef DISPATCH_FUNC
}

}  // namespace arm_common
}  // namespace megdnn
// vim: syntax=cpp.doxygen
