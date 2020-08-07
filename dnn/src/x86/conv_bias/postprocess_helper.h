/**
 * \file dnn/src/x86/conv_bias/postprocess_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/opr_param_defs.h"
#include "src/fallback/conv_bias/common.h"
#include "src/x86/elemwise_op.h"
#include "src/x86/utils.h"
#include "src/fallback/conv_bias/opr_impl.h"

namespace megdnn {
namespace x86 {

#define BIAS_CASE(mode)                                             \
    case megdnn::param::ConvBias::NonlineMode::mode:                \
        elem_mode = megdnn::param::Elemwise::Mode::FUSE_ADD_##mode; \
        break;

#define NOBIAS_CASE(mode)                                \
    case megdnn::param::ConvBias::NonlineMode::mode:     \
        elem_mode = megdnn::param::Elemwise::Mode::mode; \
        break;

#define IDENTITY_CASE(mode)                          \
    case megdnn::param::ConvBias::NonlineMode::mode: \
        break;

#define DEFAULT_CASE                            \
    default:                                    \
        megdnn_throw("unsupported nolinemode"); \
        break;

#define CALL_UNARY(_op, _simd_type)                                           \
    thin_function<void(const ctype*, ctype*, DType, DType, size_t)> run =     \
            OpCallerUnary<_op<_simd_type, ctype, ctype>, _simd_type>::run;    \
    run(static_cast<ctype*>(conv_dst_ptr), reinterpret_cast<ctype*>(dst_ptr), \
        bias_type, dst_type, N* OC* OH* OW);

#define CALL_BINARY_BROADCAST(_op, _simd_type)                                \
    thin_function<void(const ctype*, const ctype*, ctype*, DType, DType,      \
                       DType, size_t, size_t, size_t)>                        \
            run = OpCallerBinary<_op<_simd_type, ctype, ctype>, _simd_type,   \
                                 megdnn::x86::BcastType::VEC_BCAST101>::run;  \
    run(static_cast<ctype*>(conv_dst_ptr), static_cast<ctype*>(bias_ptr),     \
        reinterpret_cast<ctype*>(dst_ptr), bias_type, bias_type, dst_type, N, \
        OC, OH* OW);

#define CALL_BINARY(_op, _simd_type)                                        \
    thin_function<void(const ctype*, const ctype*, ctype*, DType, DType,    \
                       DType, size_t)>                                      \
            run = OpCallerBinary<_op<_simd_type, ctype, ctype>, _simd_type, \
                                 megdnn::x86::BcastType::VEC_VEC>::run;     \
    run(static_cast<ctype*>(conv_dst_ptr), static_cast<ctype*>(bias_ptr),   \
        reinterpret_cast<ctype*>(dst_ptr), bias_type, bias_type, dst_type,  \
        N* OC* OH* OW);

#define cb_unary(_simd_type)                                          \
    if (elem_mode == megdnn::param::Elemwise::Mode::RELU) {           \
        CALL_UNARY(ReluOp, _simd_type);                               \
    } else if (elem_mode == megdnn::param::Elemwise::Mode::SIGMOID) { \
        CALL_UNARY(SigmoidOp, _simd_type);                            \
    } else if (elem_mode == megdnn::param::Elemwise::Mode::H_SWISH) { \
        CALL_UNARY(HSwishOp, _simd_type);                             \
    }

#define FOR_NONLINEAR_NOBIAS()                    \
    if (is_supported(SIMDType::AVX2)) {           \
        cb_unary(SIMDType::AVX2)                  \
    } else if (is_supported(SIMDType::SSE4_2)) {  \
        cb_unary(SIMDType::SSE4_2)                \
    } else {                                      \
        cb_unary(SIMDType::NONE)                  \
    }

#define cb_binary(_caller, _simd_type)                                         \
    if (elem_mode == megdnn::param::Elemwise::Mode::ADD) {                     \
        _caller(AddOp, _simd_type);                                            \
    } else if (elem_mode == megdnn::param::Elemwise::Mode::FUSE_ADD_SIGMOID) { \
        _caller(FuseAddSigmoidOp, _simd_type);                                 \
    } else if (elem_mode == megdnn::param::Elemwise::Mode::FUSE_ADD_RELU) {    \
        _caller(FuseAddReluOp, _simd_type);                                    \
    } else if (elem_mode == megdnn::param::Elemwise::Mode::FUSE_ADD_H_SWISH) { \
        _caller(FuseAddHSwishOp, _simd_type);                                  \
    }

#define FOR_NONLINEAR(CALLER)                     \
    if (is_supported(SIMDType::AVX2)) {           \
        cb_binary(CALLER, SIMDType::AVX2)         \
    } else if (is_supported(SIMDType::SSE4_2)) {  \
        cb_binary(CALLER, SIMDType::SSE4_2)       \
    } else {                                      \
        cb_binary(CALLER, SIMDType::NONE)         \
    }

#define FOR_BIAS(bias_mode)                       \
    switch (bias_mode) {                          \
        case BiasMode::NO_BIAS:                   \
            FOR_NONLINEAR_NOBIAS();               \
            break;                                \
        case BiasMode::BROADCAST_CHANNEL_BIAS:    \
            FOR_NONLINEAR(CALL_BINARY_BROADCAST); \
            break;                                \
        case BiasMode::BIAS:                      \
            FOR_NONLINEAR(CALL_BINARY);           \
            break;                                \
        default:                                  \
            break;                                \
    }

template <typename ctype, typename dtype = ctype,
          megdnn::PostprocessMode postprocess_mode =
                  megdnn::PostprocessMode::FLOAT>
struct PostProcess {
    static void run(void* conv_dst_ptr, void* bias_ptr, void* dst_ptr,
                    megdnn::ConvBiasForward::BiasMode bias_mode,
                    megdnn::param::ConvBias::NonlineMode nonlineMode,
                    DType bias_type, DType dst_type, size_t N, size_t OC,
                    size_t OH, size_t OW, size_t pack_oc_size = 1) {
        MEGDNN_MARK_USED_VAR(pack_oc_size);
        megdnn_assert(pack_oc_size == 1,
                      "PostProcess only support nchw in x86");
        megdnn::param::Elemwise::Mode elem_mode =
                megdnn::param::Elemwise::Mode::ADD;
        if (bias_mode != megdnn::ConvBiasForward::BiasMode::NO_BIAS) {
            switch (nonlineMode) {
                BIAS_CASE(RELU);
                BIAS_CASE(SIGMOID);
                BIAS_CASE(H_SWISH);
                IDENTITY_CASE(IDENTITY);
                DEFAULT_CASE;
            }
        } else {
            switch (nonlineMode) {
                NOBIAS_CASE(RELU);
                NOBIAS_CASE(SIGMOID);
                NOBIAS_CASE(H_SWISH);
                IDENTITY_CASE(IDENTITY);
                DEFAULT_CASE;
            }
        }
        FOR_BIAS(bias_mode);
    }
};

template <typename ctype, typename dtype>
struct PostProcess<ctype, dtype, megdnn::PostprocessMode::NO_PROCESS> {
    static void run(void* conv_dst_ptr, void* bias_ptr, void* dst_ptr,
                    megdnn::ConvBiasForward::BiasMode bias_mode,
                    megdnn::param::ConvBias::NonlineMode nonlineMode,
                    DType bias_type, DType dst_type, size_t N, size_t OC,
                    size_t OH, size_t OW,size_t pack_oc_size = 1) {
        MEGDNN_MARK_USED_VAR(pack_oc_size);
        MEGDNN_MARK_USED_VAR(conv_dst_ptr);
        MEGDNN_MARK_USED_VAR(bias_ptr);
        MEGDNN_MARK_USED_VAR(dst_ptr);
        MEGDNN_MARK_USED_VAR(bias_mode);
        MEGDNN_MARK_USED_VAR(nonlineMode);
        MEGDNN_MARK_USED_VAR(bias_type);
        MEGDNN_MARK_USED_VAR(dst_type);
        MEGDNN_MARK_USED_VAR(N);
        MEGDNN_MARK_USED_VAR(OC);
        MEGDNN_MARK_USED_VAR(OH);
        MEGDNN_MARK_USED_VAR(OW);
    }
};
#undef FOR_NONLINEAR_NOBIAS
#undef FOR_NONLINEAR
#undef FOR_BIAS

#undef cb_binary
#undef cb_unary
#undef CALL_UNARY
#undef CALL_BINARY_BROADCAST

#define CALL_UNARY(_op, _simd_type)                                           \
    thin_function<void(const ctype*, dtype*, DType, DType, size_t)> run =     \
            OpCallerUnary<_op<_simd_type, ctype, dtype>, _simd_type>::run;    \
    run(static_cast<ctype*>(conv_dst_ptr), reinterpret_cast<dtype*>(dst_ptr), \
        bias_type, dst_type, N* OC* OH* OW);

#define CALL_BINARY_BROADCAST(_op, _simd_type)                                \
    thin_function<void(const ctype*, const ctype*, dtype*, DType, DType,      \
                       DType, size_t, size_t, size_t)>                        \
            run = OpCallerBinary<_op<_simd_type, ctype, dtype>, _simd_type,   \
                                 megdnn::x86::BcastType::VEC_BCAST101>::run;  \
    run(static_cast<ctype*>(conv_dst_ptr), static_cast<ctype*>(bias_ptr),     \
        reinterpret_cast<dtype*>(dst_ptr), bias_type, bias_type, dst_type, N, \
        OC, OH* OW);

#define cb_unary(_simd_type)                                                 \
    if (elem_mode == megdnn::param::Elemwise::Mode::RELU) {                  \
        CALL_UNARY(ReluOp, _simd_type);                                      \
    } else if (elem_mode == megdnn::param::Elemwise::Mode::H_SWISH) {        \
        CALL_UNARY(HSwishOp, _simd_type);                                    \
    } else {                                                                 \
        if (nonlineMode == megdnn::param::ConvBias::NonlineMode::IDENTITY) { \
            CALL_UNARY(TypeCvtOp, _simd_type);                               \
        } else {                                                             \
            megdnn_throw("not supported nonlinemode\n");                     \
        }                                                                    \
    }

#define FOR_NONLINEAR_NOBIAS()                                            \
    if (is_supported(SIMDType::AVX2)) {                                   \
        if (elem_mode == megdnn::param::Elemwise::Mode::RELU) {           \
            CALL_UNARY(ReluOp, SIMDType::AVX2);                           \
        } else if (elem_mode == megdnn::param::Elemwise::Mode::H_SWISH) { \
            CALL_UNARY(HSwishOp, SIMDType::NONE);                         \
        } else {                                                          \
            if (nonlineMode ==                                            \
                megdnn::param::ConvBias::NonlineMode::IDENTITY) {         \
                CALL_UNARY(TypeCvtOp, SIMDType::NONE);                    \
            } else {                                                      \
                megdnn_throw("not supported nonlinemode\n");              \
            }                                                             \
        }                                                                 \
    } else if (is_supported(SIMDType::SSE4_2)) {                          \
        cb_unary(SIMDType::SSE4_2)                                        \
    } else {                                                              \
        cb_unary(SIMDType::NONE)                                          \
    }

#define cb_binary(_caller, _simd_type)                                         \
    if (elem_mode == megdnn::param::Elemwise::Mode::ADD) {                     \
        _caller(AddOp, _simd_type);                                            \
    } else if (elem_mode == megdnn::param::Elemwise::Mode::FUSE_ADD_RELU) {    \
        _caller(FuseAddReluOp, _simd_type);                                    \
    } else if (elem_mode == megdnn::param::Elemwise::Mode::FUSE_ADD_H_SWISH) { \
        _caller(FuseAddHSwishOp, _simd_type);                                  \
    }

#define FOR_NONLINEAR(CALLER)                     \
    if (is_supported(SIMDType::AVX2)) {           \
        cb_binary(CALLER, SIMDType::AVX2)         \
    } else if (!is_supported(SIMDType::SSE4_2)) { \
        cb_binary(CALLER, SIMDType::SSE4_2)       \
    } else {                                      \
        cb_binary(CALLER, SIMDType::NONE)         \
    }

#define FOR_BIAS(bias_mode)                              \
    switch (bias_mode) {                                 \
        case BiasMode::NO_BIAS:                          \
            FOR_NONLINEAR_NOBIAS();                      \
            break;                                       \
        case BiasMode::BROADCAST_CHANNEL_BIAS:           \
            FOR_NONLINEAR(CALL_BINARY_BROADCAST);        \
            break;                                       \
        default:                                         \
            break;                                       \
    }

template <typename ctype, typename dtype>
struct PostProcess<ctype, dtype, megdnn::PostprocessMode::QUANTIZED> {
    static void run(void* conv_dst_ptr, void* bias_ptr, void* dst_ptr,
                    megdnn::ConvBiasForward::BiasMode bias_mode,
                    megdnn::param::ConvBiasV0::NonlineMode nonlineMode,
                    DType bias_type, DType dst_type, size_t N, size_t OC,
                    size_t OH, size_t OW, size_t pack_oc_size = 1) {
        MEGDNN_MARK_USED_VAR(pack_oc_size);
        megdnn_assert(pack_oc_size == 1,
                      "PostProcess only support nchw nchw in x86");
        megdnn::param::Elemwise::Mode elem_mode =
                megdnn::param::Elemwise::Mode::ADD;
        if (bias_mode != megdnn::ConvBiasForward::BiasMode::NO_BIAS) {
            switch (nonlineMode) {
                BIAS_CASE(RELU);
                BIAS_CASE(H_SWISH);
                IDENTITY_CASE(IDENTITY);
                DEFAULT_CASE;
            }
        } else {
            switch (nonlineMode) {
                NOBIAS_CASE(RELU);
                NOBIAS_CASE(H_SWISH);
                IDENTITY_CASE(IDENTITY);
                DEFAULT_CASE;
            }
        }

        FOR_BIAS(bias_mode);

#undef FOR_NONLINEAR_NOBIAS
#undef FOR_NONLINEAR
#undef FOR_BIAS
    }
};

#undef CALL_BINARY
#undef CALL_BINARY_BROADCAST

#define CALL_BINARY(_op, _simd_type)                                        \
    thin_function<void(const ctype*, const ctype*, dtype*, DType, DType,    \
                       DType, size_t)>                                      \
            run = OpCallerBinary<_op<_simd_type, ctype, dtype>, _simd_type, \
                                 megdnn::x86::BcastType::VEC_VEC>::run;     \
    run(static_cast<ctype*>(conv_dst_ptr), static_cast<ctype*>(bias_ptr),   \
        reinterpret_cast<dtype*>(dst_ptr), bias_type, bias_type, dst_type,  \
        N* OC* OH* OW);

#define CALL_BINARY_BROADCAST(_op, _simd_type)                                \
    thin_function<void(const ctype*, const ctype*, dtype*, DType, DType,      \
                       DType, size_t, size_t, size_t)>                        \
            run = OpCallerBinary<_op<_simd_type, ctype, dtype>, _simd_type,   \
                                 megdnn::x86::BcastType::VEC_BCAST101>::run;  \
    run(static_cast<ctype*>(conv_dst_ptr), static_cast<ctype*>(bias_ptr),     \
        reinterpret_cast<dtype*>(dst_ptr), bias_type, bias_type, dst_type, N, \
        OC, OH* OW);

#define FOR_SIMD(CALLER)                         \
    if (is_supported(SIMDType::AVX2)) {          \
        CALLER(AddOp, SIMDType::AVX2)            \
    } else if (is_supported(SIMDType::SSE4_2)) { \
        CALLER(AddOp, SIMDType::SSE4_2)          \
    } else {                                     \
        CALLER(AddOp, SIMDType::NONE)            \
    }

#define FOR_BIAS(bias_mode)                    \
    switch (bias_mode) {                       \
        case BiasMode::BIAS:                   \
            FOR_SIMD(CALL_BINARY);             \
            break;                             \
        case BiasMode::BROADCAST_CHANNEL_BIAS: \
            FOR_SIMD(CALL_BINARY_BROADCAST);   \
            break;                             \
        default:                               \
            break;                             \
    }

template <typename ctype, typename dtype>
struct PostProcess<ctype, dtype, megdnn::PostprocessMode::ADD_BIAS> {
    static void run(void* conv_dst_ptr, void* bias_ptr, void* dst_ptr,
                    megdnn::ConvBiasForward::BiasMode bias_mode,
                    megdnn::param::ConvBiasV0::NonlineMode nonlineMode,
                    DType bias_type, DType dst_type, size_t N, size_t OC,
                    size_t OH, size_t OW, size_t pack_oc_size = 1) {
        MEGDNN_MARK_USED_VAR(pack_oc_size);
        megdnn_assert(pack_oc_size == 1,
                      "PostProcess only support nchw in x86");
        megdnn_assert(
                nonlineMode == megdnn::param::ConvBiasV0::NonlineMode::IDENTITY,
                "Add bias PostProcess only support IDENTITY");
        if (bias_mode == megdnn::ConvBiasForward::BiasMode::NO_BIAS) {
            return;
        }
        FOR_BIAS(bias_mode);
#undef CALL_BINARY
#undef CALL_BINARY_BROADCAST
#undef FOR_SIMD
#undef FOR_BIAS
    }
};

#undef cb_unary
#undef cb_binary
#undef BIAS_CASE
#undef NOBIAS_CASE
#undef DEFAULT_CASE
#undef CALL_UNARY
#undef CALL_BINARY
#undef CALL_BINARY_BROADCAST

#define DISPATCH_CONV_WINOGRAD_NONLINE(_midout_tag, cb, _bias_id, _simd_type, \
                                       _src_type, _dst_type, _bmode,          \
                                       _nonline_mode, ...)                    \
    switch (_nonline_mode) {                                                  \
        case param::ConvBias::NonlineMode::IDENTITY:                          \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 0) {                          \
                cb(_bmode,                                                    \
                   NoneOp<_simd_type MEGDNN_COMMA _src_type MEGDNN_COMMA      \
                                  _dst_type>,                                 \
                   __VA_ARGS__);                                              \
            }                                                                 \
            MIDOUT_END();                                                     \
            break;                                                            \
        case param::ConvBias::NonlineMode::RELU: {                            \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 1) {                          \
                cb(_bmode,                                                    \
                   ReluOp<_simd_type MEGDNN_COMMA _src_type MEGDNN_COMMA      \
                                  _dst_type>,                                 \
                   __VA_ARGS__);                                              \
            }                                                                 \
            MIDOUT_END();                                                     \
            break;                                                            \
        }                                                                     \
        case param::ConvBias::NonlineMode::SIGMOID: {                         \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 2) {                          \
                cb(_bmode,                                                    \
                   SigmoidOp<_simd_type MEGDNN_COMMA _src_type MEGDNN_COMMA   \
                                     _dst_type>,                              \
                   __VA_ARGS__);                                              \
            }                                                                 \
            MIDOUT_END();                                                     \
            break;                                                            \
        }                                                                     \
        case param::ConvBias::NonlineMode::H_SWISH: {                         \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 3) {                          \
                cb(_bmode,                                                    \
                   HSwishOp<_simd_type MEGDNN_COMMA _src_type MEGDNN_COMMA    \
                                    _dst_type>,                               \
                   __VA_ARGS__);                                              \
            }                                                                 \
            MIDOUT_END();                                                     \
            break;                                                            \
        }                                                                     \
        default:                                                              \
            megdnn_assert(0);                                                 \
            break;                                                            \
    }

#define DISPATCH_CONV_WINOGRAD_BIAS(_midout_tag, cb, _simd_type, _src_type,  \
                                    _dst_type, _bmode, _nonline_mode, ...)   \
    switch (_bmode) {                                                        \
        case BiasMode::BIAS: {                                               \
            DISPATCH_CONV_WINOGRAD_NONLINE(                                  \
                    _midout_tag, cb, 0, _simd_type, _src_type, _dst_type,    \
                    BiasMode::BIAS, _nonline_mode, __VA_ARGS__)              \
            break;                                                           \
        }                                                                    \
        case BiasMode::NO_BIAS: {                                            \
            DISPATCH_CONV_WINOGRAD_NONLINE(                                  \
                    _midout_tag, cb, 1, _simd_type, _src_type, _dst_type,    \
                    BiasMode::NO_BIAS, _nonline_mode, __VA_ARGS__)           \
            break;                                                           \
        }                                                                    \
        case BiasMode::BROADCAST_CHANNEL_BIAS: {                             \
            DISPATCH_CONV_WINOGRAD_NONLINE(_midout_tag, cb, 2, _simd_type,   \
                                           _src_type, _dst_type,             \
                                           BiasMode::BROADCAST_CHANNEL_BIAS, \
                                           _nonline_mode, __VA_ARGS__)       \
            break;                                                           \
        }                                                                    \
        default:                                                             \
            megdnn_assert(0);                                                \
            break;                                                           \
    }

#define DISPATCH_CONV_WINOGRAD_NONLINE_QUANTIZED(                            \
        _midout_tag, cb, _bias_id, _simd_type, _src_type, _dst_type, _bmode, \
        _nonline_mode, ...)                                                  \
    switch (_nonline_mode) {                                                 \
        case param::ConvBias::NonlineMode::IDENTITY: {                       \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 0) {                         \
                cb(_bmode,                                                   \
                   TypeCvtOp<_simd_type MEGDNN_COMMA _src_type MEGDNN_COMMA  \
                                     _dst_type>,                             \
                   __VA_ARGS__);                                             \
            }                                                                \
            MIDOUT_END();                                                    \
            break;                                                           \
        }                                                                    \
        case param::ConvBias::NonlineMode::RELU: {                           \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 1) {                         \
                cb(_bmode,                                                   \
                   ReluOp<_simd_type MEGDNN_COMMA _src_type MEGDNN_COMMA     \
                                  _dst_type>,                                \
                   __VA_ARGS__);                                             \
            }                                                                \
            MIDOUT_END();                                                    \
            break;                                                           \
        }                                                                    \
        default:                                                             \
            megdnn_assert(0);                                                \
            break;                                                           \
    }

#define DISPATCH_CONV_WINOGRAD_BIAS_QUANTIZED(_midout_tag, cb, _simd_type,  \
                                              _src_type, _dst_type, _bmode, \
                                              _nonline_mode, ...)           \
    switch (_bmode) {                                                       \
        case BiasMode::BIAS: {                                              \
            DISPATCH_CONV_WINOGRAD_NONLINE_QUANTIZED(                       \
                    _midout_tag, cb, 0, _simd_type, _src_type, _dst_type,   \
                    BiasMode::BIAS, _nonline_mode, __VA_ARGS__)             \
            break;                                                          \
        }                                                                   \
        case BiasMode::NO_BIAS: {                                           \
            DISPATCH_CONV_WINOGRAD_NONLINE_QUANTIZED(                       \
                    _midout_tag, cb, 1, _simd_type, _src_type, _dst_type,   \
                    BiasMode::NO_BIAS, _nonline_mode, __VA_ARGS__)          \
            break;                                                          \
        }                                                                   \
        case BiasMode::BROADCAST_CHANNEL_BIAS: {                            \
            DISPATCH_CONV_WINOGRAD_NONLINE_QUANTIZED(                       \
                    _midout_tag, cb, 2, _simd_type, _src_type, _dst_type,   \
                    BiasMode::BROADCAST_CHANNEL_BIAS, _nonline_mode,        \
                    __VA_ARGS__)                                            \
            break;                                                          \
        }                                                                   \
        default:                                                            \
            megdnn_assert(0);                                               \
            break;                                                          \
    }

}  // namespace x86
}  // namespace megdnn
