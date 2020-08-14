/**
 * \file dnn/src/arm_common/conv_bias/postprocess_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/basic_types.h"
#include "src/arm_common/elemwise_helper/kimpl/op_base.h"
#include "src/arm_common/elemwise_op.h"
#include "src/fallback/conv_bias/opr_impl.h"

#include "midout.h"

MIDOUT_DECL(arm_common_conv_bias_postprocess_helper)

namespace {


#define CONCAT_OP(_name) megdnn::arm_common::_name
#define CONCAT_NL(_name) megdnn::NonlineMode::_name

#define CB(_caller, _op, _mode, midout_tag)                                    \
    case _mode:                                                                \
        MIDOUT_BEGIN(arm_common_conv_bias_postprocess_helper, 1, midout_tag) { \
            _caller(_op);                                                      \
        }                                                                      \
        MIDOUT_END();                                                          \
        break;

#define DEFAULT                                 \
    default:                                    \
        megdnn_throw("unsupported nolinemode"); \
        break;

#define HANDLE_IDENTITY()               \
    case megdnn::NonlineMode::IDENTITY: \
        break;

#define FOR_NONLINEAR_UNARY(_op)                                             \
    megdnn::arm_common::OpCallerUnary<_op<ctype>, megdnn::arm_common::VEC>:: \
            run(static_cast<ctype*>(conv_dst_ptr),                           \
                reinterpret_cast<ctype*>(dst_ptr), bias_type, dst_type,      \
                N* OC* OH* OW* pack_oc_size);

#define FOR_NONLINEAR_BINARY_BROADCAST(_op)                                    \
    megdnn::arm_common::                                                       \
            OpCallerBinary<_op<ctype>, megdnn::arm_common::VEC_BCAST101>::run( \
                    static_cast<ctype*>(conv_dst_ptr),                         \
                    reinterpret_cast<const ctype*>(bias_ptr),                  \
                    reinterpret_cast<ctype*>(dst_ptr), bias_type, bias_type,   \
                    dst_type, N, OC, OH* OW);

#define FOR_NONLINEAR_BINARY_BROADCAST_NCHW44(_op)                           \
    megdnn::arm_common::OpCallerBinary<_op<ctype>,                           \
                                       megdnn::arm_common::VEC_BCAST101x4>:: \
            run(static_cast<ctype*>(conv_dst_ptr),                           \
                reinterpret_cast<const ctype*>(bias_ptr),                    \
                reinterpret_cast<ctype*>(dst_ptr), bias_type, bias_type,     \
                dst_type, N, OC, OH* OW, pack_oc_size);

#define FOR_NONLINEAR_BINARY(_op)                                            \
    megdnn::arm_common::                                                     \
            OpCallerBinary<_op<ctype>, megdnn::arm_common::VEC_VEC>::run(    \
                    static_cast<ctype*>(conv_dst_ptr),                       \
                    reinterpret_cast<const ctype*>(bias_ptr),                \
                    reinterpret_cast<ctype*>(dst_ptr), bias_type, bias_type, \
                    dst_type, N* OC* OH* OW* pack_oc_size);

#define FOR_BIAS(_mode)                                                   \
    switch (_mode) {                                                      \
        case megdnn::BiasMode::NO_BIAS:                                   \
            MIDOUT_BEGIN(arm_common_conv_bias_postprocess_helper, 0, 0) { \
                FOR_NONLINEAR_NOBIAS(FOR_NONLINEAR_UNARY);                \
            }                                                             \
            MIDOUT_END();                                                 \
            break;                                                        \
        case megdnn::BiasMode::BROADCAST_CHANNEL_BIAS:                    \
            MIDOUT_BEGIN(arm_common_conv_bias_postprocess_helper, 0, 1) { \
                if (pack_oc_size == 1) {                                  \
                    FOR_NONLINEAR(FOR_NONLINEAR_BINARY_BROADCAST);        \
                } else {                                                  \
                    megdnn_assert(pack_oc_size == 4,                      \
                                  "Only support nchw44 in ARM");          \
                    FOR_NONLINEAR(FOR_NONLINEAR_BINARY_BROADCAST_NCHW44); \
                }                                                         \
            }                                                             \
            MIDOUT_END();                                                 \
            break;                                                        \
        case megdnn::BiasMode::BIAS:                                      \
            MIDOUT_BEGIN(arm_common_conv_bias_postprocess_helper, 0, 2) { \
                FOR_NONLINEAR(FOR_NONLINEAR_BINARY);                      \
            }                                                             \
            MIDOUT_END();                                                 \
            break;                                                        \
        default:                                                          \
            megdnn_throw("unknow biasmode");                             \
            break;                                                        \
    }

#define FOR_NONLINEAR(_caller)                                          \
    switch (nonlineMode) {                                              \
        CB(_caller, CONCAT_OP(AddOp), CONCAT_NL(IDENTITY), 3)           \
        CB(_caller, CONCAT_OP(FuseAddReluOp), CONCAT_NL(RELU), 4)       \
        CB(_caller, CONCAT_OP(FuseAddSigmoidOp), CONCAT_NL(SIGMOID), 5) \
        CB(_caller, CONCAT_OP(FuseAddHSwishOp), CONCAT_NL(H_SWISH), 6)  \
        DEFAULT                                                         \
    }

#define FOR_NONLINEAR_NOBIAS(_caller)                             \
    switch (nonlineMode) {                                        \
        HANDLE_IDENTITY()                                         \
        CB(_caller, CONCAT_OP(ReluOp), CONCAT_NL(RELU), 7);       \
        CB(_caller, CONCAT_OP(SigmoidOp), CONCAT_NL(SIGMOID), 8); \
        CB(_caller, CONCAT_OP(HSwishOp), CONCAT_NL(H_SWISH), 9);  \
        DEFAULT                                                   \
    }

template <typename ctype, typename dtype = ctype,
          megdnn::PostprocessMode postprocess_mode =
                  megdnn::PostprocessMode::FLOAT>
struct PostProcess {
    static void run(void* conv_dst_ptr, const void* bias_ptr, void* dst_ptr,
                    megdnn::BiasMode bias_mode, megdnn::NonlineMode nonlineMode,
                    megdnn::DType bias_type, megdnn::DType dst_type, size_t N,
                    size_t OC, size_t OH, size_t OW, size_t pack_oc_size = 1) {
        FOR_BIAS(bias_mode)
    }
};

template <typename ctype, typename dtype>
struct PostProcess<ctype, dtype, megdnn::PostprocessMode::NO_PROCESS> {
    static void run(void* conv_dst_ptr, void* bias_ptr, void* dst_ptr,
                    megdnn::BiasMode bias_mode, megdnn::NonlineMode nonlineMode,
                    megdnn::DType bias_type, megdnn::DType dst_type, size_t N,
                    size_t OC, size_t OH, size_t OW, size_t pack_oc_size = 1) {
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
        MEGDNN_MARK_USED_VAR(pack_oc_size);
        megdnn_assert(bias_mode == megdnn::BiasMode::NO_BIAS &&
                      nonlineMode == megdnn::NonlineMode::IDENTITY);
    }
};

#undef FOR_NONLINEAR_UNARY
#undef FOR_NONLINEAR_BINARY_BROADCAST
#undef FOR_NONLINEAR_BINARY_BROADCAST_NCHW44
#undef FOR_NONLINEAR_BINARY
#undef FOR_NONLINEAR_NOBIAS
#undef FOR_NONLINEAR
#undef FOR_BIAS
#undef HANDLE_IDENTITY

#define FOR_NONLINEAR_UNARY(_op)                                               \
    megdnn::arm_common::OpCallerUnary<                                         \
            _op<opctype, opdtype>,                                             \
            megdnn::arm_common::VEC>::run(static_cast<opctype*>(conv_dst_ptr), \
                                          reinterpret_cast<opdtype*>(dst_ptr), \
                                          bias_type, dst_type,                 \
                                          N* OC* OH* OW* pack_oc_size);

#define FOR_NONLINEAR_BINARY_BROADCAST(_op)                                \
    megdnn::arm_common::OpCallerBinary<_op<opctype, opdtype>,              \
                                       megdnn::arm_common::VEC_BCAST101>:: \
            run(static_cast<opctype*>(conv_dst_ptr),                       \
                reinterpret_cast<const opctype*>(bias_ptr),                \
                reinterpret_cast<opdtype*>(dst_ptr), bias_type, bias_type, \
                dst_type, N, OC, OH* OW);

#define FOR_NONLINEAR_BINARY_BROADCAST_NCHW44(_op)                           \
    megdnn::arm_common::OpCallerBinary<_op<opctype, opdtype>,                \
                                       megdnn::arm_common::VEC_BCAST101x4>:: \
            run(static_cast<opctype*>(conv_dst_ptr),                         \
                reinterpret_cast<const opctype*>(bias_ptr),                  \
                reinterpret_cast<opdtype*>(dst_ptr), bias_type, bias_type,   \
                dst_type, N, OC, OH* OW, pack_oc_size);

#define HANDLE_IDENTITY(_caller, _op)              \
    case megdnn::NonlineMode::IDENTITY:            \
        _caller(_op) break;

#define FOR_NONLINEAR(_caller)                                          \
    switch (nonlineMode) {                                              \
        HANDLE_IDENTITY(_caller, CONCAT_OP(AddOp))                      \
        CB(_caller, CONCAT_OP(FuseAddReluOp), CONCAT_NL(RELU), 10)      \
        CB(_caller, CONCAT_OP(FuseAddHSwishOp), CONCAT_NL(H_SWISH), 11) \
        DEFAULT                                                         \
    }

#define FOR_NONLINEAR_NOBIAS(_caller)                            \
    switch (nonlineMode) {                                       \
        HANDLE_IDENTITY(_caller, CONCAT_OP(TypeCvtOp))           \
        CB(_caller, CONCAT_OP(ReluOp), CONCAT_NL(RELU), 12)      \
        CB(_caller, CONCAT_OP(HSwishOp), CONCAT_NL(H_SWISH), 13) \
        DEFAULT                                                  \
    }

#define FOR_BIAS(_bias_mode, OH, OW)                                      \
    switch (_bias_mode) {                                                 \
        case megdnn::BiasMode::NO_BIAS:                                   \
            FOR_NONLINEAR_NOBIAS(FOR_NONLINEAR_UNARY);                    \
            break;                                                        \
        case megdnn::BiasMode::BROADCAST_CHANNEL_BIAS:                    \
            if (pack_oc_size == 1) {                                      \
                FOR_NONLINEAR(FOR_NONLINEAR_BINARY_BROADCAST);            \
            } else {                                                      \
                megdnn_assert(pack_oc_size == 4,                          \
                              "Only support nchw44 in ARM");              \
                FOR_NONLINEAR(FOR_NONLINEAR_BINARY_BROADCAST_NCHW44);     \
            }                                                             \
            break;                                                        \
        default:                                                          \
            if (OH * OW == 1) {                                           \
                if (pack_oc_size == 1) {                                  \
                    FOR_NONLINEAR(FOR_NONLINEAR_BINARY_BROADCAST);        \
                } else {                                                  \
                    megdnn_assert(pack_oc_size == 4,                      \
                                  "Only support nchw44 in ARM");          \
                    FOR_NONLINEAR(FOR_NONLINEAR_BINARY_BROADCAST_NCHW44); \
                }                                                         \
                break;                                                    \
            }                                                             \
            megdnn_throw("quantized unsupported biasmode");               \
            break;                                                        \
    }

template <typename opctype, typename opdtype>
struct PostProcess<opctype, opdtype, megdnn::PostprocessMode::QUANTIZED> {
    static void run(void* conv_dst_ptr, const void* bias_ptr, void* dst_ptr,
                    megdnn::BiasMode bias_mode, megdnn::NonlineMode nonlineMode,
                    megdnn::DType bias_type, megdnn::DType dst_type, size_t N,
                    size_t OC, size_t OH, size_t OW, size_t pack_oc_size = 1) {
        //! when OH * OW = 1, the bias_mode will be BiasMode::BIAS. It is wrong,
        //! we deal this case at default branch.
        FOR_BIAS(bias_mode, OH, OW);
    }
};

#undef FOR_NONLINEAR_UNARY
#undef FOR_NONLINEAR_BINARY_BROADCAST
#undef FOR_NONLINEAR_BINARY_BROADCAST_NCHW44
#undef FOR_NONLINEAR_BINARY
#undef FOR_NONLINEAR_NOBIAS
#undef FOR_NONLINEAR
#undef FOR_BIAS

#define FOR_BINARY_BROADCAST(_op)                                              \
    megdnn::arm_common::                                                       \
            OpCallerBinary<_op<ctype>, megdnn::arm_common::VEC_BCAST101>::run( \
                    static_cast<ctype*>(conv_dst_ptr),                         \
                    reinterpret_cast<const ctype*>(bias_ptr),                  \
                    reinterpret_cast<ctype*>(dst_ptr), bias_type, bias_type,   \
                    dst_type, N, OC, OH* OW);

#define FOR_BINARY_BROADCAST_NCHW44(_op)                                     \
    megdnn::arm_common::OpCallerBinary<_op<ctype>,                           \
                                       megdnn::arm_common::VEC_BCAST101x4>:: \
            run(static_cast<ctype*>(conv_dst_ptr),                           \
                reinterpret_cast<const ctype*>(bias_ptr),                    \
                reinterpret_cast<ctype*>(dst_ptr), bias_type, bias_type,     \
                dst_type, N, OC, OH* OW, pack_oc_size);

#define FOR_BINARY(_op)                                                      \
    megdnn::arm_common::                                                     \
            OpCallerBinary<_op<ctype>, megdnn::arm_common::VEC_VEC>::run(    \
                    static_cast<ctype*>(conv_dst_ptr),                       \
                    reinterpret_cast<const ctype*>(bias_ptr),                \
                    reinterpret_cast<ctype*>(dst_ptr), bias_type, bias_type, \
                    dst_type, N* OC* OH* OW* pack_oc_size);

#define FOR_BIAS(_bias_mode, OH, OW)                           \
    switch (_bias_mode) {                                      \
        case megdnn::BiasMode::NO_BIAS:                        \
            break;                                             \
        case megdnn::BiasMode::BROADCAST_CHANNEL_BIAS:         \
            if (pack_oc_size == 1) {                           \
                FOR_BINARY_BROADCAST(CONCAT_OP(AddOp));        \
            } else {                                           \
                megdnn_assert(pack_oc_size == 4,               \
                              "Only support nchw44 in ARM");   \
                FOR_BINARY_BROADCAST_NCHW44(CONCAT_OP(AddOp)); \
            }                                                  \
            break;                                             \
        case megdnn::BiasMode::BIAS:                           \
            FOR_BINARY(CONCAT_OP(AddOp));                      \
            break;                                             \
        default:                                               \
            megdnn_throw("unknow biasmode");                   \
            break;                                             \
    }

template <typename ctype, typename dtype>
struct PostProcess<ctype, dtype, megdnn::PostprocessMode::ADD_BIAS> {
    static void run(void* conv_dst_ptr, void* bias_ptr, void* dst_ptr,
                    megdnn::BiasMode bias_mode, megdnn::NonlineMode nonlineMode,
                    megdnn::DType bias_type, megdnn::DType dst_type, size_t N,
                    size_t OC, size_t OH, size_t OW, size_t pack_oc_size = 1) {
        megdnn_assert(nonlineMode == megdnn::NonlineMode::IDENTITY);
        FOR_BIAS(bias_mode, OH, OW);
    }
};

#undef FOR_BINARY_BROADCAST
#undef FOR_BINARY_BROADCAST_NCHW44
#undef FOR_BINARY
#undef FOR_BIAS
#undef CB
#undef CONCAT_OP
#undef CONCAT_NL
#undef DEFAULT
#undef HANDLE_IDENTITY

#define DISPATCH_CONV_WINOGRAD_NONLINE(_midout_tag, cb, _bias_id, _src_type,  \
                                       _dst_type, _bmode, _nonline_mode, ...) \
    switch (_nonline_mode) {                                                  \
        case param::ConvBias::NonlineMode::IDENTITY: {                        \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 0) {                          \
                cb(_bmode, NoneOp<_src_type MEGDNN_COMMA _dst_type>,          \
                   __VA_ARGS__);                                              \
            }                                                                 \
            MIDOUT_END();                                                     \
            break;                                                            \
        }                                                                     \
        case param::ConvBias::NonlineMode::RELU: {                            \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 1) {                          \
                cb(_bmode, ReluOp<_src_type MEGDNN_COMMA _dst_type>,          \
                   __VA_ARGS__);                                              \
            }                                                                 \
            MIDOUT_END();                                                     \
            break;                                                            \
        }                                                                     \
        case param::ConvBias::NonlineMode::SIGMOID: {                         \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 2) {                          \
                cb(_bmode, SigmoidOp<_src_type MEGDNN_COMMA _dst_type>,       \
                   __VA_ARGS__);                                              \
            }                                                                 \
            MIDOUT_END();                                                     \
            break;                                                            \
        }                                                                     \
        case param::ConvBias::NonlineMode::H_SWISH: {                         \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 3) {                          \
                cb(_bmode, HSwishOp<_src_type MEGDNN_COMMA _dst_type>,        \
                   __VA_ARGS__);                                              \
            }                                                                 \
            MIDOUT_END();                                                     \
            break;                                                            \
        }                                                                     \
        default:                                                              \
            megdnn_assert(0);                                                 \
            break;                                                            \
    }

#define DISPATCH_CONV_WINOGRAD_NONLINE_QUANTIZED(_midout_tag, cb, _bias_id,    \
                                                 _src_type, _dst_type, _bmode, \
                                                 _nonline_mode, ...)           \
    switch (_nonline_mode) {                                                   \
        case param::ConvBias::NonlineMode::IDENTITY: {                         \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 0) {                           \
                cb(_bmode, TypeCvtOp<_src_type MEGDNN_COMMA _dst_type>,        \
                   __VA_ARGS__);                                               \
            }                                                                  \
            MIDOUT_END();                                                      \
            break;                                                             \
        }                                                                      \
        case param::ConvBias::NonlineMode::RELU: {                             \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 1) {                           \
                cb(_bmode, ReluOp<_src_type MEGDNN_COMMA _dst_type>,           \
                   __VA_ARGS__);                                               \
            }                                                                  \
            MIDOUT_END();                                                      \
            break;                                                             \
        }                                                                      \
        default:                                                               \
            megdnn_assert(0);                                                  \
            break;                                                             \
    }

#define DISPATCH_CONV_WINOGRAD_BIAS(_midout_tag, cb, _src_type, _dst_type,   \
                                    _bmode, _nonline_mode, ...)              \
    switch (_bmode) {                                                        \
        case BiasMode::BIAS: {                                               \
            DISPATCH_CONV_WINOGRAD_NONLINE(_midout_tag, cb, 0, _src_type,    \
                                           _dst_type, BiasMode::BIAS,        \
                                           _nonline_mode, __VA_ARGS__)       \
            break;                                                           \
        }                                                                    \
        case BiasMode::NO_BIAS: {                                            \
            DISPATCH_CONV_WINOGRAD_NONLINE(_midout_tag, cb, 1, _src_type,    \
                                           _dst_type, BiasMode::NO_BIAS,     \
                                           _nonline_mode, __VA_ARGS__)       \
            break;                                                           \
        }                                                                    \
        case BiasMode::BROADCAST_CHANNEL_BIAS: {                             \
            DISPATCH_CONV_WINOGRAD_NONLINE(_midout_tag, cb, 2, _src_type,    \
                                           _dst_type,                        \
                                           BiasMode::BROADCAST_CHANNEL_BIAS, \
                                           _nonline_mode, __VA_ARGS__)       \
            break;                                                           \
        }                                                                    \
        default:                                                             \
            megdnn_assert(0);                                                \
            break;                                                           \
    }

#define DISPATCH_CONV_WINOGRAD_BIAS_QUANTIZED(                                \
        _midout_tag, cb, _src_type, _dst_type, _bmode, _nonline_mode, ...)    \
    switch (_bmode) {                                                         \
        case BiasMode::BIAS: {                                                \
            DISPATCH_CONV_WINOGRAD_NONLINE_QUANTIZED(                         \
                    _midout_tag, cb, 0, _src_type, _dst_type, BiasMode::BIAS, \
                    _nonline_mode, __VA_ARGS__)                               \
            break;                                                            \
        }                                                                     \
        case BiasMode::NO_BIAS: {                                             \
            DISPATCH_CONV_WINOGRAD_NONLINE_QUANTIZED(                         \
                    _midout_tag, cb, 1, _src_type, _dst_type,                 \
                    BiasMode::NO_BIAS, _nonline_mode, __VA_ARGS__)            \
            break;                                                            \
        }                                                                     \
        case BiasMode::BROADCAST_CHANNEL_BIAS: {                              \
            DISPATCH_CONV_WINOGRAD_NONLINE_QUANTIZED(                         \
                    _midout_tag, cb, 2, _src_type, _dst_type,                 \
                    BiasMode::BROADCAST_CHANNEL_BIAS, _nonline_mode,          \
                    __VA_ARGS__)                                              \
            break;                                                            \
        }                                                                     \
        default:                                                              \
            megdnn_assert(0);                                                 \
            break;                                                            \
    }

}  // namespace
