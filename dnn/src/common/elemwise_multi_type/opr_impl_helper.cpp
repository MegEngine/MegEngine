/**
 * \file dnn/src/common/elemwise_multi_type/opr_impl_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl_helper.h"
#include "src/common/utils.h"

using namespace megdnn;

#define ON_QUANTIZED_MODE(_MODE, _n)                                 \
    case Mode::Q##_MODE:                                             \
        on_quantized_mode(make_elemwise_op_param<_n>(src, dst), dst, \
                          Elemwise::Mode::_MODE);                    \
        break

void ElemwiseMultiTypeImplHelper::exec(_megdnn_in const TensorNDArray& src,
                                       _megdnn_tensor_out dst) {
    switch (m_param.mode) {
        case Mode::FUSE_MUL_ADD3_INT16x32x32x32:
            on_fuse_mul_add3_int16x32x32x32(make_elemwise_op_param<3>(src, dst),
                                            dst.ptr<dt_int32>());
            break;
        case Mode::FUSE_MUL_ADD3_IXxF32xF32xI8:
            on_fuse_mul_add3_iXxf32xf32xi8(make_elemwise_op_param<3>(src, dst),
                                           dst.ptr<dt_int8>());
            break;
        case Mode::ROUND_SHR_SATURATE_IXxI8xI8:
            on_round_shr_saturate_iXxi8xi8(make_elemwise_op_param<2>(src, dst),
                                           dst.ptr<dt_int8>());
            break;
        case Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8:
            on_fuse_add_rmulh_round_shr_saturate_int16x16x16x8(
                    make_elemwise_op_param<6>(src, dst), dst.ptr<dt_int8>());
            break;
        case Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8:
            on_fuse_add_rmulh_round_shr_saturate_int32x32x32x8(
                    make_elemwise_op_param<6>(src, dst), dst.ptr<dt_int8>());
            break;
        case Mode::ROUND_SHR_SATURATE_IXxI8xI16:
            on_round_shr_saturate_iXxi8xi16(make_elemwise_op_param<2>(src, dst),
                                            dst.ptr<dt_int16>());
            break;
        ON_QUANTIZED_MODE(RELU, 1);
        ON_QUANTIZED_MODE(ABS, 1);
        ON_QUANTIZED_MODE(ACOS, 1);
        ON_QUANTIZED_MODE(ASIN, 1);
        ON_QUANTIZED_MODE(CEIL, 1);
        ON_QUANTIZED_MODE(COS, 1);
        ON_QUANTIZED_MODE(EXP, 1);
        ON_QUANTIZED_MODE(EXPM1, 1);
        ON_QUANTIZED_MODE(FLOOR, 1);
        ON_QUANTIZED_MODE(LOG, 1);
        ON_QUANTIZED_MODE(LOG1P, 1);
        ON_QUANTIZED_MODE(NEGATE, 1);
        ON_QUANTIZED_MODE(SIGMOID, 1);
        ON_QUANTIZED_MODE(SIN, 1);
        ON_QUANTIZED_MODE(TANH, 1);
        ON_QUANTIZED_MODE(FAST_TANH, 1);
        ON_QUANTIZED_MODE(ROUND, 1);
        ON_QUANTIZED_MODE(ERF, 1);
        ON_QUANTIZED_MODE(ERFINV, 1);
        ON_QUANTIZED_MODE(ERFC, 1);
        ON_QUANTIZED_MODE(ERFCINV, 1);
        ON_QUANTIZED_MODE(H_SWISH, 1);

        ON_QUANTIZED_MODE(ABS_GRAD, 2);
        ON_QUANTIZED_MODE(ADD, 2);
        ON_QUANTIZED_MODE(FLOOR_DIV, 2);
        ON_QUANTIZED_MODE(MAX, 2);
        ON_QUANTIZED_MODE(MIN, 2);
        ON_QUANTIZED_MODE(MOD, 2);
        ON_QUANTIZED_MODE(MUL, 2);
        ON_QUANTIZED_MODE(POW, 2);
        ON_QUANTIZED_MODE(SIGMOID_GRAD, 2);
        ON_QUANTIZED_MODE(SUB, 2);
        ON_QUANTIZED_MODE(SWITCH_GT0, 2);
        ON_QUANTIZED_MODE(TANH_GRAD, 2);
        ON_QUANTIZED_MODE(TRUE_DIV, 2);
        ON_QUANTIZED_MODE(LOG_SUM_EXP, 2);

        ON_QUANTIZED_MODE(LT, 2);
        ON_QUANTIZED_MODE(LEQ, 2);
        ON_QUANTIZED_MODE(EQ, 2);

        ON_QUANTIZED_MODE(FUSE_ADD_RELU, 2);
        ON_QUANTIZED_MODE(FUSE_ADD_SIGMOID, 2);
        ON_QUANTIZED_MODE(FUSE_ADD_TANH, 2);
        ON_QUANTIZED_MODE(FAST_TANH_GRAD, 2);
        ON_QUANTIZED_MODE(ATAN2, 2);
        ON_QUANTIZED_MODE(H_SWISH_GRAD, 2);
        ON_QUANTIZED_MODE(FUSE_ADD_H_SWISH, 2);

        ON_QUANTIZED_MODE(FUSE_MUL_ADD3, 3);
        ON_QUANTIZED_MODE(COND_LEQ_MOV, 3);
        default:
            megdnn_throw("invalid mode");
    }
}

#undef ON_QUANTIZED_MODE

// vim: ft=cpp syntax=cpp.doxygen
