/**
 * \file dnn/src/fallback/elemwise_helper/op_common.h
 */
#pragma once

namespace megdnn {
/*!
 * \brief broadcast type
 * BCAST_x[0]x[1]...: x[i] == !stride[i]
 */
enum BcastType {
    VEC,
    VEC_VEC,
    VEC_BCAST101,
    VEC_BCASTX0X,
    VEC_BCAST111C,
    VEC_BCAST101xX,
    VEC_SCALAR,
    SCALAR_VEC,
    BCAST101_VEC,
    BCASTX0X_VEC,
    BCAST111C_VEC,
    BCAST101xX_VEC,
    VEC_VEC_VEC,
    VEC_VEC_SCALAR,
    BCAST101_VEC_BCAST101,
    BCAST111C_VEC_BCAST111C,
    BCAST101xX_VEC_BCAST101xX,
    VEC_BCAST101_VEC,
    VEC_BCAST111C_VEC,
    VEC_BCAST101xX_VEC,
    VEC_SCALAR_VEC,
    VEC_SCALAR_SCALAR,
    UNKNOWN_BCAST_TYPE
};

}  // namespace megdnn

// vim: syntax=cpp.doxygen
