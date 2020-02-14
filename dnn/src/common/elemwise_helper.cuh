/**
 * \file dnn/src/common/elemwise_helper.cuh
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

namespace {

template <typename T>
struct MulType {};
template<> struct MulType<int8_t> { typedef int16_t type; };
template<> struct MulType<int16_t> { typedef int32_t type; };
template<> struct MulType<int32_t> { typedef int64_t type; };
template<> struct MulType<uint8_t> { typedef uint16_t type; };

}  // namespace

namespace megdnn {

/*!
    * \brief packed param for elemwise operators
    * \tparam arity number of operands for this operator
    */
template<int arity>
struct ElemwiseOpParamN {
    int max_ndim;   //!< max ndim of all params
    size_t size;    //!< total number of elements (i.e. size of each param)

    TensorND param[arity];

    ElemwiseOpParamN():
        max_ndim(-1), size(0)
    {}

    const TensorND& operator [](int idx) const {
        return param[idx];
    }

    TensorND& operator [](int idx) {
        return param[idx];
    }

    /*!
        * \brief initialize from current *param*
        *
        * *size* and *max_ndim* would be computed; params would be collapsed
        *
        * Each param must have the same number of elements.
        */
    void init_from_given_tensor();

    void assert_initialized() const;
};

/*!
    * \brief for elemwise opr without tensor arguments (i.e. only need index input)
    */
template<>
struct ElemwiseOpParamN<0> {
    size_t size; //!< total number of elements

    ElemwiseOpParamN(size_t s = 0):
        size(s)
    {
    }

    void assert_initialized() const;
};

template <typename T>
MEGDNN_DEVICE MEGDNN_HOST inline T rounding_shift_right_away_from_zero(T x,
                                                                       int k) {
    T mask = (T(1) << k) - 1;
    T threshold = (mask >> 1) + (x < 0);
    return (x >> k) + ((x & mask) > threshold);
}

template <typename T>
MEGDNN_DEVICE MEGDNN_HOST inline T rounding_shift_right_upward(T x, int k) {
    T mask = (T(1) << k) - 1;
    T threshold = mask >> 1;
    return (x >> k) + ((x & mask) > threshold);
}

template <typename T>
MEGDNN_DEVICE MEGDNN_HOST inline T round_mulh_saturate(T a, T b) {
    MEGDNN_STATIC_ASSERT(std::numeric_limits<T>::digits <= 32,
                            "Portable RMULH is not supported for integer "
                            "types larger than 32 bits.");
    MEGDNN_STATIC_ASSERT(std::numeric_limits<T>::is_integer,
                            "Input types should be integer for RMULH");
    bool overflow = a == b && a == DTypeTrait<T>::min();
    // TODO: This really should be
    // rounding_shift_right_away_from_zero, but we haven't yet found a fast way
    // to implement it on ARM NEON. For now, we just try to align with NEON's
    // VQRDMULH and hope that it does not harm our NN badly.
    return overflow ? DTypeTrait<T>::max()
                    : static_cast<T>(rounding_shift_right_upward(
                              typename MulType<T>::type(a) *
                                      typename MulType<T>::type(b),
                              std::numeric_limits<T>::digits));
}

} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen

