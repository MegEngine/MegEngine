/**
 * \file dnn/src/fallback/elemwise/gi_impl/gi_util_impl_helper.h
 */

#pragma once

/*!
 * \brief compute fuse_add_relu on two simd packs
 *
 * Compute
 *
 * val1 = fuse_add_relu(val1, val3)
 * val2 = fuse_add_relu(val2, val4)
 *
 * This algorithm handles int overflow.
 */
#define FUSE_ADD_RELU_SIMD_PACK2_FALLBACK(val1, val2, val3, val4, func_suffix) \
    do {                                                                       \
        val1 = GiMaximum##func_suffix(val1, GiNeg##func_suffix(val3));         \
        val2 = GiMaximum##func_suffix(val2, GiNeg##func_suffix(val4));         \
        val1 = GiAdd##func_suffix(val1, val3);                                 \
        val2 = GiAdd##func_suffix(val2, val4);                                 \
    } while (0)

#define FUSE_ADD_RELU_SIMD_PACK_FALLBACK(val1, val2, func_suffix)      \
    do {                                                               \
        val1 = GiMaximum##func_suffix(val1, GiNeg##func_suffix(val2)); \
        val1 = GiAdd##func_suffix(val1, val2);                         \
    } while (0)

// vim: syntax=cpp.doxygen
