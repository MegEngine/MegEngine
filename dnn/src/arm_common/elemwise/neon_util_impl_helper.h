#pragma once

/*!
 * \brief compute fuse_add_relu on two neon packs
 *
 * Compute
 *
 * val1 = fuse_add_relu(val1, val3)
 * val2 = fuse_add_relu(val2, val4)
 *
 * This algorithm handles int overflow.
 */
#define FUSE_ADD_RELU_NEON_PACK2(val1, val2, val3, val4, func_suffix) \
    do {                                                              \
        val1 = vmaxq_##func_suffix(val1, vnegq_##func_suffix(val3));  \
        val2 = vmaxq_##func_suffix(val2, vnegq_##func_suffix(val4));  \
        val1 = vaddq_##func_suffix(val1, val3);                       \
        val2 = vaddq_##func_suffix(val2, val4);                       \
    } while (0)

#define FUSE_ADD_RELU_NEON_PACK(val1, val2, func_suffix)             \
    do {                                                             \
        val1 = vmaxq_##func_suffix(val1, vnegq_##func_suffix(val2)); \
        val1 = vaddq_##func_suffix(val1, val2);                      \
    } while (0)

// vim: syntax=cpp.doxygen
