#pragma once

#define MEGDNN_MAX_NDIM 7

/*!
 * \brief iterate through small (usually used) ndim values
 */
#define MEGDNN_FOREACH_TENSOR_NDIM_SMALL(cb, ...) \
    cb(1, ##__VA_ARGS__) cb(2, ##__VA_ARGS__) cb(3, ##__VA_ARGS__)

/*!
 * \brief iterate through large (rarely used) ndim values
 */
#define MEGDNN_FOREACH_TENSOR_NDIM_LARGE(cb, ...) \
    cb(4, ##__VA_ARGS__) cb(5, ##__VA_ARGS__) cb(6, ##__VA_ARGS__) cb(7, ##__VA_ARGS__)

/*!
 * \brief iterate through all ndim values
 */
#define MEGDNN_FOREACH_TENSOR_NDIM(cb, ...)             \
    MEGDNN_FOREACH_TENSOR_NDIM_SMALL(cb, ##__VA_ARGS__) \
    MEGDNN_FOREACH_TENSOR_NDIM_LARGE(cb, ##__VA_ARGS__)

// vim: syntax=cpp.doxygen
