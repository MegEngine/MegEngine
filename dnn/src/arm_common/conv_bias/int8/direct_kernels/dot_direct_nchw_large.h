#pragma once
#include "megdnn/arch.h"
#if MGB_ENABLE_DOT

void megdnn_dot_nchw_large_chanwise_direct_conv_9x9s1_oh4_ow16(
        const int8_t* src, const int8_t* weight, int32_t bias, int8_t* dst, size_t oh,
        size_t ow, size_t OH, size_t OW, size_t pad_iw, const float scale,
        int8_t relu_val);

void megdnn_dot_nchw_large_chanwise_direct_conv_9x9s2_oh4_ow16(
        const int8_t* src, const int8_t* weight, int32_t bias, int8_t* dst, size_t oh,
        size_t ow, size_t OH, size_t OW, size_t pad_iw, const float scale,
        int8_t relu_val);

#endif