#pragma once
#include "src/common/unroll_macro.h"

#define MATRIX_MUL4x4(sum, a, b)                   \
    sum##0 = vmla_lane_s16(sum##0, b##0, a##0, 0); \
    sum##0 = vmla_lane_s16(sum##0, b##1, a##0, 1); \
    sum##0 = vmla_lane_s16(sum##0, b##2, a##0, 2); \
    sum##0 = vmla_lane_s16(sum##0, b##3, a##0, 3); \
    sum##1 = vmla_lane_s16(sum##1, b##0, a##1, 0); \
    sum##1 = vmla_lane_s16(sum##1, b##1, a##1, 1); \
    sum##1 = vmla_lane_s16(sum##1, b##2, a##1, 2); \
    sum##1 = vmla_lane_s16(sum##1, b##3, a##1, 3); \
    sum##2 = vmla_lane_s16(sum##2, b##0, a##2, 0); \
    sum##2 = vmla_lane_s16(sum##2, b##1, a##2, 1); \
    sum##2 = vmla_lane_s16(sum##2, b##2, a##2, 2); \
    sum##2 = vmla_lane_s16(sum##2, b##3, a##2, 3); \
    sum##3 = vmla_lane_s16(sum##3, b##0, a##3, 0); \
    sum##3 = vmla_lane_s16(sum##3, b##1, a##3, 1); \
    sum##3 = vmla_lane_s16(sum##3, b##2, a##3, 2); \
    sum##3 = vmla_lane_s16(sum##3, b##3, a##3, 3);

// vim: syntax=cpp.doxygen
