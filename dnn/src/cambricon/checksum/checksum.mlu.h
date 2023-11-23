#pragma once

#include "src/cambricon/utils.mlu.h"

void checksum_kernel_union1_wrapper(
        uint32_t* dst, const uint32_t* src, int num_elems, cnrtQueue_t queue);
// not support 590
// void checksum_kernel_union4_wrapper(
//         uint32_t* dst, const uint32_t* src, int num_elems, cnrtQueue_t queue);

// vim: ft=cpp syntax=cpp.doxygen
