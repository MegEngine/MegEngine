#pragma once

#include "src/cambricon/utils.mlu.h"

#if CNRT_MAJOR_VERSION >= 5
void checksum_kernel_union1_wrapper(
        uint32_t* dst, const uint32_t* src, int num_elems, cnrtQueue_t queue);
void checksum_kernel_union4_wrapper(
        uint32_t* dst, const uint32_t* src, int num_elems, cnrtQueue_t queue);
#else
#ifdef __cplusplus
extern "C" {
#endif
void checksum_kernel_union1(uint32_t* dst, const uint32_t* src, int num_elems);
void checksum_kernel_union4(uint32_t* dst, const uint32_t* src, int num_elems);
#ifdef __cplusplus
}
#endif
#endif

// vim: ft=cpp syntax=cpp.doxygen
