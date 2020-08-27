/**
 * \file src/rocm/error_info.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */

#pragma once

#include "hip_header.h"
#include "megcore_cdefs.h"
#include "megdnn/arch.h"

typedef megcore::AsyncErrorInfo AsyncErrorInfo;
#if MEGDNN_CC_CUDA
// we can not put this function into anonymous namespace, since it would cause
// unused static func or undefined static func warning depending on whether you
// define it
namespace {
#endif

__device__ void set_async_error_info(AsyncErrorInfo* info, void* tracker,
                                     const char* msg, int arg0 = 0,
                                     int arg1 = 0, int arg2 = 0, int arg3 = 0)
#if MEGDNN_CC_CUDA
{
    if (info && !atomicAdd(&info->nr_error, 1)) {
        // use atomic expression to ensure that only the first error is reported
        info->tracker_ptr = tracker;
        char* ptr = info->msg;
        char* ptr_end = ptr + sizeof(AsyncErrorInfo::msg) - 1;
        while (ptr < ptr_end && *msg) {
            *(ptr++) = *(msg++);
        }
        *ptr = 0;
        info->msg_args[0] = arg0;
        info->msg_args[1] = arg1;
        info->msg_args[2] = arg2;
        info->msg_args[3] = arg3;
    }
}
#else
        ;
#endif

#if MEGDNN_CC_CUDA
}  // anonymous namespace
#endif

// vim: ft=cpp syntax=cpp.doxygen
