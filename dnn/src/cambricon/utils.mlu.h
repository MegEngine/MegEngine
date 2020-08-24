/**
 * \file dnn/src/cambricon/utils.mlu.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/common/utils.cuh"

#include <stdint.h>

#include <cnrt.h>

#define cnrt_check(_x)                                            \
    do {                                                          \
        cnrtRet_t _ret = (_x);                                    \
        if (_ret != CNRT_RET_SUCCESS) {                           \
            ::megdnn::cambricon::__throw_cnrt_error__(_ret, #_x); \
        }                                                         \
    } while (0)

#define after_kernel_launch()         \
    do {                              \
        cnrt_check(cnrtGetLastErr()); \
    } while (0)

namespace megdnn {
namespace cambricon {

//! Error handling funcions
MEGDNN_NORETURN void __throw_cnrt_error__(cnrtRet_t err, const char* msg);

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen

