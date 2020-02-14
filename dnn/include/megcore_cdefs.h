/**
 * \file dnn/include/megcore_cdefs.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <stdint.h>

/**
 * \brief MegCore platform types
 */
typedef enum {
    megcorePlatformCPU = 1,
    megcorePlatformCUDA = 4,
} megcorePlatform_t;

/**
 * \brief MegCore return codes
 *
 * Note: since MegCore has been merged into MegDNN and uses C++ API with
 * exception, this return status only serves for backward compatibility and all
 * API would return megcoreSuccess
 */
typedef enum {
    megcoreSuccess = 0,
    megcoreErrorMemoryAllocation = 1,
    megcoreErrorInvalidArgument = 2,
    megcoreErrorInvalidDeviceHandle = 3,
    megcoreErrorInvalidComputingHandle = 4,
    megcoreErrorInternalError = 5,
} megcoreStatus_t;


/**
 * \brief Memcpy kind
 */
typedef enum {
    megcoreMemcpyHostToDevice = 1,
    megcoreMemcpyDeviceToHost = 2,
    megcoreMemcpyDeviceToDevice = 3,
} megcoreMemcpyKind_t;

namespace megcore {
/*!
 * \brief error reporting from asynchronous execution devices
 *
 * This is currently used by CUDA kernels. It is used to report errors that
 * depend on input data.
 */
struct AsyncErrorInfo {
    //! number of errors occurred; only detailed information of the first error
    //! would be recorded
    uint32_t nr_error;

    //! tracker set by set_error_tracker()
    void* tracker_ptr;

    //! human readable message; it can contain %d which would be replaced by
    //! msg_args
    char msg[228];
    int msg_args[4];
};
} // namespace megcore

// vim: syntax=cpp.doxygen
