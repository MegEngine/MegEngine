/**
 * \file dnn/src/rocm/utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megcore_cdefs.h"
#include "src/common/utils.h"
#include "megdnn/handle.h"

#include "src/rocm/handle.h"
#include "src/rocm/utils.h.hip"

#include "src/rocm/miopen_with_check.h"
#include <rocblas.h>

namespace megdnn {
namespace rocm {

static inline HandleImpl* concrete_handle(Handle* handle) {
    return static_cast<rocm::HandleImpl*>(handle);
}

static inline miopenHandle_t miopen_handle(Handle* handle) {
    return concrete_handle(handle)->miopen_handle();
}

static inline bool enable_miopen_algo_search(Handle* handle) {
    return concrete_handle(handle)->enable_miopen_algo_search();
}

static inline void enable_miopen_algo_search(Handle* handle,
                                             bool enable_algo_search) {
    return concrete_handle(handle)->enable_miopen_algo_search(
            enable_algo_search);
}

static inline rocblas_handle get_rocblas_handle(Handle* handle) {
    return concrete_handle(handle)->get_rocblas_handle();
}

static inline hipStream_t hip_stream(Handle* handle) {
    return concrete_handle(handle)->stream();
}

static inline megcore::AsyncErrorInfo* async_error_info(Handle* handle) {
    return concrete_handle(handle)->megcore_context().error_info;
}

static inline void callback_free(hipStream_t /* stream */, hipError_t status,
                                 void* userData) {
    hip_check(status);
    free(userData);
}

//! get property of currently active device
hipDeviceProp_t current_device_prop();

//! get the MIOPEN_MAX_BATCH_X_CHANNEL_SIZE, it's just return the max size of
//! the third demension
size_t max_batch_x_channel_size();

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
