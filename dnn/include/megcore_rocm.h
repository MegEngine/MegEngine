/**
 * \file dnn/include/megcore_rocm.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./megcore.h"

#include "hip_header.h"
#include "megdnn/internal/visibility_prologue.h"

#include <atomic>

namespace megcore {
struct ROCMContext {
    hipStream_t stream = nullptr;

    static std::atomic_bool sm_miopen_algo_search;
    static inline bool enable_miopen_algo_search() { return sm_miopen_algo_search.load(); }
    static inline void enable_miopen_algo_search(bool enable_algo_search) {
        sm_miopen_algo_search.store(enable_algo_search);
    }

    //! device pointer to buffer for error reporting from kernels
    AsyncErrorInfo* error_info = nullptr;

    ROCMContext() = default;

    ROCMContext(hipStream_t s, AsyncErrorInfo* e) : stream{s}, error_info{e} {}
};

megcoreStatus_t createComputingHandleWithROCMContext(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags, const ROCMContext& ctx);

megcoreStatus_t getROCMContext(megcoreComputingHandle_t handle,
                               ROCMContext* ctx);

// Set MIOpen algo search enabled or disabled
megcoreStatus_t enableMIOpenAlgoSearch(bool enable_algo_search = true);

// Find out whether MIOpen algo search is enabled or disabled
megcoreStatus_t getMIOpenAlgoSearchStatus(bool* algo_search_enabled);
}  // namespace megcore

static inline megcoreStatus_t megcoreCreateComputingHandleWithROCMStream(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags, hipStream_t stream) {
    megcore::ROCMContext ctx;
    ctx.stream = stream;
    return megcore::createComputingHandleWithROCMContext(compHandle, devHandle,
                                                         flags, ctx);
}

static inline megcoreStatus_t megcoreGetROCMStream(
        megcoreComputingHandle_t handle, hipStream_t* stream) {
    megcore::ROCMContext ctx;
    auto ret = megcore::getROCMContext(handle, &ctx);
    *stream = ctx.stream;
    return ret;
}

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen
