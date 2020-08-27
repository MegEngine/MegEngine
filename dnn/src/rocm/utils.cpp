/**
 * \file dnn/src/rocm/utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "src/rocm/utils.h.hip"
#include "src/rocm/utils.h"

#include "src/common/utils.h"
#include "src/rocm/handle.h"
#include "src/rocm/int_fastdiv.h.hip"

#include <mutex>

using namespace megdnn;
using namespace rocm;

namespace {

struct DevicePropRec {
    bool init = false;
    hipDeviceProp_t prop;
    std::mutex mtx;
};
constexpr int MAX_NR_DEVICE = 32;
DevicePropRec device_prop_rec[MAX_NR_DEVICE];

const char* rocblasGetErrorString(rocblas_status error) {
    switch (error) {
        case rocblas_status_success:
            return "rocblas_status_success";
        case rocblas_status_invalid_handle:
            return "rocblas_status_invalid_handle";
        case rocblas_status_not_implemented:
            return "rocblas_status_not_implemented";
        case rocblas_status_invalid_pointer:
            return "rocblas_status_invalid_pointer";
        case rocblas_status_invalid_size:
            return "rocblas_status_invalid_size";
        case rocblas_status_memory_error:
            return "rocblas_status_memory_error";
        case rocblas_status_internal_error:
            return "rocblas_status_internal_error";
    }
    return "Unknown ROCBlas error";
}

}  // anonymous namespace

void rocm::__throw_hip_error__(hipError_t err, const char* msg) {
    auto s = ssprintf("hip error %s(%d) occurred; expr: %s",
                      hipGetErrorString(err), int(err), msg);
    megdnn_throw(s.c_str());
}

void rocm::__throw_miopen_error__(miopenStatus_t err, const char* msg) {
    auto s = ssprintf("miopen error %s(%d) occurred; expr: %s",
                      miopenGetErrorString(err), int(err), msg);
    megdnn_throw(s.c_str());
}

void rocm::__throw_rocblas_error__(rocblas_status err, const char* msg) {
    auto s = ssprintf("rocblas error %s(%d) occurred; expr: %s",
                      rocblasGetErrorString(err), int(err), msg);
    megdnn_throw(s.c_str());
}

void rocm::report_error(const char* msg) {
    megdnn_throw(msg);
    MEGDNN_MARK_USED_VAR(msg);
}

uint32_t rocm::safe_size_in_kern(size_t size) {
    if (!size || size > Uint32Fastdiv::MAX_DIVIDEND) {
        megdnn_throw(
                ssprintf("invalid size for element-wise kernel: %zu; "
                         "max supported size is %u",
                         size, Uint32Fastdiv::MAX_DIVIDEND));
    }
    return size;
}

hipDeviceProp_t rocm::current_device_prop() {
    int dev;
    hip_check(hipGetDevice(&dev));
    megdnn_assert(dev < MAX_NR_DEVICE, "device number too large: %d", dev);
    auto&& rec = device_prop_rec[dev];
    if (!rec.init) {
        std::lock_guard<std::mutex> lock(rec.mtx);
        if (!rec.init) {
            hip_check(hipGetDeviceProperties(&rec.prop, dev));
            rec.init = true;
        }
    }
    return rec.prop;
}

size_t rocm::max_batch_x_channel_size() {
    return current_device_prop().maxGridSize[2];
}

// vim: syntax=cpp.doxygen

