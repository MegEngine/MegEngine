/**
 * \file dnn/src/cuda/utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/utils.cuh"
#include "src/cuda/utils.h"

#include "src/common/utils.h"
#include "src/cuda/handle.h"
#include "src/cuda/int_fastdiv.cuh"

#include <mutex>

using namespace megdnn;
using namespace cuda;

namespace {

struct DevicePropRec {
    bool init = false;
    cudaDeviceProp prop;
    std::mutex mtx;
};
constexpr int MAX_NR_DEVICE = 32;
DevicePropRec device_prop_rec[MAX_NR_DEVICE];

const char *cublasGetErrorString(cublasStatus_t error) {
	switch (error)
	{
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
		case CUBLAS_STATUS_LICENSE_ERROR:
			return "CUBLAS_STATUS_LICENSE_ERROR";
		case CUBLAS_STATUS_NOT_SUPPORTED:
			return "CUBLAS_STATUS_NOT_SUPPORTED";
	}
	return "Unknown CUBLAS error";
}

} // anonymous namespace

void cuda::__throw_cuda_error__(cudaError_t err, const char *msg) {
    auto s = ssprintf("cuda error %s(%d) occurred; expr: %s",
            cudaGetErrorString(err), int(err), msg);
    megdnn_throw(s.c_str());
}

void cuda::__throw_cudnn_error__(cudnnStatus_t err, const char *msg) {
    auto s = ssprintf("cudnn error %s(%d) occurred; expr: %s",
            cudnnGetErrorString(err), int(err), msg);
    megdnn_throw(s.c_str());
}

void cuda::__throw_cublas_error__(cublasStatus_t err, const char *msg) {
    auto s = ssprintf("cublas error %s(%d) occurred; expr: %s",
            cublasGetErrorString(err), int(err), msg);
    megdnn_throw(s.c_str());
}

void cuda::__throw_cusolver_error__(cusolverStatus_t err, const char* msg) {
    auto s = ssprintf("cusolver error %d occurred; expr: %s", int(err), msg);
    megdnn_throw(s.c_str());
}

void cuda::report_error(const char *msg) {
    megdnn_throw(msg);
    MEGDNN_MARK_USED_VAR(msg);
}

uint32_t cuda::safe_size_in_kern(size_t size) {
    if (!size || size > Uint32Fastdiv::MAX_DIVIDEND) {
        megdnn_throw(ssprintf(
                    "invalid size for element-wise kernel: %zu; "
                    "max supported size is %u",
                    size, Uint32Fastdiv::MAX_DIVIDEND));
    }
    return size;
}

cudaDeviceProp cuda::current_device_prop() {
    int dev;
    cuda_check(cudaGetDevice(&dev));
    megdnn_assert(dev < MAX_NR_DEVICE, "device number too large: %d", dev);
    auto &&rec = device_prop_rec[dev];
    if (!rec.init) {
        std::lock_guard<std::mutex> lock(rec.mtx);
        if (!rec.init) {
            cuda_check(cudaGetDeviceProperties(&rec.prop, dev));
            rec.init = true;
        }
    }
    return rec.prop;
}

bool cuda::is_compute_capability_required(int major, int minor) {
    auto&& device_prop = cuda::current_device_prop();
    return device_prop.major > major ||
           (device_prop.major == major && device_prop.minor >= minor);
}

size_t cuda::max_batch_x_channel_size() {
    return current_device_prop().maxGridSize[2];
}

// vim: syntax=cpp.doxygen

