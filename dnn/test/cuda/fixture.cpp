/**
 * \file dnn/test/cuda/fixture.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"
#include "test/common/utils.h"
#include "test/common/memory_manager.h"
#include "src/cuda/utils.h"
#include "src/cuda/handle.h"
#include "test/common/random_state.h"

#include <cstdlib>
#include <cuda_runtime_api.h>

using namespace megdnn;
using namespace test;

namespace {
    void setup_device() {
#if !defined(_WIN32)
        auto device_id_env = std::getenv("MEGDNN_DEVICE_ID");
        int device_id = -1;
        if (device_id_env) {
            device_id = std::atoi(device_id_env);
            std::cout << "Select device " << device_id
                << " because MEGDNN_DEVICE_ID is set." << std::endl;
        }
        auto pci_bus_id_env = std::getenv("MEGDNN_PCI_BUS_ID");
        if (pci_bus_id_env) {
            megdnn_assert(cudaSuccess == cudaDeviceGetByPCIBusId(&device_id,
                        pci_bus_id_env));
            std::cout << "Select device " << pci_bus_id_env << " ("
                << device_id << ") because MEGDNN_PCI_BUS_ID is set."
                << std::endl;
        }
        if (device_id_env && pci_bus_id_env) {
            std::cout << "MEGDNN_DEVICE_ID and MEGDNN_PCI_BUS_ID should not "
                "be set simultaneously." << std::endl;
            exit(1);
        }
        if (device_id_env || pci_bus_id_env) {
            megdnn_assert(cudaSuccess == cudaSetDevice(device_id));
        }
#endif
    }
} // anonymous namespace

void CUDA::SetUp() {
    RandomState::reset();

    setup_device();
    megcoreDeviceHandle_t dev_handle;
    megcore_check(megcoreCreateDeviceHandle(&dev_handle,
                megcorePlatformCUDA));


    megcoreComputingHandle_t comp_handle;
    megcore_check(megcoreCreateComputingHandle(&comp_handle,
                dev_handle));
    m_handle_cuda = Handle::make(comp_handle);
    megdnn_assert(m_handle_cuda);
}

Handle* CUDA::handle_naive() {
    if (!m_handle_naive)
        m_handle_naive = create_cpu_handle(2);
    return m_handle_naive.get();
}

void CUDA::TearDown() {
    m_handle_naive.reset();
    m_handle_cuda.reset();
    MemoryManagerHolder::instance()->clear();
}

void CUDA_ERROR_INFO::SetUp() {
    setup_device();
    megcoreDeviceHandle_t dev_handle;
    megcore_check(megcoreCreateDeviceHandle(&dev_handle, megcorePlatformCUDA));

    m_error_info_dev = nullptr;
    void* ptr;
    cuda_check(cudaMalloc(&ptr, sizeof(megcore::AsyncErrorInfo)));
    cuda_check(cudaMemset(ptr, 0, sizeof(megcore::AsyncErrorInfo)));
    cuda_check(cudaDeviceSynchronize());
    m_error_info_dev = static_cast<megcore::AsyncErrorInfo*>(ptr);

    // create handle bind with error_info
    megcoreComputingHandle_t comp_handle;
    megcore_check(megcore::createComputingHandleWithCUDAContext(
            &comp_handle, dev_handle, 0, {nullptr, m_error_info_dev}));
    m_handle_cuda = Handle::make(comp_handle);
    megdnn_assert(static_cast<bool>(m_handle_cuda));
}

void CUDA_ERROR_INFO::TearDown() {
    if (m_error_info_dev) {
        cuda_check(cudaFree(m_error_info_dev));
    }
    m_handle_cuda.reset();
    MemoryManagerHolder::instance()->clear();
}

megcore::AsyncErrorInfo CUDA_ERROR_INFO::get_error_info() {
    megcore::AsyncErrorInfo ret;
    auto stream = cuda::cuda_stream(m_handle_cuda.get());
    cuda_check(cudaMemcpyAsync(&ret, m_error_info_dev, sizeof(ret),
                cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    return ret;
}

// vim: syntax=cpp.doxygen
