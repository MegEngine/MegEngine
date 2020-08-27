/**
 * \file dnn/test/rocm/fixture.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "test/rocm/fixture.h"
#include "src/rocm/handle.h"
#include "test/common/memory_manager.h"
#include "test/common/random_state.h"
#include "test/common/utils.h"
#include "src/rocm/utils.h"

#include <cstdlib>
#include "hip_header.h"

using namespace megdnn;
using namespace test;

namespace {
void setup_device() {
#if !defined(WIN32)
    auto device_id_env = std::getenv("MEGDNN_DEVICE_ID");
    int device_id = -1;
    if (device_id_env) {
        device_id = std::atoi(device_id_env);
        std::cout << "Select device " << device_id
                  << " because MEGDNN_DEVICE_ID is set." << std::endl;
    }
    auto pci_bus_id_env = std::getenv("MEGDNN_PCI_BUS_ID");
    if (pci_bus_id_env) {
        megdnn_assert(hipSuccess ==
                      hipDeviceGetByPCIBusId(&device_id, pci_bus_id_env));
        std::cout << "Select device " << pci_bus_id_env << " (" << device_id
                  << ") because MEGDNN_PCI_BUS_ID is set." << std::endl;
    }
    if (device_id_env && pci_bus_id_env) {
        std::cout << "MEGDNN_DEVICE_ID and MEGDNN_PCI_BUS_ID should not "
                     "be set simultaneously."
                  << std::endl;
        exit(1);
    }
    if (device_id_env || pci_bus_id_env) {
        megdnn_assert(hipSuccess == hipSetDevice(device_id));
    }
#endif
}
}  // anonymous namespace

void ROCM::SetUp() {
    RandomState::reset();

    setup_device();
    megcoreDeviceHandle_t dev_handle;
    megcore_check(megcoreCreateDeviceHandle(&dev_handle, megcorePlatformROCM));

    megcoreComputingHandle_t comp_handle;
    megcore_check(megcoreCreateComputingHandle(&comp_handle, dev_handle));
    m_handle_rocm = Handle::make(comp_handle);
    megdnn_assert(m_handle_rocm);
}

Handle* ROCM::handle_naive(bool check_dispatch) {
    if (!m_handle_naive)
        m_handle_naive = create_cpu_handle(2, check_dispatch);
    return m_handle_naive.get();
}

void ROCM::TearDown() {
    m_handle_naive.reset();
    m_handle_rocm.reset();
    MemoryManagerHolder::instance()->clear();
}

void ROCM_ERROR_INFO::SetUp() {
    setup_device();
    megcoreDeviceHandle_t dev_handle;
    megcore_check(megcoreCreateDeviceHandle(&dev_handle, megcorePlatformROCM));

    m_error_info_dev = nullptr;
    void* ptr;
    hip_check(hipMalloc(&ptr, sizeof(megcore::AsyncErrorInfo)));
    hip_check(hipMemset(ptr, 0, sizeof(megcore::AsyncErrorInfo)));
    hip_check(hipDeviceSynchronize());
    m_error_info_dev = static_cast<megcore::AsyncErrorInfo*>(ptr);

    // create handle bind with error_info
    megcoreComputingHandle_t comp_handle;
    megcore_check(megcore::createComputingHandleWithROCMContext(
            &comp_handle, dev_handle, 0, {nullptr, m_error_info_dev}));
    m_handle_rocm = Handle::make(comp_handle);
    megdnn_assert(static_cast<bool>(m_handle_rocm));
}

void ROCM_ERROR_INFO::TearDown() {
    if (m_error_info_dev) {
        hip_check(hipFree(m_error_info_dev));
    }
    m_handle_rocm.reset();
    MemoryManagerHolder::instance()->clear();
}

megcore::AsyncErrorInfo ROCM_ERROR_INFO::get_error_info() {
    megcore::AsyncErrorInfo ret;
    auto stream = rocm::hip_stream(m_handle_rocm.get());
    hip_check(hipMemcpyAsync(&ret, m_error_info_dev, sizeof(ret),
                             hipMemcpyDeviceToHost, stream));
    hip_check(hipStreamSynchronize(stream));
    return ret;
}

// vim: syntax=cpp.doxygen
