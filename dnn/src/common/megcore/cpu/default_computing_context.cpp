/**
 * \file dnn/src/common/megcore/cpu/default_computing_context.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.h"
#include "./default_computing_context.hpp"

#include <cstring>

namespace {
class InplaceDispatcher final : public MegcoreCPUDispatcher {
public:
    void dispatch(Task&& task) override { task(); }

    void dispatch(MultiThreadingTask&& task, size_t parallelism) override {
        for (size_t i = 0; i < parallelism; i++) {
            task(i, 0);
        }
    }

    void sync() override {}

    size_t nr_threads() override { return 1; };
};
}  // namespace

using namespace megcore;
using namespace cpu;

DefaultComputingContext::DefaultComputingContext(
        megcoreDeviceHandle_t dev_handle, unsigned int flags):
    ComputingContext(dev_handle, flags),
    m_dispatcher{megdnn::make_unique<InplaceDispatcher>()}
{
    megcorePlatform_t platform;
    megcoreGetPlatform(dev_handle, &platform);
    megdnn_assert(platform & megcorePlatformCPU);
}

DefaultComputingContext::~DefaultComputingContext() noexcept = default;

void DefaultComputingContext::memcpy(void *dst, const void *src,
        size_t size_in_bytes,
        megcoreMemcpyKind_t /* kind */)
{
    ::memcpy(dst, src, size_in_bytes);
}

void DefaultComputingContext::memset(void *dst, int value, size_t size_in_bytes)
{
    ::memset(dst, value, size_in_bytes);
}

void DefaultComputingContext::synchronize()
{
    m_dispatcher->sync();
}

// vim: syntax=cpp.doxygen
