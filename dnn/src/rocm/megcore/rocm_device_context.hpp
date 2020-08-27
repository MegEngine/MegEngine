/**
 * \file dnn/src/rocm/megcore/rocm_device_context.hpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/common/megcore/common/device_context.hpp"

namespace megcore {
namespace rocm {

class ROCMDeviceContext: public DeviceContext {
    public:
        ROCMDeviceContext(int device_id, unsigned int flags);
        ~ROCMDeviceContext() noexcept;

        size_t mem_alignment_in_bytes() const noexcept override;

        void activate() override;
        void *malloc(size_t size_in_bytes) override;
        void free(void *ptr) override;
    private:
        hipDeviceProp_t prop_;
};

} // namespace rocm
} // namespace megcore

// vim: syntax=cpp.doxygen
