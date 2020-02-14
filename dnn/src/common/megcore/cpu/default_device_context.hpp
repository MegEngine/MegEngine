/**
 * \file dnn/src/common/megcore/cpu/default_device_context.hpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "../common/device_context.hpp"

namespace megcore {
namespace cpu {

/**
 * \brief A thin wrapper class over malloc and free.
 *
 * No magic thing happens here.
 */
class DefaultDeviceContext: public DeviceContext {
    public:
        DefaultDeviceContext(int device_id, unsigned int flags);
        ~DefaultDeviceContext() noexcept;

        size_t mem_alignment_in_bytes() const noexcept override;

        void activate() noexcept override;
        void *malloc(size_t size_in_bytes) override;
        void free(void *ptr) override;
};

} // namespace cpu
} // namespace megcore

// vim: syntax=cpp.doxygen
