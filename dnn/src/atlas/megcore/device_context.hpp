/**
 * \file dnn/src/atlas/megcore/device_context.hpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "src/common/megcore/common/device_context.hpp"
#include "src/common/utils.h"
#include "megcore_atlas.h"

#include <mutex>
#include "acl/acl.h"

namespace megcore {
namespace atlas {

class AtlasDeviceContext : public DeviceContext {
public:
    AtlasDeviceContext(int device_id, unsigned int flags,
                       bool global_initialized = false);
    ~AtlasDeviceContext() noexcept;

    size_t mem_alignment_in_bytes() const noexcept override;

    void activate() override;
    void deactivate() override;
    void* malloc(size_t size_in_bytes) override;
    void free(void* ptr) override;

    struct InitStatus {
        bool initialized;
        std::mutex mtx;
        InitStatus() : initialized{false} {}
        void init() {
            std::lock_guard<std::mutex> guard{mtx};
            if (!initialized) {
                auto err = aclInit(nullptr);
                initialized = err == ACL_ERROR_NONE;
                megdnn_assert(initialized,
                              "aclrt initialize failed: (acl:%d): %s",
                              static_cast<int>(err),
                              megcore::atlas::get_error_str(err));
            }
        }
        ~InitStatus() {
            if (initialized) {
                initialized = false;
            }
        }
    };
    static InitStatus init_status;
};

}  // namespace atlas
}  // namespace megcore

// vim: syntax=cpp.doxygen
