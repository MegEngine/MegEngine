/**
 * \file dnn/src/cambricon/megcore/cambricon_device_context.hpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <mutex>
#include "megcore_cambricon.h"
#include "src/common/megcore/common/device_context.hpp"
#include "src/common/utils.h"

namespace megcore {
namespace cambricon {

class CambriconDeviceContext : public DeviceContext {
public:
    CambriconDeviceContext(int device_id, unsigned int flags,
                           bool global_initialized = false);
    ~CambriconDeviceContext() noexcept;

    size_t mem_alignment_in_bytes() const noexcept override;

    void activate() override;
    void* malloc(size_t size_in_bytes) override;
    void free(void* ptr) override;

    struct InitStatus {
        bool initialized;
        std::mutex mtx;
        InitStatus() : initialized{false} {}
        void init() {
            std::lock_guard<std::mutex> guard{mtx};
            if (!initialized) {
                auto cnrt_err = cnrtInit(0);
                initialized = cnrt_err == CNRT_RET_SUCCESS;
                megdnn_assert(initialized, "cnrt initialize failed: (cnrt:%d)",
                              static_cast<int>(cnrt_err));
            }
        }
        ~InitStatus() {
            if (initialized) {
                cnrtDestroy();
                initialized = false;
            }
        }
    };
    static InitStatus init_status;

private:
    cnrtDeviceInfo_t device_info;
};

}  // namespace cambricon
}  // namespace megcore

// vim: syntax=cpp.doxygen

