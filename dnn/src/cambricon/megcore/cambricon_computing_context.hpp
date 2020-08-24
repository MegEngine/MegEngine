/**
 * \file dnn/src/cambricon/megcore/cambricon_computing_context.hpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megcore_cambricon.h"
#include "src/common/megcore/common/computing_context.hpp"

namespace megcore {
namespace cambricon {

class CambriconComputingContext final : public ComputingContext {
public:
    CambriconComputingContext(megcoreDeviceHandle_t dev_handle,
                              unsigned int flags,
                              const CambriconContext& ctx = {});
    ~CambriconComputingContext();

    void memcpy(void* dst, const void* src, size_t size_in_bytes,
                megcoreMemcpyKind_t kind) override;
    void memset(void* dst, int value, size_t size_in_bytes) override;
    void synchronize() override;

    const CambriconContext& context() const { return context_; }

    cnrtQueue_t queue() const { return context().queue; }

private:
    bool own_queue;
    CambriconContext context_;
};

}  // namespace cambricon
}  // namespace megcore

// vim: syntax=cpp.doxygen

