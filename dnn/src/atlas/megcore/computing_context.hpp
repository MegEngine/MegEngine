/**
 * \file dnn/src/atlas/megcore/computing_context.hpp
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

#include "megcore_atlas.h"
#include "src/common/megcore/common/computing_context.hpp"

#include <acl/acl_rt.h>

namespace megcore {
namespace atlas {

class AtlasComputingContext final : public ComputingContext {
public:
    AtlasComputingContext(megcoreDeviceHandle_t dev_handle, unsigned int flags,
                          const AtlasContext& ctx = {});
    ~AtlasComputingContext();

    void memcpy(void* dst, const void* src, size_t size_in_bytes,
                megcoreMemcpyKind_t kind) override;
    void memset(void* dst, int value, size_t size_in_bytes) override;
    void synchronize() override;

    const AtlasContext& context() const { return m_ctx; }

    aclrtStream stream() const { return context().stream; }

private:
    bool m_own_stream;
    AtlasContext m_ctx;
};

}  // namespace atlas
}  // namespace megcore

// vim: syntax=cpp.doxygen
