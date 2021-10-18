/**
 * \file dnn/test/aarch64/fixture.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/aarch64/fixture.h"

#include "test/common/memory_manager.h"
#include "test/common/random_state.h"
#include "test/common/utils.h"

namespace megdnn {
namespace test {

Handle* AARCH64::fallback_handle() {
    if (!m_fallback_handle) {
        m_fallback_handle = create_cpu_handle(1);
    }
    return m_fallback_handle.get();
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
