/**
 * \file dnn/test/fallback/fixture.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/fallback/fixture.h"

#include "test/common/utils.h"
#include "test/common/memory_manager.h"
#include "test/common/random_state.h"

namespace megdnn {
namespace test {

void FALLBACK::SetUp()
{
    RandomState::reset();
    m_handle = create_cpu_handle(1);
}

void FALLBACK::TearDown()
{
    m_handle.reset();
    MemoryManagerHolder::instance()->clear();
}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen

