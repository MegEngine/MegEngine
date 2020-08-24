/**
 * \file dnn/test/cambricon/fixture.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cambricon/fixture.h"
#include "src/cambricon/handle.h"
#include "src/cambricon/utils.h"
#include "test/common/memory_manager.h"
#include "test/common/random_state.h"
#include "test/common/utils.h"

#include <cnrt.h>
#include <cstdlib>

using namespace megdnn;
using namespace test;

void CAMBRICON::SetUp() {
    RandomState::reset();

    megcoreDeviceHandle_t dev_handle;
    // use card 0
    megcore_check(megcoreCreateDeviceHandle(&dev_handle,
                                            megcorePlatformCambricon, 0));

    megcoreComputingHandle_t comp_handle;
    megcore_check(megcoreCreateComputingHandle(&comp_handle, dev_handle));
    m_handle_cambricon = Handle::make(comp_handle);
    megdnn_assert(m_handle_cambricon);
}

Handle* CAMBRICON::handle_naive() {
    if (!m_handle_naive)
        m_handle_naive = create_cpu_handle(2);
    return m_handle_naive.get();
}

void CAMBRICON::TearDown() {
    m_handle_naive.reset();
    m_handle_cambricon.reset();
    MemoryManagerHolder::instance()->clear();
}

// vim: syntax=cpp.doxygen

