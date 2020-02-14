/**
 * \file dnn/test/common/random_state.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/random_state.h"

namespace megdnn {
namespace test {

const int RandomState::m_seed;
RandomState RandomState::m_instance = RandomState();

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen
