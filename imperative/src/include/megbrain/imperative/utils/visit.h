/**
 * \file imperative/src/include/megbrain/imperative/utils/visit.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <vector>

#include "megbrain/utils/small_vector.h"

namespace mgb::imperative {

template <typename... TVisitors>
class Visitor : public TVisitors... {
public:
    using TVisitors::operator()...;
};

}  // namespace mgb::imperative
