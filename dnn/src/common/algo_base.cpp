/**
 * \file dnn/src/common/algo_base.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/common/algo_base.h"
#include "src/common/utils.h"

using namespace megdnn;

bool Algorithm::contain_attribute(const Attribute& attr) const {
    return bool(attribute() & attr);
}

// vim: syntax=cpp.doxygen
