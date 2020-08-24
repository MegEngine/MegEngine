/**
 * \file dnn/src/atlas/utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/atlas/utils.h"
#include "megcore_atlas.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace atlas;

void atlas::__throw_acl_error__(aclError err, const char* msg) {
    auto s = ssprintf("acl return %s(%d) occurred; expr: %s",
                      megcore::atlas::get_error_str(err), int(err), msg);
    megdnn_throw(s.c_str());
}

// vim: syntax=cpp.doxygen
