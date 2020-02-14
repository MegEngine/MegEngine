/**
 * \file dnn/src/common/version.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/version.h"
#include "src/common/version_symbol.h"

using namespace megdnn;

Version megdnn::get_version() {
    return {MEGDNN_MAJOR, MEGDNN_MINOR, MEGDNN_PATCH};
}

MEGDNN_VERSION_SYMBOL3(MEGDNN, MEGDNN_MAJOR, MEGDNN_MINOR, MEGDNN_PATCH);

// vim: syntax=cpp.doxygen
