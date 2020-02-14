/**
 * \file dnn/src/common/megcore/public_api/misc.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megcore.h"
#include "src/common/utils.h"

const char *megcoreGetErrorName(megcoreStatus_t status)
{
#define CASE(x) case x: return megdnn_mangle(#x)
    switch (status) {
        CASE(megcoreSuccess);
        CASE(megcoreErrorMemoryAllocation);
        CASE(megcoreErrorInvalidArgument);
        CASE(megcoreErrorInvalidDeviceHandle);
        CASE(megcoreErrorInternalError);
        CASE(megcoreErrorInvalidComputingHandle);
        default:
            return megdnn_mangle("<Unknown MegCore Error>");
    }
#undef CASE
}

// vim: syntax=cpp.doxygen
