/**
 * \file dnn/test/common/powc.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/handle.h"
#include "megdnn/oprs/general.h"
#include "test/common/opr_proxy.h"

namespace megdnn {
namespace test {

DEF(PowC, 2, false, true);

void run_powc_test(Handle* handle, DType dtype);

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
