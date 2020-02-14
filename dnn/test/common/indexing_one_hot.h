/**
 * \file dnn/test/common/indexing_one_hot.h
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

namespace megdnn {
namespace test {

void run_indexing_one_hot_test(Handle* handle,
                               const thin_function<void()>& fail_test = {});
void run_indexing_set_one_hot_test(Handle* handle);

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
