/**
 * \file dnn/test/x86/lrn.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/x86/fixture.h"

#include "test/common/checker.h"
#include "test/common/local.h"

namespace megdnn {
namespace test {

TEST_F(X86, LRN)
{
    Checker<LRNForward> checker(handle());
    checker.execs({{2, 11, 12, 13}, {}});
    for (size_t w = 10; w <= 50; ++w) {
        checker.execs({{2, w, 12, 13}, {}});
    }
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen

