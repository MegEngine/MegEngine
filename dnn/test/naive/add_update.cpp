/**
 * \file test/naive/add_update.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "test/naive/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/extra_impl_helper.h"

namespace megdnn {
namespace test {

TEST_F(NAIVE, ADD_UPDATE_BFLOAT16) {
    Checker<AddUpdate> checker(handle(), false);
    param::AddUpdate p{2, -1, 3};
    auto extra_impl = extra_impl_helper<AddUpdate>(handle(), p);
    checker.set_param(p)
            .set_dtype(0, dtype::BFloat16())
            .set_dtype(1, dtype::BFloat16())
            .set_extra_opr_impl(extra_impl)
            .execs({{2, 2, 3}, {2, 2, 3}});
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
