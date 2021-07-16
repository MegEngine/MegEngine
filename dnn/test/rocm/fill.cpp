/**
 * \file dnn/test/rocm/fill.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/fill.h"

#include "test/rocm/fixture.h"

namespace megdnn {
namespace test {
namespace fill {

TEST_F(ROCM, FILL_F32) {
    run_fill_test(handle_rocm(), dtype::Float32{});
}

TEST_F(ROCM, FILL_I32) {
    run_fill_test(handle_rocm(), dtype::Int32{});
}

#if !MEGDNN_DISABLE_FLOAT16
TEST_F(ROCM, FILL_F16) {
    run_fill_test(handle_rocm(), dtype::Float16{});
}
#endif

} // namespace fill
} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
