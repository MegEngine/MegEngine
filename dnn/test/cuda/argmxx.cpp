/**
 * \file dnn/test/cuda/argmxx.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/rng.h"

namespace {

using namespace megdnn;
using namespace test;

class ArgmxxRNG final: public RNG {
    public:
        void gen(const TensorND &tensor) override {
            auto offset = tensor.layout.span().low_elem;
            auto nr_elems = tensor.layout.span().dist_elem();

#define cb(DType)                                             \
    if (tensor.layout.dtype == DType()) {                     \
        using ctype = typename DTypeTrait<DType>::ctype;      \
        auto ptr = tensor.ptr<ctype>();                       \
        for (size_t i = 0; i < nr_elems; ++i) {               \
            ptr[offset + i] = i;                              \
        }                                                     \
        COMPAT_RANDOM(ptr + offset, ptr + offset + nr_elems); \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
#undef cb
        }
};

template <typename Argmxx>
void test_argmxx(Handle *handle)
{
    Checker<Argmxx> checker(handle);
    checker.set_dtype(1, dtype::Int32());
    using Param = typename Argmxx::Param;
    ArgmxxRNG rng;
    checker.set_rng(0, &rng);
    for (size_t axis = 0; axis < 4; ++axis) {
        Param param;
        param.axis = axis;
        checker.set_param(param).set_dtype(0, dtype::Float32()).
            execs({{2, 3, 4, 5}, {}});
        checker.set_param(param).set_dtype(0, dtype::Float16()).
            execs({{2, 3, 4, 5}, {}});
        checker.set_param(param).set_dtype(0, dtype::Int32()).
            execs({{2, 3, 4, 5}, {}});
        checker.set_param(param).set_dtype(0, dtype::Int16()).
            execs({{2, 3, 4, 5}, {}});
        checker.set_param(param).set_dtype(0, dtype::Int8()).
            execs({{2, 3, 4, 5}, {}});
        checker.set_param(param).set_dtype(0, dtype::Uint8()).
            execs({{2, 3, 4, 5}, {}});
    }
    checker.set_dtype(0, dtype::Float32());
    Param param;
    param.axis = 1;
    checker.set_param(param);
    // 1-step
    checker.execs({{2, 64, 32}, {}});
    // 2-step
    checker.execs({{2, 192, 32}, {}});
    // 3-step
    checker.execs({{2, 4333, 32}, {}});
    // single reduce
    checker.execs({{2, 1, 1}, {}});
    checker.execs({{2, 1+1, 1}, {}});
    checker.execs({{2, 2048+1, 1}, {}});
    checker.execs({{2, 2048*2048+1, 1}, {}});
    checker.execs({{2, 1+1, 31}, {}});
    checker.execs({{2, 16+1, 31}, {}});
    checker.execs({{2, 16*16+1, 31}, {}});
    checker.execs({{2, 16*16*16+1, 31}, {}});
    checker.execs({{2, 16*16*16*16+1, 31}, {}});
    checker.execs({{3, 256*256+1, 2}, {}});
    checker.execs({{3, 128*128+1, 3}, {}});
    checker.execs({{3, 64*64+1, 7}, {}});
    checker.execs({{3, 32*32+1, 15}, {}});
    checker.execs({{3, 512, 500}, {}});
    // very large reduce
    checker.execs({{1, 4194304, 1}, {}});
}

} // anonymous namespace

namespace megdnn {
namespace test {

TEST_F(CUDA, ARGMAX)
{
    test_argmxx<Argmax>(handle_cuda());
}

TEST_F(CUDA, ARGMIN)
{
    test_argmxx<Argmin>(handle_cuda());
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
