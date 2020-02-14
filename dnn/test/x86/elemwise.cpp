/**
 * \file dnn/test/x86/elemwise.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/elemwise.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/x86/fixture.h"

using namespace megdnn;
using namespace test;

void print4D(const TensorND& tensor) {
    TensorLayout layout = tensor.layout;
    float* result = tensor.ptr<float>();
    size_t N = layout.shape[0], C = layout.shape[1], H = layout.shape[2],
           W = layout.shape[3];
    size_t it = 0;
    rep(n, N) {
        rep(c, C) {
            rep(h, H) {
                rep(w, W) { printf("%.4f ", result[it++]); }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

#define UNARY_TEST_CASE(_optr)                                \
    checker.set_param(Mode::_optr).execs({{1, 1556011}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 7}, {}});

#define BUILD_UNARY_TEST_CASE_INT \
    UNARY_TEST_CASE(RELU)         \
    UNARY_TEST_CASE(ABS)

#define BUILD_UNARY_TEST_CASE_FLOAT \
    UNARY_TEST_CASE(ABS)            \
    UNARY_TEST_CASE(LOG)            \
    UNARY_TEST_CASE(COS)            \
    UNARY_TEST_CASE(SIN)            \
    UNARY_TEST_CASE(FLOOR)          \
    UNARY_TEST_CASE(CEIL)           \
    UNARY_TEST_CASE(SIGMOID)        \
    UNARY_TEST_CASE(EXP)            \
    UNARY_TEST_CASE(TANH)           \
    UNARY_TEST_CASE(RELU)           \
    UNARY_TEST_CASE(ROUND)

TEST_F(X86, ELEMWISE_FORWARD_UNARY) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle());
    // case int
    checker.set_dtype(0, dtype::Int8());
    BUILD_UNARY_TEST_CASE_INT

    checker.set_dtype(0, dtype::Int16());
    BUILD_UNARY_TEST_CASE_INT

    checker.set_dtype(0, dtype::Int32());
    BUILD_UNARY_TEST_CASE_INT

    // case float
    UniformFloatRNG rng(1e-2, 6e1);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-6);
    checker.set_dtype(0, dtype::Float32());
    BUILD_UNARY_TEST_CASE_FLOAT
}

#define BINARY_TEST_CASE(_optr)                                             \
    checker.set_param(Mode::_optr).execs({{3, 4, 17}, {3, 4, 17}, {}});     \
    checker.set_param(Mode::_optr).execs({{3, 4, 5, 7}, {1, 1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 2, 2}, {1, 1, 1}, {}});       \
    checker.set_param(Mode::_optr).execs({{1, 7}, {1, 7}, {}});

#define BUILD_BINARY_TEST_CASE \
    BINARY_TEST_CASE(MIN)      \
    BINARY_TEST_CASE(MAX)

#define BINARY_COMPLATE_TEST_CASE(_optr)                                    \
    printf("Check binary optr %s by all cases.\n", #_optr);                 \
    checker.set_param(Mode::_optr).execs({{3, 4, 7}, {3, 4, 7}, {}});       \
    checker.set_param(Mode::_optr).execs({{3, 4, 5, 7}, {1, 4, 1, 1}, {}}); \
    checker.set_param(Mode::_optr)                                          \
            .execs({{3, 4, 5, 7, 8}, {3, 4, 5, 7, 8}, {}});                 \
    checker.set_param(Mode::_optr)                                          \
            .execs({{3, 4, 5, 7, 8}, {1, 4, 1, 1, 8}, {}});                 \
    checker.set_param(Mode::_optr).execs({{3, 4, 7}, {1, 4, 1}, {}});       \
    checker.set_param(Mode::_optr).execs({{3, 4, 5, 7}, {1, 1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 7}, {1, 7}, {}});             \
    checker.set_param(Mode::_optr).execs({{1, 2, 2}, {1, 2, 1}, {}});       \
    checker.set_param(Mode::_optr).execs({{1, 2, 2}, {1, 1, 1}, {}});       \
    checker.set_param(Mode::_optr).execs({{3, 4, 1}, {3, 4, 1}, {}});

#define BUILD_BINARY_COMPLATE_TEST_CASE \
    BINARY_COMPLATE_TEST_CASE(ADD)      \
    BINARY_COMPLATE_TEST_CASE(MUL)      \
    BINARY_COMPLATE_TEST_CASE(MAX)      \
    BINARY_COMPLATE_TEST_CASE(MIN)      \
    BINARY_COMPLATE_TEST_CASE(SUB)

#define BUILD_BINARY_COMPLATE_TEST_CASE_FLOAT32 \
    BINARY_COMPLATE_TEST_CASE(TRUE_DIV)         \
    BINARY_COMPLATE_TEST_CASE(FUSE_ADD_SIGMOID) \
    BINARY_COMPLATE_TEST_CASE(FUSE_ADD_TANH)    \
    BINARY_COMPLATE_TEST_CASE(FUSE_ADD_RELU)

TEST_F(X86, ELEMWISE_FORWARD_NCHW88) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle());

    // case float
    UniformFloatRNG rng(1e-5, 7e1);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-5);
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());

    checker.set_param(Mode::ADD).execs({{1, 3, 2, 2, 8}, {1, 3, 1, 1, 8}, {}});
    checker.set_param(Mode::ADD).execs({{2, 3, 2, 2, 8}, {1, 3, 1, 1, 8}, {}});
    checker.set_param(Mode::ADD).execs({{3, 8, 5, 3, 8}, {1, 8, 1, 1, 8}, {}});
    checker.set_param(Mode::ADD).execs({{3, 4, 5, 7, 8}, {3, 4, 5, 7, 8}, {}});
    checker.set_param(Mode::ADD).execs({{1, 2, 5, 7, 8}, {1, 2, 1, 1, 8}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 3, 2, 2, 8}, {1, 3, 1, 1, 8}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{2, 3, 2, 2, 8}, {1, 3, 1, 1, 8}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{3, 8, 5, 3, 8}, {1, 8, 1, 1, 8}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{3, 4, 5, 7, 8}, {3, 4, 5, 7, 8}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 2, 5, 7, 8}, {1, 2, 1, 1, 8}, {}});
}
TEST_F(X86, ELEMWISE_FORWARD_BINARY) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle());

    // case float
    UniformFloatRNG rng(1e-5, 7e1);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-5);
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());
    BUILD_BINARY_COMPLATE_TEST_CASE
    BUILD_BINARY_COMPLATE_TEST_CASE_FLOAT32

    // case int
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    BUILD_BINARY_TEST_CASE
    BUILD_BINARY_COMPLATE_TEST_CASE

    checker.set_dtype(0, dtype::Int16());
    checker.set_dtype(1, dtype::Int16());
    BUILD_BINARY_TEST_CASE
    BUILD_BINARY_COMPLATE_TEST_CASE

    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());
    BUILD_BINARY_TEST_CASE
    BUILD_BINARY_COMPLATE_TEST_CASE
}

#define TERNARY_COMPLATE_TEST_CASE(_optr)                               \
    printf("Check ternary optr %s by all cases.\n", #_optr);            \
    checker.set_param(Mode::_optr)                                      \
            .execs({{3, 4, 7}, {3, 4, 7}, {3, 4, 7}, {}});              \
    checker.set_param(Mode::_optr)                                      \
            .execs({{1, 4, 1, 1}, {3, 4, 5, 7}, {1, 4, 1, 1}, {}});     \
    checker.set_param(Mode::_optr)                                      \
            .execs({{1, 4, 1}, {3, 4, 7}, {1, 4, 1}, {}});              \
    checker.set_param(Mode::_optr)                                      \
            .execs({{3, 4, 5, 7}, {3, 4, 5, 7}, {1, 1, 1, 1}, {}});     \
    checker.set_param(Mode::_optr).execs({{1, 7}, {1, 7}, {1, 7}, {}}); \
    checker.set_param(Mode::_optr)                                      \
            .execs({{1, 2, 1}, {1, 2, 2}, {1, 2, 1}, {}});              \
    checker.set_param(Mode::_optr)                                      \
            .execs({{1, 2, 2}, {1, 2, 2}, {1, 1, 1}, {}});              \
    checker.set_param(Mode::_optr).execs({{3, 4, 1}, {3, 4, 1}, {3, 4, 1}, {}});

#define BUILD_TERNARY_COMPLATE_TEST_CASE \
    TERNARY_COMPLATE_TEST_CASE(FUSE_MUL_ADD3)

TEST_F(X86, ELEMWISE_FORWARD_TERNARY) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle());
    // case int
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int8());
    // BUILD_TERNARY_TEST_CASE
    BUILD_TERNARY_COMPLATE_TEST_CASE

    checker.set_dtype(0, dtype::Int16());
    checker.set_dtype(1, dtype::Int16());
    checker.set_dtype(2, dtype::Int16());
    // BUILD_TERNARY_TEST_CASE
    BUILD_TERNARY_COMPLATE_TEST_CASE

    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());
    checker.set_dtype(2, dtype::Int32());
    // BUILD_TERNARY_TEST_CASE
    BUILD_TERNARY_COMPLATE_TEST_CASE

    // case float
    UniformFloatRNG rng(1e-5, 7e1);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-5);
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());
    checker.set_dtype(2, dtype::Float32());

    // BUILD_TERNARY_TEST_CASE
    BUILD_TERNARY_COMPLATE_TEST_CASE
}

template <typename tag>
class X86_ELEMWISE : public X86 {};
TYPED_TEST_CASE(X86_ELEMWISE, elemwise::test_types);
TYPED_TEST(X86_ELEMWISE, run) {
    elemwise::run_test<TypeParam>(this->handle());
}

// vim: syntax=cpp.doxygen
