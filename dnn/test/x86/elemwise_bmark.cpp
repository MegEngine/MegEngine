/**
 * \file dnn/test/x86/elemwise_bmark.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/x86/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/benchmarker.h"

using namespace megdnn;
using namespace test;

#define TEST_IN_DIFF_DISTRUBUTION(proportion_of_inf, dataset_number) \
    max_val = 88.3762626647949f / (1 - proportion_of_inf); \
    UniformFloatRNG rng##dataset_number(0.f, max_val); \
    B.set_rng(0, &rng##dataset_number); \
    B.execs({{355600}, {}});


TEST_F(X86, BENCHMARK_ELEM_EXP_BASED_OPTRS)
{
    Benchmarker<ElemwiseForward> B(handle());
    using Mode = ElemwiseForward::Param::Mode;
    //UniformFloatWithZeroRNG rng(80, 100, 0.1);
    printf("Test Optr exp(x)\n");
    B.set_param(Mode::EXP);
    B.execs({{355600}, {}});
    float max_val = 0;
    TEST_IN_DIFF_DISTRUBUTION(0.25, 1)
    TEST_IN_DIFF_DISTRUBUTION(0.5, 2)
    TEST_IN_DIFF_DISTRUBUTION(0.75, 3)
    TEST_IN_DIFF_DISTRUBUTION(0.9999, 4)

    printf("Test Optr tanh(x)\n");
    B.set_param(Mode::TANH);
    B.execs({{355600}, {}});
    max_val = 0;
    TEST_IN_DIFF_DISTRUBUTION(0.25, 5)
    TEST_IN_DIFF_DISTRUBUTION(0.5, 6)
    TEST_IN_DIFF_DISTRUBUTION(0.75, 7)
    TEST_IN_DIFF_DISTRUBUTION(0.9999, 8)

    printf("Test Optr fast_tanh(x)\n");
    B.set_param(Mode::FAST_TANH);
    B.execs({{355600}, {}});

    printf("Test Optr sigmoid(x)\n");
    B.set_param(Mode::SIGMOID);
    B.execs({{355600}, {}});
    max_val = 0;
    TEST_IN_DIFF_DISTRUBUTION(0.25, 13)
    TEST_IN_DIFF_DISTRUBUTION(0.5, 14)
    TEST_IN_DIFF_DISTRUBUTION(0.75, 15)
    TEST_IN_DIFF_DISTRUBUTION(0.9999, 16)

    printf("Test Optr tanh_grad(x)\n");
    B.set_param(Mode::TANH_GRAD);
    B.execs({{355600}, {355600}, {}});

    printf("Test Optr fast_tanh_grad(x)\n");
    B.set_param(Mode::FAST_TANH_GRAD);
    B.execs({{355600}, {355600}, {}});
}

// 1. Unary
#define BENCHMARK_UNARY(Optr, size) \
    printf("Test for %s \n", #Optr); \
    B.set_param(Mode::Optr); \
    B.execs({{4, 4, 4, 1+size/64, }, {}});

// 2. Binary
#define BENCHMARK_BINARY(Optr, size) \
    B.set_param(Mode::Optr); \
    B.execs({{size}, {size}, {}});

#define BENCHMARK_BINARY_SCALAR(Optr, size) \
    B.set_param(Mode::Optr); \
    B.execs({{size}, {1}, {}});

#define BENCHMARK_BINARY_1C11(Optr, chan) \
    B.set_param(Mode::Optr); \
    B.execs({{9, chan, 33, 127}, {1, chan, 1, 1}, {}});

#define BENCHMARK_BINARY_ALL_KINDS(Optr, size) \
    printf("Test for %s \n", #Optr); \
    BENCHMARK_BINARY(Optr, size) \
    BENCHMARK_BINARY_SCALAR(Optr, size) \
    BENCHMARK_BINARY_1C11(Optr, (1+size/37719))

// 3. Ternary
#define BENCHMARK_TERNARY(Optr, size) \
    B.set_param(Mode::Optr); \
    B.execs({{size}, {size}, {size}, {}});

#define BENCHMARK_TERNARY_SCALAR(Optr, size) \
    B.set_param(Mode::Optr); \
    B.execs({{size}, {size}, {1}, {}});

#define BENCHMARK_TERNARY_1C11(Optr, chan) \
    B.set_param(Mode::Optr); \
    B.execs({{1, chan, 1, 1}, {9, chan, 33, 127}, {1, chan, 1, 1}, {}});

#define BENCHMARK_TERNARY_ALL_KINDS(Optr, size) \
    printf("Test for %s \n", #Optr); \
    BENCHMARK_TERNARY(Optr, size) \
    BENCHMARK_TERNARY_SCALAR(Optr, size) \
    BENCHMARK_TERNARY_1C11(Optr, (size/37719))

#define BENCHMARK_CASE_INT(size) \
    BENCHMARK_BINARY_ALL_KINDS(ADD, size) \
    BENCHMARK_BINARY_ALL_KINDS(SUB, size) \
    BENCHMARK_BINARY_ALL_KINDS(MUL, size) \
    BENCHMARK_BINARY_ALL_KINDS(TRUE_DIV, size) \
    BENCHMARK_BINARY_ALL_KINDS(MIN, size) \
    BENCHMARK_BINARY_ALL_KINDS(MAX, size) \
    BENCHMARK_UNARY(RELU, size) \
    BENCHMARK_UNARY(ABS, size) \
    BENCHMARK_BINARY_ALL_KINDS(FUSE_ADD_RELU, size) \
    BENCHMARK_TERNARY_ALL_KINDS(FUSE_MUL_ADD3, size)


#define BENCHMARK_CASE_FLOAT(size) \
    BENCHMARK_CASE_INT(size) \
    BENCHMARK_BINARY_ALL_KINDS(FUSE_ADD_TANH, size) \
    BENCHMARK_BINARY_ALL_KINDS(FUSE_ADD_SIGMOID, size) \


TEST_F(X86, BENCHMARK_ELEM_EVERY_DTYPE)
{
    Benchmarker<ElemwiseForward> B(handle());
    using Mode = ElemwiseForward::Param::Mode;

    printf("\nTest case float32:\n");
    B.set_dtype(0, dtype::Float32());
    B.set_dtype(1, dtype::Float32());
    B.set_dtype(2, dtype::Float32());
    BENCHMARK_CASE_FLOAT(1556011)

    //printf("\nTest case int32:\n");
    //B.set_dtype(0, dtype::Int32());
    //B.set_dtype(1, dtype::Int32());
    //B.set_dtype(2, dtype::Int32());
    //BENCHMARK_CASE_INT(1556011)

    //printf("\nTest case int16:\n");
    //B.set_dtype(0, dtype::Int16());
    //B.set_dtype(1, dtype::Int16());
    //B.set_dtype(2, dtype::Int16());
    //BENCHMARK_CASE_INT(1556011)

    //printf("\nTest case int8:\n");
    //B.set_dtype(0, dtype::Int8());
    //B.set_dtype(1, dtype::Int8());
    //B.set_dtype(2, dtype::Int8());
    //BENCHMARK_CASE_INT(1556011)

}
