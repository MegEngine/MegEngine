/**
 * \file dnn/test/common/elemwise.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/elemwise.h"
#include "src/common/utils.cuh"
#include "test/common/utils.h"
#include "test/common/checker.h"

#include "megdnn/oprs/general.h"

#include "test/common/fix_gtest_on_platforms_without_exception.inl"

using namespace megdnn;
using namespace test;

namespace {
    void fma3_extra_opr_impl(const TensorNDArray &data) {
        megdnn_assert(data.size() == 4);
        auto handle = create_cpu_handle(2);
        auto opr = handle->create_operator<Elemwise>();
        using Mode = Elemwise::Mode;
        opr->param().mode = Mode::MUL;
        opr->exec({data[0], data[1]}, data[3]);
        opr->param().mode = Mode::ADD;
        opr->exec({data[2], data[3]}, data[3]);
    }

    void fma4_extra_opr_impl(const TensorNDArray &data) {
        megdnn_assert(data.size() == 5);
        std::vector<uint8_t> tmp_storage(data[4].layout.span().dist_byte());
        TensorND tmp;
        tmp.raw_ptr = tmp_storage.data();
        tmp.layout = data[4].layout;
        tmp.layout.init_contiguous_stride();
        auto handle = create_cpu_handle(2);
        auto opr = handle->create_operator<Elemwise>();
        using Mode = Elemwise::Mode;
        opr->param().mode = Mode::MUL;
        opr->exec({data[0], data[1]}, data[4]);
        opr->exec({data[2], data[3]}, tmp);
        opr->param().mode = Mode::ADD;
        opr->exec({tmp, data[4]}, data[4]);
    }

    TensorLayout make_layout(const TensorShape &shp,
            std::initializer_list<ptrdiff_t> stride) {
        TensorLayout ret{shp, dtype::Float32()};
        megdnn_assert(stride.size() == shp.ndim);
        auto idx = 0;
        for (auto i: stride)
            ret.stride[idx ++] = i;
        return ret;
    }
} // anonymous namespace

namespace megdnn {
namespace test {
namespace elemwise {

#define DEF_TEST(name) \
template<> \
void run_test<name>(Handle *handle)

DEF_TEST(unary) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle);
    checker.set_param(Mode::SIN);
    checker.set_dtype(0, dtype::Float32()).execs({{3, 4, 1}, {}});
    checker.set_dtype(0, dtype::Float16()).execs({{3, 4, 1}, {}});
}

DEF_TEST(binary_brdcst) {
    auto run = [&](DType dtype) {
        using Mode = ElemwiseForward::Param::Mode;
        Checker<ElemwiseForward> checker(handle);
        checker.set_param(Mode::ADD);
        checker.set_dtype(0, dtype);
        checker.set_dtype(1, dtype);
        checker.execs({{3, 1}, {1, 3}, {3, 3}});
        {
            checker.execs({{10, 11},
                    {10, 11},
                    {10, 11}});
            //
            checker.execs({{2, 3, 4, 5, 6, 7},
                    {1, 3, 1, 1, 6, 1},
                    {2, 3, 4, 5, 6, 7}});
            checker.execs({{1, 3, 1, 1, 6, 1},
                    {2, 3, 4, 5, 6, 7},
                    {2, 3, 4, 5, 6, 7}});
            //
            checker.execs({{256, 256, 3},
                    {1, 1, 3},
                    {256, 256, 3}});
            checker.execs({{1, 1, 3},
                    {256, 256, 3},
                    {256, 256, 3}});
            //
            checker.execs({{8, 1, 6, 1},
                    {1, 7, 1, 5},
                    {8, 7, 6, 5}});
            checker.execs({{1, 7, 1, 5},
                    {8, 1, 6, 1},
                    {8, 7, 6, 5}});
            //
            checker.execs({{5, 4},
                    {1, 1},
                    {5, 4}});
            checker.execs({{1, 1},
                    {5, 4},
                    {5, 4}});
            //
            checker.execs({{5, 4},
                    {1, 4},
                    {5, 4}});
            checker.execs({{1, 4},
                    {5, 4},
                    {5, 4}});
            //
            checker.execs({{15, 3, 5},
                    {15, 1, 5},
                    {15, 3, 5}});
            checker.execs({{15, 1, 5},
                    {15, 3, 5},
                    {15, 3, 5}});
            //
            checker.execs({{15, 3, 5},
                    {1, 3, 5},
                    {15, 3, 5}});
            checker.execs({{1, 3, 5},
                    {15, 3, 5},
                    {15, 3, 5}});
            //
            checker.execs({{15, 3, 5},
                    {1, 3, 1},
                    {15, 3, 5}});
            checker.execs({{1, 3, 1},
                    {15, 3, 5},
                    {15, 3, 5}});
            //
            checker.execs({{3, 1},
                    {1, 4},
                    {3, 4}});
            // numpy broadcast
            checker.execs({
                    {2, 3, 1, 5}, {4, 5}, {2, 3, 4, 5}});
            checker.execs({
                    {3, 1, 1}, {4, 5}, {3, 4, 5}});
        }

        {
            // 1d
            {
                auto n = 1000u;
                checker.execs({{n}, {n}, {n}});
                checker.execs({{1}, {n}, {n}});
                checker.execs({{n}, {1}, {n}});
            }

            // 2d
            {
                auto m = 200u, n = 100u;
                auto collapse = [](size_t n, bool is_collapsed) {
                    return is_collapsed ? 1u : n;
                };

                for (auto msk = 0u; msk < 16; ++msk) {
                    checker.execs({
                            {collapse(m, msk&1), collapse(n,msk&2)},
                            {collapse(m, msk&4), collapse(n,msk&8)},
                            {}});
                }
            }
            // nd
            {
                checker.execs({
                        {2, 3, 4, 5, 6},
                        {1, 3, 1, 5, 6},
                        {2, 3, 4, 5, 6}});
                checker.execs({
                        {2, 3, 4, 5, 6},
                        {2, 1, 4, 1, 6},
                        {2, 3, 4, 5, 6}});
            }
        }

    };
    run(dtype::Float32());
    //run(dtype::Float16());
}

DEF_TEST(binary_non_contig) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle);
    checker.set_param(Mode::ADD);
    TensorLayout ly{{2, 3}, dtype::Float32()};
    ly.stride[0] = 4;
    checker.execl({ly, ly, {{2, 3}, dtype::Float32()}});
}

DEF_TEST(ternary) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle);
    checker.set_param(Mode::COND_LEQ_MOV);
    checker.execs({{1, 3, 4}, {2, 1, 4}, {2, 3, 1}, {2, 3, 4}});
    checker.set_dtype(0, dtype::Float32()).
        set_dtype(1, dtype::Float32()).
        set_dtype(2, dtype::Float32()).
        execs({{1, 3, 4}, {2, 1, 4}, {2, 3, 1}, {2, 3, 4}});
    checker.set_dtype(0, dtype::Float16()).
        set_dtype(1, dtype::Float16()).
        set_dtype(2, dtype::Float16()).
        set_dtype(3, dtype::Float16()).
        execs({{1, 3, 4}, {2, 1, 4}, {2, 3, 1}, {2, 3, 4}});
    checker.execs({{2, 1, 1, 5}, {4, 5}, {3, 1, 1}, {2, 3, 4, 5}});
    checker.execs({{3, 1, 1}, {5}, {4, 1}, {3, 4, 5}});
    ASSERT_THROW(checker.execs({{2, 3, 4}, {4, 1}, {1}, {2, 3, 4}}),
                MegDNNError);
    ASSERT_THROW(checker.execs({{2, 4, 4}, {4, 1}, {3, 1, 1}, {2, 3, 4}}),
                MegDNNError);
}

DEF_TEST(ternary_non_contig) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle);
    checker.set_param(Mode::COND_LEQ_MOV);
    TensorLayout ly{{2, 3}, dtype::Float32()};
    ly.stride[0] = 4;
    checker.execl({ly, ly, ly, {{2, 3}, dtype::Float32()}});
}


DEF_TEST(fuse_mul_add3) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle);
    checker.set_param(Mode::FUSE_MUL_ADD3)
        .set_extra_opr_impl(fma3_extra_opr_impl);
    auto make_shape = [](const TensorShape &s0, const TensorShape &s1,
            const TensorShape &s2) {
        TensorShape dest;
        dest.ndim = s0.ndim;
        for (size_t i = 0; i < dest.ndim; ++ i) {
            auto a = i < s0.ndim ? s0[i] : 1;
            auto b = i < s1.ndim ? s1[i] : 1;
            dest[i] = std::max(a, b);
        }
        return TensorShapeArray{s0, s1, s2, dest};
    };
    checker.exec(make_shape({2, 1}, {2, 2}, {2, 2}));
    checker.exec(make_shape({2, 2}, {2, 1}, {2, 2}));
    checker.exec(make_shape({2, 2}, {2, 2}, {1}));
    checker.exec(make_shape({3, 1}, {1, 3}, {3, 1}));
    checker.exec(make_shape(
                {2, 1, 2, 1, 2, 1},
                {1, 2, 1, 2, 1, 2},
                {1}));
    checker.exec(make_shape({1, 1, 3}, {5, 8, 1}, {5, 8, 1}));
    checker.exec(make_shape({1, 192, 9, 16}, {1}, {1, 192, 9, 16}));
}

DEF_TEST(fuse_mul_add3_non_contig) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle);
    checker.set_param(Mode::FUSE_MUL_ADD3)
        .set_extra_opr_impl(fma3_extra_opr_impl);
    TensorLayout ly{{2, 3}, dtype::Float32()};
    ly.stride[0] = 4;
    checker.execl({ly, ly, ly, {{2, 3}, dtype::Float32()}});
}

DEF_TEST(fuse_mul_add4) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle);
    checker.set_param(Mode::FUSE_MUL_ADD4)
        .set_extra_opr_impl(fma4_extra_opr_impl);
    auto make_shape = [](const TensorShape &s0, const TensorShape &s1,
            bool swap = false) {
        TensorShape dest;
        dest.ndim = s0.ndim;
        for (size_t i = 0; i < dest.ndim; ++ i) {
            auto a = i < s0.ndim ? s0[i] : 1;
            auto b = i < s1.ndim ? s1[i] : 1;
            dest[i] = std::max(a, b);
        }
        TensorShapeArray ret{s0, s1, s0, s1, dest};
        if (swap)
            std::swap(ret[2], ret[3]);
        return ret;
    };
    checker.exec(make_shape({2, 2}, {2, 2}));
    checker.exec(make_shape({3, 1}, {1, 3}));
    checker.exec(make_shape(
                {2, 1, 2, 1, 2, 1},
                {1, 2, 1, 2, 1, 2}));
    checker.exec(make_shape({4, 2}, {1, 2}, true));
}

DEF_TEST(rmulh) {

    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle);

    auto run_for_dtype = [&checker](auto dtype) {
        auto minv = DTypeTrait<decltype(dtype)>::min();
        auto maxv = DTypeTrait<decltype(dtype)>::max();
        UniformIntRNG rng0{minv, maxv};
        UniformIntRNG rngM{(maxv >> 1) + 1, maxv};
        checker.set_param({Mode::RMULH})
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_rng(0, &rng0)
                .set_rng(1, &rngM);
        checker.execs({{7, 9, 11, 13}, {1}, {}})
                .execs({{16, 3, 256, 256}, {1}, {}})
                .execs({{2, 3, 1, 7}, {2, 3, 1, 7}, {}})
                .execs({{9, 5, 4}, {1, 5, 1}, {}})
                .execs({{233}, {1}, {}});
    };

    run_for_dtype(dtype::Int8());
    run_for_dtype(dtype::Int16());
    run_for_dtype(dtype::Int32());
}

/* ============= migrated from x86 tests ============= */

#define UNARY_TEST_CASE(_optr) \
    checker.set_param(Mode::_optr).execs({{1, 127}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 7}, {}});

#define BUILD_UNARY_TEST_CASE_INT \
    UNARY_TEST_CASE(RELU) \
    UNARY_TEST_CASE(ABS)

#define BUILD_UNARY_TEST_CASE_FLOAT \
    UNARY_TEST_CASE(ABS) \
    UNARY_TEST_CASE(LOG) \
    UNARY_TEST_CASE(COS) \
    UNARY_TEST_CASE(SIN) \
    UNARY_TEST_CASE(FLOOR) \
    UNARY_TEST_CASE(CEIL) \
    UNARY_TEST_CASE(SIGMOID) \
    UNARY_TEST_CASE(EXP) \
    UNARY_TEST_CASE(TANH) \
    UNARY_TEST_CASE(FAST_TANH) \
    UNARY_TEST_CASE(RELU) \
    UNARY_TEST_CASE(ROUND)

DEF_TEST(unary1) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle);
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
        checker.set_epsilon(1e-5);
    checker.set_dtype(0, dtype::Float32());
    BUILD_UNARY_TEST_CASE_FLOAT
}

#undef UNARY_TEST_CASE
#undef BUILD_UNARY_TEST_CASE_INT
#undef BUILD_UNARY_TEST_CASE_FLOAT


#define BINARY_TEST_CASE(_optr) \
    checker.set_param(Mode::_optr).execs({{3, 4, 7}, {3, 4, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 5, 7}, {1, 1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 1, 1, 1}, {3, 4, 5, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 2, 2}, {1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 1, 1}, {1, 2, 2}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 7}, {1, 7}, {}});

#define BUILD_BINARY_TEST_CASE \
    BINARY_TEST_CASE(MIN) \
    BINARY_TEST_CASE(MAX)

#define BINARY_COMPLATE_TEST_CASE(_optr) \
    checker.set_param(Mode::_optr).execs({{3, 4, 7}, {3, 4, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 5, 7}, {1, 4, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 4, 1, 1}, {3, 4, 5, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 7}, {1, 4, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 4, 1}, {3, 4, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 5, 7}, {1, 1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 1, 1, 1}, {3, 4, 5, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 7}, {1, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 2, 2}, {1, 2, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 2, 1}, {1, 2, 2}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 2, 2}, {1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 1, 1}, {1, 2, 2}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 1}, {3, 4, 1}, {}}); \

#define BUILD_BINARY_COMPLATE_TEST_CASE \
    BINARY_COMPLATE_TEST_CASE(ADD) \
    BINARY_COMPLATE_TEST_CASE(MUL) \
    BINARY_COMPLATE_TEST_CASE(MAX) \
    BINARY_COMPLATE_TEST_CASE(MIN) \
    BINARY_COMPLATE_TEST_CASE(SUB)

#define BUILD_BINARY_COMPLATE_TEST_CASE_FLOAT32 \
    BINARY_COMPLATE_TEST_CASE(POW)              \
    BINARY_COMPLATE_TEST_CASE(TRUE_DIV)         \
    BINARY_COMPLATE_TEST_CASE(FUSE_ADD_SIGMOID) \
    BINARY_COMPLATE_TEST_CASE(FUSE_ADD_TANH)    \
    BINARY_COMPLATE_TEST_CASE(FUSE_ADD_RELU)    \
    BINARY_COMPLATE_TEST_CASE(FUSE_ADD_H_SWISH) \
    BINARY_COMPLATE_TEST_CASE(FAST_TANH_GRAD)   \
    BINARY_COMPLATE_TEST_CASE(H_SWISH_GRAD)

DEF_TEST(binary1) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle);

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

#undef BINARY_TEST_CASE
#undef BUILD_BINARY_TEST_CASE
#undef BINARY_COMPLATE_TEST_CASE
#undef BUILD_BINARY_COMPLATE_TEST_CASE
#undef BUILD_BINARY_COMPLATE_TEST_CASE_FLOAT32

#define TERNARY_COMPLATE_TEST_CASE(_optr) \
    checker.set_param(Mode::_optr).execs({{3, 4, 7}, {3, 4, 7}, {3, 4, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 4, 1, 1}, {3, 4, 5, 7}, {1, 4, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 4, 1}, {3, 4, 7}, {1, 4, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 5, 7}, {3, 4, 5, 7}, {1, 1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 7}, {1, 7}, {1, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 2, 1}, {1, 2, 2}, {1, 2, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 2, 2}, {1, 2, 2}, {1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 1}, {3, 4, 1}, {3, 4, 1}, {}}); \

#define BUILD_TERNARY_COMPLATE_TEST_CASE \
    TERNARY_COMPLATE_TEST_CASE(FUSE_MUL_ADD3)


DEF_TEST(ternary1) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle);
    // case int
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int8());
    //BUILD_TERNARY_TEST_CASE
    BUILD_TERNARY_COMPLATE_TEST_CASE

    checker.set_dtype(0, dtype::Int16());
    checker.set_dtype(1, dtype::Int16());
    checker.set_dtype(2, dtype::Int16());
    //BUILD_TERNARY_TEST_CASE
    BUILD_TERNARY_COMPLATE_TEST_CASE

    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());
    checker.set_dtype(2, dtype::Int32());
    //BUILD_TERNARY_TEST_CASE
    BUILD_TERNARY_COMPLATE_TEST_CASE

    // case float
    UniformFloatRNG rng(1e-5, 7e1);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-5);
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());
    checker.set_dtype(2, dtype::Float32());

    //BUILD_TERNARY_TEST_CASE
    BUILD_TERNARY_COMPLATE_TEST_CASE
    //TERNARY_COMPLATE_TEST_CASE(FUSE_MUL_ADD3)
}

#undef TERNARY_COMPLATE_TEST_CASE
#undef BUILD_TERNARY_COMPLATE_TEST_CASE

/* ============= migrated from arm tests ============= */

#define UNARY_TEST_CASE(_optr) \
    checker.set_param(Mode::_optr).execs({{1, 129}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 7}, {}});

#define BUILD_UNARY_TEST_CASE_INT \
    UNARY_TEST_CASE(RELU) \
    UNARY_TEST_CASE(ABS) \
    UNARY_TEST_CASE(NEGATE)

#define BUILD_UNARY_TEST_CASE_FLOAT \
    BUILD_UNARY_TEST_CASE_INT       \
    UNARY_TEST_CASE(SIGMOID)        \
    UNARY_TEST_CASE(EXP)            \
    UNARY_TEST_CASE(TANH)           \
    UNARY_TEST_CASE(FAST_TANH)      \
    UNARY_TEST_CASE(H_SWISH)

DEF_TEST(unary2) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle);
    // case int
    checker.set_dtype(0, dtype::Int8());
    BUILD_UNARY_TEST_CASE_INT

    checker.set_dtype(0, dtype::Int16());
    BUILD_UNARY_TEST_CASE_INT

    checker.set_dtype(0, dtype::Int32());
    BUILD_UNARY_TEST_CASE_INT

    // case float
    {
        UniformFloatRNG rng(1e-5, 7e1);
        checker.set_rng(0, &rng);
        checker.set_epsilon(1e-5);
        checker.set_dtype(0, dtype::Float32());
        BUILD_UNARY_TEST_CASE_FLOAT
    }

    {
        UniformFloatRNG rng(1e-2, 1e1);
        checker.set_rng(0, &rng);
        checker.set_epsilon(6e-3);
        checker.set_dtype(0, dtype::Float16());
        BUILD_UNARY_TEST_CASE_FLOAT
    }

    // tanh NaN bug case
    {
        UniformFloatRNG rng(100, 200);
        checker.set_rng(0, &rng);
        checker.set_epsilon(1e-5);
        checker.set_dtype(0, dtype::Float32());
        checker.set_param(Mode::TANH).execs({{1, 1025}, {}});
        checker.set_param(Mode::TANH).execs({{1, 7}, {}});
    }
}

#undef UNARY_TEST_CASE
#undef BUILD_UNARY_TEST_CASE_INT
#undef BUILD_UNARY_TEST_CASE_FLOAT

#define BINARY_TEST_CASE(_optr) \
    checker.set_param(Mode::_optr).execs({{3, 4, 7}, {3, 4, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 5, 7}, {1, 1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 1, 1, 1}, {3, 4, 5, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 2, 2}, {1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 1, 1}, {1, 2, 2}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 7}, {1, 7}, {}});

#define BUILD_BINARY_TEST_CASE \
    BINARY_TEST_CASE(MIN) \
    BINARY_TEST_CASE(MAX)

#define BINARY_COMPLATE_TEST_CASE(_optr) \
    checker.set_param(Mode::_optr).execs({{3, 4, 7}, {3, 4, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 5, 7}, {1, 4, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 4, 1, 1}, {3, 4, 5, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 7}, {1, 4, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 4, 1}, {3, 4, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 5, 7}, {1, 1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 1, 1, 1}, {3, 4, 5, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 7}, {1, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 2, 2}, {1, 2, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 2, 1}, {1, 2, 2}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 2, 2}, {1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 1, 1}, {1, 2, 2}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 1}, {3, 4, 1}, {}});

#define BUILD_BINARY_COMPLATE_TEST_CASE \
    BINARY_COMPLATE_TEST_CASE(ADD)      \
    BINARY_COMPLATE_TEST_CASE(MUL)      \
    BINARY_COMPLATE_TEST_CASE(MAX)      \
    BINARY_COMPLATE_TEST_CASE(MIN)      \
    BINARY_COMPLATE_TEST_CASE(SUB)      \
    BINARY_COMPLATE_TEST_CASE(FUSE_ADD_RELU)

DEF_TEST(binary2) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle);

    // case float
    UniformFloatRNG rng(1e-5, 7e1);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-5);
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());

    BUILD_BINARY_COMPLATE_TEST_CASE
    BINARY_COMPLATE_TEST_CASE(FUSE_ADD_SIGMOID)
    BINARY_COMPLATE_TEST_CASE(FUSE_ADD_TANH)

    // case int
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    //BUILD_BINARY_TEST_CASE
    BUILD_BINARY_COMPLATE_TEST_CASE

    checker.set_dtype(0, dtype::Int16());
    checker.set_dtype(1, dtype::Int16());
    //BUILD_BINARY_TEST_CASE
    BUILD_BINARY_COMPLATE_TEST_CASE

    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());
    BUILD_BINARY_TEST_CASE
    BUILD_BINARY_COMPLATE_TEST_CASE

    // case float
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-5);
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());
    checker.set_param(Mode::FUSE_ADD_SIGMOID).execs({{3, 4, 7}, {1}, {}});
    checker.set_param(Mode::FUSE_ADD_TANH).execs({{3, 4, 7}, {1}, {}});

    // commutable
    checker.set_param(Mode::TRUE_DIV).execs({{1}, {4}, {}});

    BUILD_BINARY_TEST_CASE
    BUILD_BINARY_COMPLATE_TEST_CASE
    BINARY_COMPLATE_TEST_CASE(TRUE_DIV)

    {
        UniformFloatRNG rng(1e-3, 3e1);
        checker.set_rng(0, &rng);
        checker.set_rng(1, &rng);
        checker.set_epsilon(1e-3);
        checker.set_dtype(0, dtype::Float16());
        checker.set_dtype(1, dtype::Float16());
        checker.set_param(Mode::FUSE_ADD_SIGMOID).execs({{3, 4, 7}, {1}, {}});
        checker.set_param(Mode::FUSE_ADD_TANH).execs({{3, 4, 7}, {1}, {}});

        BUILD_BINARY_TEST_CASE
        BUILD_BINARY_COMPLATE_TEST_CASE
        BINARY_COMPLATE_TEST_CASE(TRUE_DIV)

        // commutable
        checker.set_param(Mode::TRUE_DIV).execs({{1}, {4}, {}});
    }

}

#undef BINARY_TEST_CASE
#undef BUILD_BINARY_TEST_CASE
#undef BINARY_COMPLATE_TEST_CASE
#undef BUILD_BINARY_COMPLATE_TEST_CASE

#define TERNARY_COMPLATE_TEST_CASE(_optr) \
    checker.set_param(Mode::_optr).execs({{1, 123, 1}, {300, 123, 253}, {300, 123, 253}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 7}, {3, 4, 7}, {3, 4, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 4, 1, 1}, {3, 4, 5, 7}, {1, 4, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 4, 1}, {3, 4, 7}, {1, 4, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 5, 7}, {3, 4, 5, 7}, {1, 1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 7}, {1, 7}, {1, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 2, 1}, {1, 2, 2}, {1, 2, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 2, 2}, {1, 2, 2}, {1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 1}, {3, 4, 1}, {3, 4, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 1}, {1, 1, 1}, {3, 4, 1}, {}}); \

#define BUILD_TERNARY_COMPLATE_TEST_CASE \
    TERNARY_COMPLATE_TEST_CASE(FUSE_MUL_ADD3)


DEF_TEST(ternary2) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle);
    // case int
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int8());
    BUILD_TERNARY_COMPLATE_TEST_CASE

    checker.set_dtype(0, dtype::Int16());
    checker.set_dtype(1, dtype::Int16());
    checker.set_dtype(2, dtype::Int16());
    BUILD_TERNARY_COMPLATE_TEST_CASE

    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());
    checker.set_dtype(2, dtype::Int32());
    BUILD_TERNARY_COMPLATE_TEST_CASE

    // case float
    UniformFloatRNG rng(1e-5, 7e1);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-5);
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());
    checker.set_dtype(2, dtype::Float32());
    BUILD_TERNARY_COMPLATE_TEST_CASE

    {
        UniformFloatRNG rng(1e-3, 3e1);
        checker.set_rng(0, &rng);
        checker.set_rng(1, &rng);
        checker.set_rng(2, &rng);
        checker.set_epsilon(1e-3);
        checker.set_dtype(0, dtype::Float16());
        checker.set_dtype(1, dtype::Float16());
        checker.set_dtype(2, dtype::Float16());
        BUILD_TERNARY_COMPLATE_TEST_CASE

    }
}

#undef TERNARY_COMPLATE_TEST_CASE
#undef BUILD_TERNARY_COMPLATE_TEST_CASE

/* ============= migrated from fallback tests ============= */

DEF_TEST(unary3) {
    Checker<Elemwise> checker(handle);
    auto make_layouts = [](
            const TensorShape &shp, std::initializer_list<ptrdiff_t> stride) ->
            TensorLayoutArray{
        return {make_layout(shp, stride), {shp, dtype::Float32()}};
    };
    checker.set_param({Elemwise::Mode::SIN});

    checker.exec(make_layouts({2, 2}, {2, 1}));
    checker.exec(make_layouts({4}, {3}));
}

DEF_TEST(binary3) {
    Checker<Elemwise> checker(handle);
    checker.set_param({Elemwise::Mode::ADD});

    auto run = [&](
            const TensorShape &shp0, std::initializer_list<ptrdiff_t> stride0,
            const TensorShape &shp1, std::initializer_list<ptrdiff_t> stride1) {
        TensorShape shpo;
        Elemwise::deduce_shape({shp0, shp1}, shpo);
        auto ly0 = make_layout(shp0, stride0),
             ly1 = make_layout(shp1, stride1),
             lyo = TensorLayout{shpo, dtype::Float32()};
        checker.execl({ly0, ly1, lyo});
        checker.execl({ly1, ly0, lyo});
    };

    run({2, 2}, {2, 1}, {2, 2}, {2, 1});
    run({1}, {1}, {3, 3}, {1, 2});
    run({3, 4, 5}, {40, 10, 2}, {1, 4, 1}, {1, 1, 1});
}

DEF_TEST(all_modes) {
    // test correctness of all elemwise modes
    Checker<Elemwise> checker(handle);
    TensorShapeArray shapes;
    UniformFloatRNG default_rng_f32{-100.f, 100.f}, pos_rng_f32{.1f, 1000.f},
            small_pos_rng_f32{.1f, .10f}, small_rng_f32{-3.f, 3.f},
            abslt1_rng_f32{-1.f, 1.f}, uniform_0_2_rng{0.f, 2.f},
            tanh_rng_f32{-5.f, 5.f};
    UniformFloatNonZeroRNG nonzero_rng_f32{.1f, 1000.f},
            big_nonzero_rng_f32{100.f, 1000.f};
    UniformIntRNG default_rng_i32{-100, 100}, small_rng_i32{-2, 2},
            shift_rng_i32_i32{0, 31}, shift_rng_i32_i8{0, 7};
    UniformIntNonZeroRNG nonzero_rng_i32{1, 100};

    using Mode = Elemwise::Mode;

    auto should_ignore = [handle](Mode mode) {
        MEGDNN_MARK_USED_VAR(mode);
        return false;
    };

    for (int mode_nr = 0;
         mode_nr < static_cast<int>(Elemwise::Param::MODE_NR_MEMBER);
         ++mode_nr) {
        auto mode = static_cast<Mode>(mode_nr);

        // ignore unsupported modes
        if (should_ignore(mode)) {
            continue;
        }

        checker.set_param({mode});
        auto&& trait = Elemwise::ModeTrait::from_mode(mode);
        shapes.resize(trait.arity + 1);
        for (size_t i = 0; i < shapes.size() - 1; ++i) {
            shapes[i] = {3, 9, 7};
        }
        auto do_run = [&](DType dtype, float eps = 1e-3) {
            // limit value ranges for some modes
            if (mode == Mode::LOG || mode == Mode::LOG1P) {
                checker.set_rng(0, &pos_rng_f32);
            } else if (mode == Mode::POW) {
                checker.set_rng(0, &small_pos_rng_f32);
                checker.set_rng(1, &small_rng_f32);
            } else if (mode == Mode::EXP || mode == Mode::EXPM1) {
                checker.set_rng(0, &small_rng_f32);
            } else if (mode == Mode::FAST_TANH) {
                checker.set_rng(0, &tanh_rng_f32);
            } else if (mode == Mode::LOG_SUM_EXP) {
                // check numerical stability with large values
                checker.set_rng(0, &big_nonzero_rng_f32);
                checker.set_rng(1, &big_nonzero_rng_f32);
            } else if (mode == Mode::ASIN || mode == Mode::ACOS ||
                       mode == Mode::SIGMOID_GRAD || mode == Mode::TANH_GRAD ||
                       mode == Mode::ERFINV) {
                checker.set_rng(0, &abslt1_rng_f32);
                checker.set_rng(1, &default_rng_f32);
            } else if (mode == Mode::ERFCINV) {
                checker.set_rng(0, &uniform_0_2_rng);
            } else if (mode == Mode::MOD || mode == Mode::TRUE_DIV ||
                       mode == Mode::FLOOR_DIV) {
                if (dtype.category() == DTypeCategory::INT) {
                    checker.set_rng(0, &default_rng_i32);
                    checker.set_rng(1, &nonzero_rng_i32);
                } else {
                    checker.set_rng(0, &default_rng_f32);
                    checker.set_rng(1, &nonzero_rng_f32);
                }
            } else if (mode == Mode::EQ) {
                checker.set_rng(0, &small_rng_i32);
                checker.set_rng(1, &small_rng_i32);
            } else if (mode == Mode::SHL || mode == Mode::SHR) {
                checker.set_rng(0, &default_rng_i32);
                if (dtype.size() == 4) {
                    checker.set_rng(1, &shift_rng_i32_i32);
                } else {
                    megdnn_assert(dtype.size() == 1);
                    checker.set_rng(1, &shift_rng_i32_i8);
                }
            } else if (mode == Mode::ATAN2) {
                checker.set_rng(0, &nonzero_rng_f32);
                checker.set_rng(1, &nonzero_rng_f32);
            } else {
                RNG* rng;
                if (dtype.category() == DTypeCategory::INT) {
                    rng = &default_rng_i32;
                } else {
                    rng = &default_rng_f32;
                }
                for (size_t i = 0; i < shapes.size(); ++i) {
                    checker.set_rng(i, rng);
                }
            }

            checker.set_epsilon(eps);
            for (size_t i = 0; i < shapes.size(); ++i) {
                checker.set_dtype(i, dtype);
            }
            EXPECT_NO_THROW(checker.execs(shapes));
            if (!::testing::Test::HasFailure() && shapes.size() == 3) {
                // channel bcast
                shapes[1][0] = 1;
                shapes[1][2] = 1;
                EXPECT_NO_THROW(checker.execs(shapes));

                if (!::testing::Test::HasFailure()) {
                    // scalar bcast
                    shapes[1][1] = 1;
                    EXPECT_NO_THROW(checker.execs(shapes));
                }
            }
            if (::testing::Test::HasFailure()) {
                printf("failed on mode=%d(%s) dtype=%s\n", mode_nr, trait.name,
                       dtype.name());
                for (auto&& i : shapes) {
                    printf("ishape: %s\n", i.to_string().c_str());
                }
                return false;
            }
            return true;
        };
#define run(args...)         \
    do {                     \
        if (!do_run(args)) { \
            return;          \
        }                    \
    } while (0)

        if (trait.allow_int) {
            run(dtype::Int8{});
            run(dtype::Int32{});
        }
        if (trait.allow_float) {
            MEGDNN_FLOAT16_SELECT(
                    run(dtype::Float16{},
                        mode == Mode::FAST_TANH_GRAD ? 0.5 : 0.05), );
            run(dtype::Float32{});
        }
    }

#undef run
}

TEST(TEST_ELEMWISE, MODE_TRAIT) {
    using M = Elemwise::Mode;
    using T = Elemwise::ModeTrait;
    ASSERT_EQ(1u, T::from_mode(M::RELU).arity);
    ASSERT_EQ(2u, T::from_mode(M::ADD).arity);
    ASSERT_EQ(3u, T::from_mode(M::FUSE_MUL_ADD3).arity);
    ASSERT_EQ(4u, T::from_mode(M::FUSE_MUL_ADD4).arity);
    ASSERT_TRUE(T::from_mode(M::ADD).commutable);
    ASSERT_FALSE(T::from_mode(M::TRUE_DIV).commutable);

    ASSERT_TRUE(T::from_mode(M::ADD).allow_int);
    ASSERT_FALSE(T::from_mode(M::EXP).allow_int);

    ASSERT_TRUE(T::from_mode(M::ADD).allow_float);
    ASSERT_FALSE(T::from_mode(M::SHL).allow_float);

    ASSERT_TRUE(T::from_mode(M::RMULH).commutable);
    ASSERT_FALSE(T::from_mode(M::RMULH).allow_float);

    ASSERT_TRUE(T::from_mode(M::XOR).allow_bool);
}

} // namespace elemwise
} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
