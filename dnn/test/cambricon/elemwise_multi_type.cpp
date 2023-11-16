#include "test/common/elemwise_multi_type.h"
#include "megdnn/oprs/nn_int.h"
#include "src/cambricon/utils.h"
#include "test/cambricon/fixture.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

namespace {
template <typename tag>
class CAMBRICON_ELEMWISE_MULTI_TYPE : public CAMBRICON {};
TYPED_TEST_SUITE(CAMBRICON_ELEMWISE_MULTI_TYPE, elemwise_multi_type::test_types);
}  // anonymous namespace

TYPED_TEST(CAMBRICON_ELEMWISE_MULTI_TYPE, run) {
    elemwise_multi_type::run_test<TypeParam>(this->handle_cambricon());
}

using Mode = ElemwiseMultiType::Param::Mode;
static void run_test_bool(int arity, Checker<ElemwiseMultiType>& checker, Mode mode) {
    for (DType type :
         std::vector<DType>{{dtype::Int8()}, {dtype::Float32()}, {dtype::Float16()}}) {
        checker.set_param(mode);
        UniformIntRNG rng_int8{1, 1};
        NormalRNG rng_normal{0, 1};

        auto set_inp_rng = [&](DType dtype, size_t i) {
            if (dtype.enumv() == DTypeEnum::Int8) {
                checker.set_rng(i, &rng_int8);
            } else if (
                    dtype.enumv() == DTypeEnum::Float32 ||
                    dtype.enumv() == DTypeEnum::Float16) {
                checker.set_rng(i, &rng_normal);
            } else {
                megdnn_assert(0);
            }
            checker.set_dtype(i, dtype);
        };
        auto src_type = type;
        for (int i = 0; i < arity; i++) {
            set_inp_rng(src_type, i);
        }

        if (arity == 2) {
            checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {}});
        } else {
            megdnn_assert(0);
        }
    }
}

static void run_test_bool_with_non_contiguous_input(
        int arity, Checker<ElemwiseMultiType>& checker, Mode mode) {
    for (DType type :
         std::vector<DType>{{dtype::Int8()}, {dtype::Float32()}, {dtype::Float16()}}) {
        checker.set_param(mode);
        UniformIntRNG rng_int8{1, 1};
        NormalRNG rng_normal{0, 1};

        auto set_inp_rng = [&](DType dtype, size_t i) {
            if (dtype.enumv() == DTypeEnum::Int8) {
                checker.set_rng(i, &rng_int8);
            } else if (
                    dtype.enumv() == DTypeEnum::Float32 ||
                    dtype.enumv() == DTypeEnum::Float16) {
                checker.set_rng(i, &rng_normal);
            } else {
                megdnn_assert(0);
            }
            checker.set_dtype(i, dtype);
        };
        auto src_type = type;
        for (int i = 0; i < arity; i++) {
            set_inp_rng(src_type, i);
        }

        if (arity == 2) {
            checker.execl(
                    {{{3, 4, 5, 6}, {6, 5, 8, 10}, type},
                     {{3, 4, 5, 6}, {6, 5, 8, 10}, type},
                     {}});
            checker.execl(
                    {{{3, 4, 5, 6}, {6, 5, 8, 10}, type},
                     {{4, 5, 6}, {5, 8, 10}, type},
                     {}});
            checker.execl(
                    {{{3, 4, 5, 6}, {6, 5, 8, 10}, type},
                     {{4, 5, 6}, {10, 16, 20}, type},
                     {}});
            checker.execl(
                    {{{3, 4, 5, 6}, {6, 5, 8, 10}, type},
                     {{4, 5, 6}, {2, 3, 1}, type},
                     {}});
            checker.execl(
                    {{{3, 4, 5, 6}, {6, 5, 6, 1}, type},
                     {{3, 4, 5, 6}, {6, 40, 8, 10}, type},
                     {}});
            checker.execl(
                    {{{3, 4, 5, 6}, {20, 5, 60, 10}, type},
                     {{4, 5, 6}, {5, 60, 10}, type},
                     {}});
            checker.execl(
                    {{{3, 4, 5, 6}, {20, 5, 60, 10}, type},
                     {{4, 5, 6}, {300, 60, 10}, type},
                     {}});
            checker.execl(
                    {{{3, 4, 1, 6}, {6, 5, 0, 1}, type},
                     {{3, 4, 1, 6}, {6, 40, 8, 10}, type},
                     {}});
            checker.execl(
                    {{{3, 4, 1, 6}, {6, 5, 6, 1}, type},
                     {{3, 4, 2, 6}, {6, 40, 8, 10}, type},
                     {}});
        } else {
            megdnn_assert(0);
        }
    }
}

TEST_F(CAMBRICON, ELEMWISE_BOOL_MODE_BINARY) {
    using Mode = ElemwiseMultiType::Param::Mode;

    Checker<ElemwiseMultiType> checker(handle_cambricon());
    for (auto mode : {Mode::EQ, Mode::LEQ, Mode::LT, Mode::NEQ}) {
        run_test_bool(2, checker, mode);
        run_test_bool_with_non_contiguous_input(2, checker, mode);
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
