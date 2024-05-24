#include "test/common/fill.h"

#include "test/atlas/fixture.h"
#include "test/naive/fixture.h"

using namespace megdnn;
using namespace test;

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
        checker.set_dtype(2, dtype::Bool());

        if (arity == 2) {
            // checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {}});
            checker.execs({{3, 4, 5, 6}, {3, 4, 5, 1}, {}});
        } else {
            megdnn_assert(0);
        }
    }
}

TEST_F(ATLAS, ELEMWISE_BOOL_MODE_BINARY) {
    using Mode = ElemwiseMultiType::Param::Mode;

    Checker<ElemwiseMultiType> checker(handle_atlas());
    for (auto mode : {Mode::EQ, Mode::LT, Mode::LEQ, Mode::NEQ}) {
        run_test_bool(2, checker, mode);
    }
}