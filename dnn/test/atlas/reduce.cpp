#include "megdnn/oprs.h"
#include "test/atlas/fixture.h"
#include "test/common/checker.h"
#include "test/common/elemwise.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
namespace megdnn {
namespace test {
TEST_F(ATLAS, REDUCE_FORWARD_DEFAULT) {
    using Param = megdnn::param::Reduce;
    using Mode = Param::Mode;
    auto run_reduce = [&](DType dtype) {
        Checker<ReduceForward> checker(handle_atlas());
        Param param;
        param.data_type = Param::DataType::DEFAULT;
        checker.set_dtype(0, dtype).set_dtype(1, dtype);
        if (dtype == dtype::Float16()) {
            checker.set_epsilon(1e-2);
        }
        for (auto mode :
             {Mode::MIN, Mode::MAX, Mode::SUM, Mode::PRODUCT, Mode::MEAN,
              Mode::SUM_SQR}) {
            param.mode = mode;
            for (size_t n : {1, 2}) {
                for (size_t ic : {8, 256, 1024}) {
                    param.axis = 1;
                    checker.set_param(param);
                    checker.execs({{n, ic}, {n, 1}});
                    checker.execs({{n, ic, 28, 28}, {n, 1, 28, 28}});
                    param.axis = 2;
                    checker.set_param(param);
                    checker.execs({{n, ic, 28, 28}, {n, ic, 1, 28}});
                    param.axis = 3;
                    checker.set_param(param);
                    checker.execs({{n, ic, 28, 28}, {n, ic, 28, 1}});
                    if (n != 1) {
                        param.axis = 0;
                        checker.set_param(param);
                        checker.execs({{n, ic, 56, 56}, {1, ic, 56, 56}});
                    }
                }
            }
            //! test {N} to {1}
            param.axis = 0;
            checker.set_param(param);
            checker.execs({{1000}, {1}});
        }
    };
    run_reduce(dtype::Float32());
    //! TODO: atlas reduction with fp16 have precision problem
    // run_reduce(dtype::Float16());
    run_reduce(dtype::Int32());

    Checker<ReduceForward> checker(handle_atlas());
    UniformFloatRNG rng(-1.0f, 1.0f);
    checker.set_rng(0, &rng);
    checker.set_param({Mode::SUM, 1});

    checker.execs({{100, 160, 1}, {}});
    checker.execs({{100, 31, 1}, {}});
    // 1-step
    checker.execs({{2, 64, 32}, {}});
    // 2-step
    checker.execs({{2, 192, 32}, {}});
    // 3-step
    checker.execs({{2, 4333, 32}, {}});
    // single reduce
    checker.execs({{2, 1, 1}, {}});
    checker.execs({{2, 1 + 1, 1}, {}});
    checker.execs({{2, 2048 + 1, 1}, {}});
    checker.execs({{2, 2048 * 2048 + 1, 1}, {}});
    checker.execs({{2, 1 + 1, 31}, {}});
    checker.execs({{2, 16 + 1, 31}, {}});
    checker.execs({{2, 16 * 16 + 1, 31}, {}});
    checker.execs({{2, 16 * 16 * 16 + 1, 31}, {}});
    checker.execs({{2, 16 * 16 * 16 * 16 + 1, 31}, {}});
    checker.execs({{2, 8 * 16 * 16 * 16 * 16 + 1, 31}, {}});
    checker.execs({{2, 16 * 16 * 16 * 16 * 16 + 1, 31}, {}});
    checker.execs({{3, 256 * 256 + 1, 2}, {}});
    checker.execs({{3, 128 * 128 + 1, 3}, {}});
    checker.execs({{3, 64 * 64 + 1, 7}, {}});
    checker.execs({{3, 32 * 32 + 1, 15}, {}});
    checker.execs({{3, 512, 500}, {}});

    // very large reduce
    checker.execs({{1, 4194304, 1}, {}});
    {
        // very large reduce for CO32
        Reduce::Param param{Mode::SUM, 1, Reduce::Param::DataType::FLOAT_O32xC32};
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_param(param)
                .execs({{1, 4194304, 1}, {1, 1, 1}});
    }
    {
        // large reduce_mean for O16C32
        Param param{Mode::MEAN, 1, Reduce::Param::DataType::FLOAT_O16xC32};
        checker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_param(param)
                .execs({{1, 65536, 5}, {1, 1, 5}});
    }

    // inputs have nan
    {
        const auto nan = std::numeric_limits<float>::quiet_NaN();
        UniformFloatWithValueRNG rng1 =
                UniformFloatWithValueRNG(-1.0f, 1.0f, 0.5f, nan);
        checker.set_allow_invalid_check(true).set_rng(0, &rng1);
        for (auto mode : {Mode::MIN, Mode::MAX}) {
            checker.set_param({mode, 1});
            checker.execs({{2, 64, 32}, {}});
        }
        checker.set_allow_invalid_check(false);
    }
}

TEST_F(ATLAS, REDUCE_FORWARD_FLOAT_O32xC32) {
    using Param = megdnn::param::Reduce;
    using Mode = Param::Mode;
    Checker<ReduceForward> checker(handle_atlas());
    auto run_reduce = [&](DType dtype) {
        Param param;
        param.data_type = Param::DataType::FLOAT_O32xC32;
        checker.set_dtype(0, dtype).set_dtype(1, dtype);
        for (auto mode :
             {Mode::MIN, Mode::MAX, Mode::SUM, Mode::PRODUCT, Mode::MEAN,
              Mode::SUM_SQR}) {
            param.mode = mode;
            for (size_t n : {1, 2}) {
                for (size_t ic : {8, 256, 1024}) {
                    param.axis = 1;
                    checker.set_param(param);
                    checker.execs({{n, ic}, {n, 1}});
                    checker.execs({{n, ic, 28, 28}, {n, 1, 28, 28}});
                    param.axis = 2;
                    checker.set_param(param);
                    checker.execs({{n, ic, 28, 28}, {n, ic, 1, 28}});
                    param.axis = 3;
                    checker.set_param(param);
                    checker.execs({{n, ic, 28, 28}, {n, ic, 28, 1}});
                    if (n != 1) {
                        param.axis = 0;
                        checker.set_param(param);
                        checker.execs({{n, ic, 56, 56}, {1, ic, 56, 56}});
                    }
                }
            }
            //! test {N} to {1}
            param.axis = 0;
            checker.set_param(param);
            checker.execs({{1000}, {1}});
        }
    };
    run_reduce(dtype::Float32());
}

TEST_F(ATLAS, REDUCE_FORWARD_FLOAT_O16xC32) {
    using Param = megdnn::param::Reduce;
    using Mode = Param::Mode;
    Checker<ReduceForward> checker(handle_atlas());
    auto run_reduce = [&](DType dtype) {
        Param param;
        param.data_type = Param::DataType::FLOAT_O16xC32;
        checker.set_dtype(0, dtype).set_dtype(1, dtype);
        for (auto mode :
             {Mode::MIN, Mode::MAX, Mode::SUM, Mode::PRODUCT, Mode::MEAN,
              Mode::SUM_SQR}) {
            param.mode = mode;
            for (size_t n : {1, 2}) {
                for (size_t ic : {8, 256, 1024}) {
                    param.axis = 1;
                    checker.set_param(param);
                    checker.execs({{n, ic}, {n, 1}});
                    checker.execs({{n, ic, 28, 28}, {n, 1, 28, 28}});
                    param.axis = 2;
                    checker.set_param(param);
                    checker.execs({{n, ic, 28, 28}, {n, ic, 1, 28}});
                    param.axis = 3;
                    checker.set_param(param);
                    checker.execs({{n, ic, 28, 28}, {n, ic, 28, 1}});
                    if (n != 1) {
                        param.axis = 0;
                        checker.set_param(param);
                        checker.execs({{n, ic, 56, 56}, {1, ic, 56, 56}});
                    }
                }
            }
            //! test {N} to {1}
            param.axis = 0;
            checker.set_param(param);
            checker.execs({{1000}, {1}});
        }
    };
    run_reduce(dtype::Float16());
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
