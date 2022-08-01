#include "test/fallback/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/task_record_check.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
using namespace megdnn;
using namespace test;

TEST_F(FALLBACK, REDUCE_FULL) {
    using Param = Reduce::Param;
    using Mode = Param::Mode;
    Checker<Reduce> checker(handle());
    UniformIntRNG rng{INT8_MIN >> 1, INT8_MAX >> 1};
    checker.set_rng(0, &rng);
    struct Config {
        Param param;
        DType dtype;
        TensorShape shape;
        Config(Param param, DType dtype, TensorShape shape)
                : param(param), dtype(dtype), shape(shape) {}
    };
    std::vector<Config> configs;
    for (auto mode : {Mode::MEAN, Mode::MAX, Mode::MIN})
        for (auto dtype : std::vector<DType>{
                     dtype::Float32(), dtype::Float16(), dtype::QuantizedS8(1.3f),
                     dtype::Quantized8Asymm(1.3f, static_cast<uint8_t>(3))})
            for (int32_t axis : {0, 1, 2}) {
                for (size_t A : {1, 3, 5, 20}) {
                    for (size_t B : {4, 6, 9, 16, 33, 45}) {
                        for (size_t C : {2, 3, 4, 6, 9, 16, 33, 45}) {
                            TensorShape shape{A, B, C};
                            Param param(mode, axis);
                            Config config(param, dtype, shape);
                            configs.push_back(config);
                        }
                    }
                }
            }
    for (auto&& config : configs) {
        auto&& dtype = config.dtype;
        auto&& param = config.param;
        auto&& shape = config.shape;

        checker.set_dtype(0, dtype).set_param(param).execs({shape, {}});
    }
    configs.clear();
    for (auto mode : {Mode::SUM, Mode::PRODUCT, Mode::SUM_SQR})
        for (auto dtype : std::vector<DType>{dtype::Float32(), dtype::Float16()})
            for (int32_t axis : {0, 1, 2}) {
                for (size_t A : {1, 3, 5, 20}) {
                    for (size_t B : {4, 6, 9, 16, 33, 45}) {
                        for (size_t C : {2, 3, 4, 6, 9, 16, 33, 45}) {
                            TensorShape shape{A, B, C};
                            Param param(mode, axis);
                            Config config(param, dtype, shape);
                            configs.push_back(config);
                        }
                    }
                }
            }

    UniformFloatRNG rng_float(-2, 2);
    checker.set_rng(0, &rng_float);
    checker.set_epsilon(1e-1);
    for (auto&& config : configs) {
        auto&& dtype = config.dtype;
        auto&& param = config.param;
        auto&& shape = config.shape;
        if (dtype == dtype::Float16())
            checker.set_epsilon(1e-1);
        else
            checker.set_epsilon(1e-3);

        checker.set_dtype(0, dtype).set_param(param).execs({shape, {}});
    }
}

TEST_F(FALLBACK, REDUCE) {
    using Param = Reduce::Param;
    using Mode = Param::Mode;
    using DataType = Param::DataType;
    Checker<Reduce> checker(handle());
    struct Config {
        Param param;
        DType dtype;
        TensorShape shape;
        Config(Param param, DType dtype, TensorShape shape)
                : param(param), dtype(dtype), shape(shape) {}
    };
    std::vector<Config> configs;
    // general
    for (auto mode :
         {Mode::SUM, Mode::MEAN, Mode::SUM_SQR, Mode::PRODUCT, Mode::MIN, Mode::MAX})
        for (auto dtype : std::vector<DType>{
                     dtype::Float16(), dtype::Float32(), dtype::Int32(), dtype::Int16(),
                     dtype::Int8(), dtype::Uint8()})
            for (int32_t axis : {0, 1, 2, 3}) {
                TensorShape shape{2, 3, 20, 5};
                Param param(mode, axis);
                Config config(param, dtype, shape);
                configs.push_back(config);
                if (dtype.category() == DTypeCategory::FLOAT) {
                    Param param(mode, axis, DataType::FLOAT_O16xC32);
                    Config config(param, dtype, shape);
                    configs.push_back(config);

                    param.data_type = DataType::FLOAT_O32xC32;
                    config = Config(param, dtype, shape);
                    configs.push_back(config);
                } else if (dtype == dtype::Int32()) {
                    Param param(mode, axis, DataType::FLOAT_O32xC32);
                    Config config(param, dtype, shape);
                    configs.push_back(config);
                }
            }
    // large (ABC) -> (A1C) case
    for (auto mode : {Mode::SUM_SQR})
        for (auto dtype : std::vector<DType>{dtype::Int32()})
            for (int32_t axis : {0, 1, 2, 3}) {
                TensorShape shape{2, 3, 10000, 5};
                Param param(mode, axis);
                Config config(param, dtype, shape);
                configs.push_back(config);
            }
    // large (AB) -> (A1) case
    for (auto mode : {Mode::SUM_SQR})
        for (auto dtype : std::vector<DType>{dtype::Int32()})
            for (int32_t axis : {0, 1, 2, 3}) {
                TensorShape shape{2, 3, 5, 10000};
                Param param(mode, axis);
                Config config(param, dtype, shape);
                configs.push_back(config);
            }

    {
        // large reduce_mean for O16C32
        TensorShape shape{1, 65536, 5};
        Param param(Mode::MEAN, 1, DataType::FLOAT_O16xC32);
        Config config(param, dtype::Float16(), shape);
        configs.push_back(config);
    }

    for (auto&& config : configs) {
        auto&& dtype = config.dtype;
        auto&& param = config.param;
        auto&& mode = config.param.mode;
        auto&& shape = config.shape;
        auto&& data_type = config.param.data_type;
        // when input/output both float16, the internal compute is float16, mode
        // is SUM or SUM_SQR, need set epsilon to 1e-2 to pass test
        if (dtype == dtype::Float16() && data_type == DataType::DEFAULT &&
            (mode == Mode::SUM || mode == Mode::SUM_SQR)) {
            checker.set_epsilon(1e-2);
        }

        checker.set_dtype(0, dtype).set_param(param).execs({shape, {}});
    }
    {
        static size_t N = 1 << 26;
        {
            // cpu vs naive
            Checker<Reduce> checker(handle());
            Reduce::Param param;
            param.axis = 0;
            UniformFloatRNG rng(1, 1);
            checker.set_param(param);
            checker.set_rng(0, &rng);
            checker.execs({{N}, {}});
        }
        {
            // naive vs groundtruth
            TensorLayout layoutN(TensorShape{N}, dtype::Float32()),
                    layout1(TensorShape{1}, dtype::Float32());
            auto handle = this->handle();
            Tensor<float> src(handle, layoutN), dst(handle, layout1);
            float* ptr = src.ptr();
            for (size_t i = 0; i < N; ++i)
                ptr[i] = 1;
            auto opr = handle->create_operator<Reduce>();
            opr->param().axis = 0;
            auto wsize = opr->get_workspace_in_bytes(layoutN, layout1);
            WorkspaceWrapper workspace(handle, wsize);
            opr->exec(src.tensornd(), dst.tensornd(), workspace.workspace());
            megdnn_sync(handle);
            ASSERT_EQ(N, dst.ptr()[0]);
        }
    }
}

TEST_F(FALLBACK, REDUCE_RECORD) {
    using Param = Reduce::Param;
    using Mode = Param::Mode;
    using DataType = Param::DataType;
    TaskRecordChecker<Reduce> checker(1);
    struct Config {
        Param param;
        DType dtype;
        TensorShape shape;
        Config(Param param, DType dtype, TensorShape shape)
                : param(param), dtype(dtype), shape(shape) {}
    };
    std::vector<Config> configs;
    // general
    for (auto mode :
         {Mode::SUM, Mode::MEAN, Mode::SUM_SQR, Mode::PRODUCT, Mode::MIN, Mode::MAX})
        for (auto dtype : std::vector<DType>{
                     dtype::Float16(), dtype::Float32(), dtype::Int32(), dtype::Int16(),
                     dtype::Int8(), dtype::Uint8()})
            for (int32_t axis : {0, 1, 2, 3}) {
                TensorShape shape{2, 3, 20, 5};
                Param param(mode, axis);
                Config config(param, dtype, shape);
                configs.push_back(config);
                if (dtype.category() == DTypeCategory::FLOAT) {
                    Param param(mode, axis, DataType::FLOAT_O16xC32);
                    Config config(param, dtype, shape);
                    configs.push_back(config);

                    param.data_type = DataType::FLOAT_O32xC32;
                    config = Config(param, dtype, shape);
                    configs.push_back(config);
                } else if (dtype == dtype::Int32()) {
                    Param param(mode, axis, DataType::FLOAT_O32xC32);
                    Config config(param, dtype, shape);
                    configs.push_back(config);
                }
            }
    // large (ABC) -> (A1C) case
    for (auto mode : {Mode::SUM_SQR})
        for (auto dtype : std::vector<DType>{dtype::Int32()})
            for (int32_t axis : {0, 1, 2, 3}) {
                TensorShape shape{2, 3, 10000, 5};
                Param param(mode, axis);
                Config config(param, dtype, shape);
                configs.push_back(config);
            }
    // large (AB) -> (A1) case
    for (auto mode : {Mode::SUM_SQR})
        for (auto dtype : std::vector<DType>{dtype::Int32()})
            for (int32_t axis : {0, 1, 2, 3}) {
                TensorShape shape{2, 3, 5, 10000};
                Param param(mode, axis);
                Config config(param, dtype, shape);
                configs.push_back(config);
            }

    {
        // large reduce_mean for O16C32
        TensorShape shape{1, 65536, 5};
        Param param(Mode::MEAN, 1, DataType::FLOAT_O16xC32);
        Config config(param, dtype::Float16(), shape);
        configs.push_back(config);
    }

    for (auto&& config : configs) {
        auto&& dtype = config.dtype;
        auto&& param = config.param;
        auto&& mode = config.param.mode;
        auto&& shape = config.shape;
        auto&& data_type = config.param.data_type;
        // when input/output both float16, the internal compute is float16, mode
        // is SUM or SUM_SQR, need set epsilon to 1e-2 to pass test
        if (dtype == dtype::Float16() && data_type == DataType::DEFAULT &&
            (mode == Mode::SUM || mode == Mode::SUM_SQR)) {
            checker.set_epsilon(1e-2);
        }

        checker.set_dtype(0, dtype).set_param(param).execs({shape, {}});
    }
    {
        static size_t N = 1 << 26;
        {
            // cpu vs naive
            TaskRecordChecker<Reduce> checker(1);
            Reduce::Param param;
            param.axis = 0;
            UniformFloatRNG rng(1, 1);
            checker.set_param(param);
            checker.set_rng(0, &rng);
            checker.execs({{N}, {}});
        }
        {
            // naive vs groundtruth
            TensorLayout layoutN(TensorShape{N}, dtype::Float32()),
                    layout1(TensorShape{1}, dtype::Float32());
            auto handle = this->handle();
            Tensor<float> src(handle, layoutN), dst(handle, layout1);
            float* ptr = src.ptr();
            for (size_t i = 0; i < N; ++i)
                ptr[i] = 1;
            auto opr = handle->create_operator<Reduce>();
            opr->param().axis = 0;
            auto wsize = opr->get_workspace_in_bytes(layoutN, layout1);
            WorkspaceWrapper workspace(handle, wsize);
            opr->exec(src.tensornd(), dst.tensornd(), workspace.workspace());
            megdnn_sync(handle);
            ASSERT_EQ(N, dst.ptr()[0]);
        }
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(FALLBACK, BENCHMARK_REDUCE_VS_CONV) {
    auto run = [&]() {
        Benchmarker<Reduce> benchmarker_reduce(handle());
        Benchmarker<Convolution> benchmarker_conv(handle());
        benchmarker_reduce.set_display(false);
        benchmarker_conv.set_display(false);
        constexpr size_t RUNS = 50;
        benchmarker_reduce.set_times(RUNS);
        benchmarker_conv.set_times(RUNS);
        param::Reduce param;
        param.axis = 3;
        param.mode = param::Reduce::Mode::SUM;
        benchmarker_reduce.set_param(param);
        param::Convolution param_conv;
        benchmarker_conv.set_param(param_conv);

        {
            TensorLayout src({24, 240, 128, 2}, dtype::Float32());
            auto reduce = benchmarker_reduce.execs({src, {}}) / RUNS;
            TensorLayout conv_src({24, 2, 240, 128}, dtype::Float32());
            TensorLayout conv_weight({1, 2, 1, 1}, dtype::Float32());
            auto conv = benchmarker_conv.execs({conv_src, conv_weight, {}}) / RUNS;

            printf("case 1: reduce use time %fms, convolution use time %fms\n", reduce,
                   conv);
        }
        {
            TensorLayout src({24, 240, 128, 3}, dtype::Float32());
            auto reduce = benchmarker_reduce.execs({src, {}}) / RUNS;
            TensorLayout conv_src({24, 3, 240, 128}, dtype::Float32());
            TensorLayout conv_weight({1, 3, 1, 1}, dtype::Float32());
            auto conv = benchmarker_conv.execs({conv_src, conv_weight, {}}) / RUNS;

            printf("case 2: reduce use time %fms, convolution use time %fms\n", reduce,
                   conv);
        }
        {
            TensorLayout src({24, 240, 128, 4}, dtype::Float32());
            auto reduce = benchmarker_reduce.execs({src, {}}) / RUNS;
            TensorLayout conv_src({24, 4, 240, 128}, dtype::Float32());
            TensorLayout conv_weight({1, 4, 1, 1}, dtype::Float32());
            auto conv = benchmarker_conv.execs({conv_src, conv_weight, {}}) / RUNS;

            printf("case 3: reduce use time %fms, convolution use time %fms\n", reduce,
                   conv);
        }
    };
    run();
}

TEST_F(FALLBACK, BENCHMARK_REDUCE) {
    auto run = [&]() {
        Benchmarker<Reduce> benchmarker_reduce(handle());
        benchmarker_reduce.set_display(false);
        using Mode = param::Reduce::Mode;

        constexpr size_t RUNS = 100;
        benchmarker_reduce.set_times(RUNS);

        TensorShape small{3 * 224 * 224};
        TensorShape large{3 * 224 * 224 * 100};
        param::Reduce param;
        param.axis = 0;

        for (auto i = 224; i < 224 * 2; i++) {
            for (auto mode : {Mode::SUM, Mode::MEAN, Mode::SUM_SQR}) {
                param.mode = mode;
                benchmarker_reduce.set_param(param);
                auto reduce = benchmarker_reduce.execs({{3 * 224 * i}, {}}) / RUNS;
            }
        }
        param.mode = param::Reduce::Mode::SUM;
        benchmarker_reduce.set_param(param);
        printf("SUM\n");
        {
            TensorLayout src(small, dtype::Float32());
            auto reduce = benchmarker_reduce.execs({src, {}}) / RUNS;

            printf("case 1: reduce use time %fms\n", reduce);
        }
        {
            TensorLayout src(large, dtype::Float32());
            auto reduce = benchmarker_reduce.execs({src, {}}) / RUNS;

            printf("case 1: reduce use time %fms\n", reduce);
        }

        param.mode = param::Reduce::Mode::MEAN;
        benchmarker_reduce.set_param(param);
        printf("MEAN\n");
        {
            TensorLayout src(small, dtype::Float32());
            auto reduce = benchmarker_reduce.execs({src, {}}) / RUNS;

            printf("case 2: reduce use time %fms\n", reduce);
        }
        {
            TensorLayout src(large, dtype::Float32());
            auto reduce = benchmarker_reduce.execs({src, {}}) / RUNS;

            printf("case 2: reduce use time %fms\n", reduce);
        }

        param.mode = param::Reduce::Mode::SUM_SQR;
        benchmarker_reduce.set_param(param);
        printf("SUM_SQR\n");
        {
            TensorLayout src(small, dtype::Float32());
            auto reduce = benchmarker_reduce.execs({src, {}}) / RUNS;

            printf("case 3: reduce use time %fms\n", reduce);
        }
        {
            TensorLayout src(large, dtype::Float32());
            auto reduce = benchmarker_reduce.execs({src, {}}) / RUNS;

            printf("case 3: reduce use time %fms\n", reduce);
        }
    };
    run();
}
#endif

// vim: syntax=cpp.doxygen
