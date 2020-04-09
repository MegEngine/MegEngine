/**
 * \file dnn/test/fallback/reduce.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/fallback/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

using namespace megdnn;
using namespace test;

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
    for (auto mode : {Mode::SUM, Mode::MEAN, Mode::SUM_SQR, Mode::PRODUCT,
                      Mode::MIN, Mode::MAX})
        for (auto dtype : std::vector<DType>{dtype::Float16(), dtype::Float32(),
                                             dtype::Int32(), dtype::Int16(),
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

// vim: syntax=cpp.doxygen
