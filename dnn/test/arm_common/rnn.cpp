/**
 * \file dnn/test/arm_common/rnn.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/arm_common/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/task_record_check.h"

using namespace megdnn;
using namespace test;

TEST_F(ARM_COMMON, RNNCell) {
    Checker<RNNCell> checker(handle());
    using NonlineMode = param::RNNCell::NonlineMode;
    param::RNNCell param;
    for (auto mode : {NonlineMode::IDENTITY, NonlineMode::RELU, NonlineMode::TANH})
        for (size_t batch : {1, 4})
            for (size_t n : {3, 4, 5, 23, 100})
                for (size_t h : {5, 23, 100})
                    for (size_t out : {3, 6, 25, 100}) {
                        param.nonlineMode = mode;
                        checker.set_param(param);
                        checker.exec(
                                {{batch, n},
                                 {out, n},
                                 {1, out},
                                 {batch, h},
                                 {out, h},
                                 {1, out},
                                 {}});
                        checker.exec(
                                {{batch, n},
                                 {out, n},
                                 {batch, out},
                                 {batch, h},
                                 {out, h},
                                 {batch, out},
                                 {}});
                    }
}

TEST_F(ARM_COMMON, RNNCellRecord) {
    TaskRecordChecker<RNNCell> checker(0);
    using NonlineMode = param::RNNCell::NonlineMode;
    param::RNNCell param;
    for (auto mode : {NonlineMode::IDENTITY, NonlineMode::RELU, NonlineMode::TANH}) {
        param.nonlineMode = mode;
        checker.set_param(param);
        checker.exec({{1, 100}, {10, 100}, {1, 10}, {1, 100}, {10, 100}, {1, 10}, {}});
        checker.exec({{1, 34}, {15, 34}, {1, 15}, {1, 34}, {15, 34}, {1, 15}, {}});
        checker.exec({{1, 73}, {25, 73}, {1, 25}, {1, 73}, {25, 73}, {1, 25}, {}});
    }
}

#if MEGDNN_WITH_BENCHMARK

#endif
// vim: syntax=cpp.doxygen
