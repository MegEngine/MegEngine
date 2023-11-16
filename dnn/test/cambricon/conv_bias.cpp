/**
 * \file dnn/test/cambricon/conv_bias.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 */

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"

#include "test/cambricon/fixture.h"
#include "test/common/checker.h"
#include "test/common/conv_bias.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

#include "src/cambricon/utils.h"

namespace megdnn {
namespace test {

TEST_F(CAMBRICON, CONVBIAS_FORWARD_FLOAT) {
    using namespace conv_bias;
    Checker<ConvBiasForward> checker(handle_cambricon());
    using NLMode = param::ConvBias::NonlineMode;
    param::ConvBias cur_param;
    cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
    cur_param.format = param::ConvBias::Format::NHWC;
    for (auto nonlineMode : {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID}) {
        for (size_t n : {1, 2}) {
            for (size_t ic : {16, 64}) {
                for (size_t oc : {32, 256}) {
                    for (size_t i : {56}) {
                        for (size_t pad : {0, 1}) {
                            for (size_t stride : {1, 2}) {
                                for (size_t k : {3, 7}) {
                                    cur_param.nonlineMode = nonlineMode;
                                    cur_param.pad_h = pad;
                                    cur_param.pad_w = pad;
                                    cur_param.stride_h = stride;
                                    cur_param.stride_w = stride;
                                    checker.set_dtype(0, dtype::Float32())
                                            .set_dtype(1, dtype::Float32())
                                            .set_dtype(2, dtype::Float32())
                                            .set_dtype(3, dtype::Float32())
                                            .set_dtype(4, dtype::Float32());
                                    checker.set_param(cur_param);
                                    size_t j = (i - k + 2 * pad) / stride + 1;
                                    //! BROADCAST
                                    checker.execs(
                                            {{n, i, i, ic},
                                             {oc, k, k, ic},
                                             {1, 1, 1, oc},
                                             {},
                                             {n, j, j, oc}});
                                    // todo: wait elemwise
                                    //! z
                                    checker.execs(
                                            {{n, i, i, ic},
                                             {oc, k, k, ic},
                                             {1, 1, 1, oc},
                                             {n, j, j, oc},
                                             {n, j, j, oc}});
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_F(CAMBRICON, CONVBIAS_FORWARD_1X1) {
    using namespace conv_bias;
    Checker<ConvBiasForward> checker(handle_cambricon());
    using NLMode = param::ConvBias::NonlineMode;
    param::ConvBias cur_param;
    cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
    cur_param.format = param::ConvBias::Format::NHWC;
    // wait elemwise
    for (auto nonlineMode : {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID}) {
        for (size_t n : {1, 2}) {
            for (size_t ic : {8, 16, 64, 128}) {
                for (size_t oc : {32, 256}) {
                    cur_param.nonlineMode = nonlineMode;
                    checker.set_dtype(0, dtype::Float32())
                            .set_dtype(1, dtype::Float32())
                            .set_dtype(2, dtype::Float32())
                            .set_dtype(3, dtype::Float32())
                            .set_dtype(4, dtype::Float32());
                    checker.set_param(cur_param);
                    //! bias
                    checker.execs(
                            {{n, 28, 28, ic},
                             {oc, 1, 1, ic},
                             {1, 1, 1, oc},
                             {},
                             {n, 28, 28, oc}});
                    // wait elemwise
                    //! z
                    checker.execs(
                            {{n, 28, 28, ic},
                             {oc, 1, 1, ic},
                             {1, 1, 1, oc},
                             {n, 28, 28, oc},
                             {n, 28, 28, oc}});
                }
            }
        }
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
