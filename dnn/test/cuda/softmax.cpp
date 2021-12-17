/**
 * \file dnn/test/cuda/softmax.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "megdnn/tensor_iter.h"
#include "test/common/checker.h"
#include "test/common/softmax.h"

#include "src/common/utils.h"
#include "test/cuda/utils.h"

// to check cudnn version
#include <cudnn.h>
#include "test/cuda/benchmark.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, SOFTMAX_FORWARD) {
    auto args = softmax::get_args();
    std::vector<DType> dtypes{dtype::Float16(), dtype::Float32()};

    for (auto dtype : dtypes)
        for (auto&& arg : args) {
            auto param = arg.param;
            auto src = arg.ishape;
            Checker<Softmax> checker(handle_cuda());
            if (dtype == dtype::BFloat16()) {
                checker.set_epsilon(2e-2);
            } else {
                checker.set_epsilon(1e-2);
            }
            checker.set_param(param).set_dtype(0, dtype).set_dtype(1, dtype).exec(
                    TensorShapeArray{src, {}});
        }
}

TEST_F(CUDA, SOFTMAX_BACKWARD) {
    auto args = softmax::get_args();
    for (auto&& arg : args) {
        Checker<SoftmaxBackward> checker(handle_cuda());
        TensorLayout ilayout = TensorLayout(arg.ishape, dtype::Float32());
        TensorLayout olayout;

        {
            auto opr = handle_cuda()->create_operator<SoftmaxForward>();
            opr->param() = arg.param;
            opr->deduce_layout(ilayout, olayout);
        }
        auto set_dtype = [&checker](DType dtype) {
            checker.set_dtype(0, dtype).set_dtype(1, dtype).set_dtype(2, dtype);
        };

        set_dtype(dtype::Float32());
        checker.set_epsilon(1e-3).set_param(arg.param).exec(
                TensorShapeArray{ilayout, olayout, ilayout});
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
