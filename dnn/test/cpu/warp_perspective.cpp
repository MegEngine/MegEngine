/**
 * \file dnn/test/cpu/warp_perspective.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cpu/fixture.h"

#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/random_state.h"

#include "test/common/warp_perspective.h"

namespace megdnn {
namespace test {

TEST_F(CPU, WARP_PERSPECTIVE_CV) {
    //! Just for the format NHWC
    Checker<WarpPerspective> checker(handle());
    param::WarpPerspective param;
    class ResizeMatRNG : public RNG {
        void gen(const TensorND& tensor_) override {
            auto& gen = RandomState::generator();
            std::uniform_real_distribution<dt_float32> pdist3(1.9f, 3.1f);
            std::uniform_real_distribution<dt_float32> pdist(0.9f, 1.1f);
            std::uniform_real_distribution<dt_float32> pdisth(0.4f, 0.6f);
            std::uniform_real_distribution<dt_float32> ndist(-1.1f, -0.9f);
            std::uniform_real_distribution<dt_float32> ndist3(-3.1f, -1.9f);
            std::uniform_real_distribution<dt_float32> ndisth(-0.6f, -0.4f);
            std::uniform_int_distribution<int> dice(0, 5);
            float* ptr = tensor_.ptr<dt_float32>();
            auto N = tensor_.layout.shape[0];
            for (size_t n = 0; n < N; ++n) {
                for (size_t i = 0; i < 9; ++i) {
                    switch (dice(gen)) {
                        case 0:
                            ptr[i] = pdist3(gen);
                            break;
                        case 1:
                            ptr[i] = pdist(gen);
                            break;
                        case 2:
                            ptr[i] = pdisth(gen);
                            break;
                        case 3:
                            ptr[i] = ndist(gen);
                            break;
                        case 4:
                            ptr[i] = ndist3(gen);
                            break;
                        case 5:
                            ptr[i] = ndisth(gen);
                            break;
                    }
                }
                // is resize?
                if (n & 1) {
                    ptr[1] = 0;
                    ptr[3] = 0;
                    ptr[6] = ptr[7] = 0;
                }
                ptr += 9;
            }
        }
    } rng;
    checker.set_rng(1, &rng);
    using BMode = param::WarpPerspective::BorderMode;
    param.format = param::WarpPerspective::Format::NHWC;
    for (auto mode : {BMode::REFLECT_101, BMode::REPLICATE, BMode::REFLECT,
                      BMode::WRAP, BMode::CONSTANT}) {
        param.bmode = mode;
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{1000, 2, 10, 3}, {1000, 3, 3}, {1000, 2, 12, 3}});
    }
    // resize nan case
    UniformFloatRNG rng_zero(0, 0);
    checker.set_rng(1, &rng_zero);
    {
        param.bmode = BMode::CONSTANT;
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{1000, 2, 10, 3}, {1000, 3, 3}, {1000, 2, 12, 3}});
    }

    auto args = warp_perspective::get_cv_args();
    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_epsilon(1 + 1e-3)
                .set_dtype(0, dtype::Uint8())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Uint8())
                .execs({arg.src, arg.trans, arg.dst});
    }
    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .execs({arg.src, arg.trans, arg.dst});
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
