/**
 * \file dnn/test/common/roi_pooling.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"

#include "test/common/rng.h"
#include "test/common/random_state.h"

namespace megdnn {
namespace test {

class ROIPoolingRNG final : public IIDRNG {
public:
    ROIPoolingRNG(size_t n) : n(n), idx(0) {}
    dt_float32 gen_single_val() override {
        std::uniform_real_distribution<dt_float32> distf(0.0f, 1.0f);
        std::uniform_int_distribution<int> disti(0, n - 1);
        dt_float32 res;
        if (idx == 0) {
            res = static_cast<dt_float32>(disti(RandomState::generator()));
        }
        if (idx == 1 || idx == 2) {
            res = distf(RandomState::generator()) * 0.5;
        } else {
            res = distf(RandomState::generator()) * 0.5 + 0.5;
        }
        idx = (idx + 1) % 5;
        return res;
    }

private:
    size_t n;
    size_t idx;
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
