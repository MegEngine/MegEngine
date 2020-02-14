/**
 * \file dnn/test/common/warp_affine.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/opr_param_defs.h"
#include "megdnn/basic_types.h"
#include "test/common/opr_proxy.h"

#include <iostream>

namespace megdnn {
namespace test {
namespace warp_affine {

struct TestArg {
    param::WarpAffine param;
    TensorShape src;
    TensorShape trans;
    TensorShape dst;
    TestArg(param::WarpAffine param_, TensorShape src_, TensorShape trans_,
            TensorShape dst_)
            : param(param_), src(src_), trans(trans_), dst(dst_) {}
};

inline std::vector<TestArg> get_cv_args() {
    std::vector<TestArg> args;

    //! if the format of WarpAffine is NHWC, not support LINEAR and AREA
    using BorderMode = param::WarpAffine::BorderMode;
    using InterpolationMode = param::WarpAffine::InterpolationMode;
    param::WarpAffine cur_param;
    cur_param.format = param::WarpAffine::Format::NHWC;

    for (size_t i = 4; i <= 168; i *= 8) {
        for (size_t ic : {1, 2, 3}) {
            for (BorderMode bmode :
                 {BorderMode::BORDER_REPLICATE, BorderMode::BORDER_REFLECT,
                  BorderMode::BORDER_REFLECT_101, BorderMode::BORDER_WRAP,
                  BorderMode::BORDER_CONSTANT}) {
                for (InterpolationMode imode :
                     {InterpolationMode::LINEAR,
                      InterpolationMode::INTER_NEAREST,
                      InterpolationMode::INTER_CUBIC,
                      InterpolationMode::INTER_LANCZOS4}) {
                    cur_param.border_mode = bmode;
                    cur_param.border_val = 1.1f;

                    cur_param.imode = imode;
                    args.emplace_back(cur_param, TensorShape{1, i, i, ic},
                                      TensorShape{1, 2, 3},
                                      TensorShape{1, i, i, ic});
                    args.emplace_back(cur_param, TensorShape{1, i, i * 2, ic},
                                      TensorShape{1, 2, 3},
                                      TensorShape{1, i, i * 2, ic});
                    args.emplace_back(cur_param, TensorShape{1, i * 3, i, ic},
                                      TensorShape{1, 2, 3},
                                      TensorShape{1, i * 3, i, ic});

                    args.emplace_back(cur_param, TensorShape{1, i, i, ic},
                                      TensorShape{1, 2, 3},
                                      TensorShape{1, 8, 8, ic});
                    args.emplace_back(cur_param, TensorShape{1, i, i * 2, ic},
                                      TensorShape{1, 2, 3},
                                      TensorShape{1, 8, 8, ic});
                    args.emplace_back(cur_param, TensorShape{1, i * 3, i, ic},
                                      TensorShape{1, 2, 3},
                                      TensorShape{1, 8, 8, ic});
                }
            }
        }
    }
    return args;
}

}  // namespace warp_affine
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
