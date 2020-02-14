/**
 * \file dnn/test/common/cvt_color.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace cvt_color {

struct TestArg {
    param::CvtColor param;
    TensorShape src;
    DType dtype;
    TestArg(param::CvtColor param, TensorShape src, DType dtype)
            : param(param), src(src), dtype(dtype) {}
};

inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    using Mode = param::CvtColor::Mode;
    param::CvtColor cur_param;
    for (size_t i = 2; i <= 10; ++i) {
        for (size_t j = 2; j <= 10; ++j) {
            cur_param.mode = Mode::RGB2GRAY;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Uint8());
            cur_param.mode = Mode::RGB2YUV;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Uint8());
            cur_param.mode = Mode::YUV2RGB;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Uint8());
            cur_param.mode = Mode::GRAY2RGB;
            args.emplace_back(cur_param, TensorShape{1, i, j, 1},
                              dtype::Uint8());
            cur_param.mode = Mode::RGBA2RGB;
            args.emplace_back(cur_param, TensorShape{1, i, j, 4},
                              dtype::Uint8());
            cur_param.mode = Mode::RGBA2BGR;
            args.emplace_back(cur_param, TensorShape{1, i, j, 4},
                              dtype::Uint8());
            cur_param.mode = Mode::RGBA2GRAY;
            args.emplace_back(cur_param, TensorShape{1, i, j, 4},
                              dtype::Uint8());
            cur_param.mode = Mode::RGB2BGR;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Uint8());
            cur_param.mode = Mode::BGR2GRAY;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Uint8());
            cur_param.mode = Mode::BGR2RGB;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Uint8());
            // float32 test
            cur_param.mode = Mode::RGB2GRAY;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Float32());
            cur_param.mode = Mode::RGB2YUV;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Float32());
            cur_param.mode = Mode::YUV2RGB;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Float32());
            cur_param.mode = Mode::GRAY2RGB;
            args.emplace_back(cur_param, TensorShape{1, i, j, 1},
                              dtype::Float32());
        }
    }
    for (size_t i = 18; i < 164; i *= 3) {
        for (auto mode : {
                     Mode::YUV2GRAY_NV21,       Mode::YUV2BGR_NV21,
                     Mode::YUV2RGB_NV21,        Mode::YUV2GRAY_NV12,
                     Mode::YUV2BGR_NV12,        Mode::YUV2RGB_NV12,
                     Mode::YUV2GRAY_YV12,       Mode::YUV2BGR_YV12,
                     Mode::YUV2RGB_YV12,        Mode::YUV2GRAY_YU12,
                     Mode::YUV2BGR_YU12,        Mode::YUV2RGB_YU12,

                     Mode::BT601_YUV2GRAY_NV21, Mode::BT601_YUV2BGR_NV21,
                     Mode::BT601_YUV2RGB_NV21,  Mode::BT601_YUV2GRAY_NV12,
                     Mode::BT601_YUV2BGR_NV12,  Mode::BT601_YUV2RGB_NV12,
                     Mode::BT601_YUV2GRAY_YV12, Mode::BT601_YUV2BGR_YV12,
                     Mode::BT601_YUV2RGB_YV12,  Mode::BT601_YUV2GRAY_YU12,
                     Mode::BT601_YUV2BGR_YU12,  Mode::BT601_YUV2RGB_YU12,

             }) {
            cur_param.mode = mode;
            args.emplace_back(cur_param, TensorShape{1, i, i, 1},
                              dtype::Uint8());
        }
    }

    //! test case for nv12(nv21), which height is not even, height % 3 == 0,
    //! height % 2 == 1
    for (auto mode : {
                 Mode::YUV2GRAY_NV21,
                 Mode::YUV2BGR_NV21,
                 Mode::YUV2RGB_NV21,
                 Mode::YUV2GRAY_NV12,
                 Mode::YUV2BGR_NV12,
                 Mode::YUV2RGB_NV12,

                 Mode::BT601_YUV2GRAY_NV21,
                 Mode::BT601_YUV2BGR_NV21,
                 Mode::BT601_YUV2RGB_NV21,
                 Mode::BT601_YUV2GRAY_NV12,
                 Mode::BT601_YUV2BGR_NV12,
                 Mode::BT601_YUV2RGB_NV12,

         }) {
        cur_param.mode = mode;
        args.emplace_back(cur_param, TensorShape{1, 3, 18, 1}, dtype::Uint8());
        args.emplace_back(cur_param, TensorShape{1, 9, 18, 1}, dtype::Uint8());
    }

    return args;
}

inline std::vector<TestArg> get_cuda_args() {
    std::vector<TestArg> args;
    using Mode = param::CvtColor::Mode;
    param::CvtColor cur_param;

    for (size_t i = 2; i <= 10; ++i) {
        for (size_t j = 2; j <= 10; ++j) {
            cur_param.mode = Mode::RGB2GRAY;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Uint8());
            cur_param.mode = Mode::RGB2YUV;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Uint8());
            cur_param.mode = Mode::YUV2RGB;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Uint8());
            cur_param.mode = Mode::GRAY2RGB;
            args.emplace_back(cur_param, TensorShape{1, i, j, 1},
                              dtype::Uint8());
            // float32 test
            cur_param.mode = Mode::RGB2GRAY;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Float32());
            cur_param.mode = Mode::RGB2YUV;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Float32());
            cur_param.mode = Mode::YUV2RGB;
            args.emplace_back(cur_param, TensorShape{1, i, j, 3},
                              dtype::Float32());
            cur_param.mode = Mode::GRAY2RGB;
            args.emplace_back(cur_param, TensorShape{1, i, j, 1},
                              dtype::Float32());
            args.emplace_back(cur_param, TensorShape{3, i, j, 1},
                              dtype::Float32());
        }
    }

    for (size_t i = 18; i < 164; i *= 3) {
        for (auto mode : {
                     Mode::YUV2GRAY_NV21,
                     Mode::YUV2BGR_NV21,
                     Mode::YUV2RGB_NV21,
                     Mode::YUV2GRAY_NV12,
                     Mode::YUV2BGR_NV12,
                     Mode::YUV2RGB_NV12,
                     Mode::YUV2GRAY_YV12,
                     Mode::YUV2BGR_YV12,
                     Mode::YUV2RGB_YV12,
                     Mode::YUV2GRAY_YU12,
                     Mode::YUV2BGR_YU12,
                     Mode::YUV2RGB_YU12,
             }) {
            cur_param.mode = mode;
            args.emplace_back(cur_param, TensorShape{1, i, i, 1},
                              dtype::Uint8());
        }
    }

    return args;
}

}  // namespace cvt_color
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
