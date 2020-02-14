/**
 * \file dnn/test/common/conv_pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/conv_pooling.h"

namespace megdnn {
namespace test {
namespace conv_pooling {

/* ConvPooling(
    Method method_=Method::WITH_TEXTURE_OBJ,
    ConvMode convMode_=ConvMode::CROSS_CORRELATION,
    PoolMode poolMode_=PoolMode::AVERAGE,
    NonlineMode nonlineMode_=NonlineMode::IDENTITY,
    uint32_t pool_shape_h_=1,
    uint32_t pool_shape_w_=1,
    uint32_t pool_stride_h_=1,
    uint32_t pool_stride_w_=1,
    uint32_t pool_pad_h_=0,
    uint32_t pool_pad_w_=0,
    uint32_t conv_stride_h_=1,
    uint32_t conv_stride_w_=1,
    uint32_t conv_pad_h_=0,
    uint32_t conv_pad_w_=0,
    float *bias_=NULL)
*/

std::vector<TestArg> get_args()
{
    std::vector<TestArg> args;
    uint32_t pool_shape_h = 3;
    uint32_t pool_shape_w = 3;
    uint32_t pool_stride_h = pool_shape_h;
    uint32_t pool_stride_w = pool_shape_w;

    param::ConvPooling cur_param(
        param::ConvPooling::Method::WITH_TEXTURE_OBJ,
        param::ConvPooling::ConvMode::CONVOLUTION,
        param::ConvPooling::PoolMode::MAX,
        param::ConvPooling::NonlineMode::RELU,
        pool_shape_h, pool_shape_w,
        pool_stride_h, pool_stride_w,
        0, 0, 1, 1, 0, 0
    );
    std::vector<param::ConvPooling::ConvMode> conv_mode;
    conv_mode.push_back(param::ConvPooling::ConvMode::CONVOLUTION);
    conv_mode.push_back(param::ConvPooling::ConvMode::CROSS_CORRELATION);

    std::vector<param::ConvPooling::NonlineMode> nonline_mode;
    nonline_mode.push_back(param::ConvPooling::NonlineMode::IDENTITY);
    nonline_mode.push_back(param::ConvPooling::NonlineMode::SIGMOID);
    nonline_mode.push_back(param::ConvPooling::NonlineMode::RELU);

    for (size_t i = 19; i < 21; ++i) {
        for(size_t i_nl_mode = 0; i_nl_mode < nonline_mode.size(); ++ i_nl_mode) {
            cur_param.nonlineMode = nonline_mode[i_nl_mode];
            for (size_t i_conv_mode = 0; i_conv_mode < conv_mode.size(); ++ i_conv_mode) {
                for(size_t kernel_size = 1; kernel_size < 7; ++ kernel_size) {
                    for(size_t pool_size = 1; pool_size < 5; ++ pool_size) {
                        if (pool_size >= kernel_size)
                            continue;
                        cur_param.convMode = conv_mode[i_conv_mode];
                        args.emplace_back(cur_param,
                        TensorShape{20, 4, i, i},
                        TensorShape{3, 4, 4, 4},
                        TensorShape{1, 3, 1, 1});
                    }
                
                }
            }
        }
    }
/*
    // large channel
    for (size_t i = 20; i < 22; ++i) {
        cur_param.convMode = param::ConvPooling::ConvMode::CONVOLUTION;
        args.emplace_back(cur_param,
                TensorShape{2, 20, i, i+1},
                TensorShape{30, 20, 4, 4},
                TensorShape{1, 30, 1, 1});

        cur_param.convMode = param::ConvPooling::ConvMode::CROSS_CORRELATION;
        args.emplace_back(cur_param,
                TensorShape{2, 20, i, i+1},
                TensorShape{30, 20, 3, 3},
                TensorShape{1, 30, 1, 1});
    }

    // large filter
    for (size_t i = 20; i < 22; ++i) {
        cur_param.convMode = param::ConvPooling::ConvMode::CONVOLUTION;
        args.emplace_back(cur_param,
                TensorShape{2, 2, i, i+1},
                TensorShape{3, 2, 5, 5},
                TensorShape{1, 3, 1, 1});

        cur_param.convMode = param::ConvPooling::ConvMode::CROSS_CORRELATION;
                cur_param.convMode = param::ConvPooling::ConvMode::CROSS_CORRELATION;
        args.emplace_back(cur_param,
                TensorShape{2, 2, i, i+1},
                TensorShape{3, 2, 5, 5},
                TensorShape{1, 3, 1, 1});
    }
*/

    return args;
}

} // namespace conv_pooling
} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen