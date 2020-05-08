/**
 * \file dnn/test/cuda/group_conv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs/nn.h"

#include "test/cuda/fixture.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/convolution.h"

#include "src/cuda/utils.h"

namespace megdnn {
namespace test {


TEST_F(CUDA, GROUP_CONV_FORWARD)
{
    bool is_int_available = cuda::is_compute_capability_required(6, 1);
    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW,
            size_t FH, size_t FW,
            size_t OC, size_t /* OH */, size_t /* OW */,
            size_t PH, size_t PW,
            size_t SH, size_t SW,
            size_t DH, size_t DW,
            size_t group)
    {
        {
            // float case
            Checker<Convolution> checker(handle_cuda());
            Convolution::Param param;
            param.sparse = Convolution::Param::Sparse::GROUP;
            param.pad_h = PH;
            param.pad_w = PW;
            param.stride_h = SH;
            param.stride_w = SW;
            param.dilate_h = DH;
            param.dilate_w = DW;
            auto ICg = IC / group;
            auto OCg = OC / group;
            checker.set_param(param).exec({{N, IC, IH, IW},
                    {group, OCg, ICg, FH, FW}, {}});
        }
        if (is_int_available) {
            // int 8x8x32 case
            Checker<Convolution> checker(handle_cuda());
            Convolution::Param param;
            param.sparse = Convolution::Param::Sparse::GROUP;
            param.format = Convolution::Param::Format::NHWC;
            param.pad_h = PH;
            param.pad_w = PW;
            param.stride_h = SH;
            param.stride_w = SW;
            param.dilate_h = DH;
            param.dilate_w = DW;
            auto ICg = IC / group;
            auto OCg = OC / group;
            UniformIntRNG rng(-4, 4);
            checker.set_param(param).
                set_dtype(0, dtype::Int8()).
                set_dtype(1, dtype::Int8()).
                set_dtype(2, dtype::Int32()).
                set_rng(0, &rng).
                set_rng(1, &rng).
                exec({{N, IH, IW, IC}, {group, OCg, FH, FW, ICg}, {}});
        }
    };
    // normal case
    run(2, 64, 7, 7,
            3, 3,
            32, 5, 5,
            0, 0,
            1, 1,
            1, 1,
            2);
    // padded case
    run(2, 32, 7, 7,
            3, 3,
            64, 7, 7,
            1, 1,
            1, 1,
            1, 1,
            4);
    // strided case
    run(2, 32, 7, 7,
            3, 3,
            64, 3, 3,
            0, 0,
            2, 2,
            1, 1,
            8);
    // dilated case
    run(2, 32, 7, 7,
            3, 3,
            64, 3, 3,
            0, 0,
            1, 1,
            2, 2,
            8);

}

TEST_F(CUDA, GROUP_CONV_FORWARD_1x1) {
    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW,
            size_t FH, size_t FW,
            size_t OC, size_t group) {
        Checker<Convolution> checker(handle_cuda());
#if CUDNN_MAJOR <= 6
        std::string conv1x1_name =
                ConvBiasForward::algo_name<ConvBiasForward::MatmulParam>(
                        "MATMUL1X1", {});
        checker.set_before_exec_callback(AlgoChecker<Convolution>(
                ConvBiasForward::algo_name<ConvBiasForward::DirectParam>(
                        ssprintf("%s:%s", "CUDA:GROUP_CONV",
                                 conv1x1_name.c_str()),
                        {})
                        .c_str()));
#endif
        Convolution::Param param;
        param.sparse = Convolution::Param::Sparse::GROUP;
        auto ICg = IC / group;
        auto OCg = OC / group;
        checker.set_param(param).exec({{N, IC, IH, IW},
                {group, OCg, ICg, FH, FW}, {}});
    };
    size_t ic = 192;
    for (size_t g = 2; g <= 3; g += 1) {
        for (size_t ih = 8; ih <= 128; ih *= 4) {
            size_t iw = ih;
            run(2, ic, ih, iw, 1, 1, ic / g, g);
            run(2, ic, ih+1, iw+1, 1, 1, ic / g, g);
        }
    }
}

TEST_F(CUDA, GROUP_CONV_BACKWARD_DATA)
{
    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW,
            size_t FH, size_t FW,
            size_t OC, size_t OH, size_t OW,
            size_t PH, size_t PW,
            size_t SH, size_t SW,
            size_t group)
    {
        Checker<ConvolutionBackwardData> checker(handle_cuda());
        ConvolutionBackwardData::Param param;
        param.sparse = Convolution::Param::Sparse::GROUP;
        param.pad_h = PH;
        param.pad_w = PW;
        param.stride_h = SH;
        param.stride_w = SW;
        auto ICg = IC / group;
        auto OCg = OC / group;
        checker.set_param(param).exec({{group, OCg, ICg, FH, FW},
                {N, OC, OH, OW}, {N, IC, IH, IW}});
    };
    // normal case
    run(2, 64, 7, 7,
            3, 3,
            32, 5, 5,
            0, 0,
            1, 1,
            2);
    // padded case
    run(2, 32, 7, 7,
            3, 3,
            64, 7, 7,
            1, 1,
            1, 1,
            4);
    // strided case
    run(2, 32, 7, 7,
            3, 3,
            64, 3, 3,
            0, 0,
            2, 2,
            8);
}

TEST_F(CUDA, GROUP_CONV_BACKWARD_FILTER)
{
    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW,
            size_t FH, size_t FW,
            size_t OC, size_t OH, size_t OW,
            size_t PH, size_t PW,
            size_t SH, size_t SW,
            size_t group)
    {
        Checker<ConvolutionBackwardFilter> checker(handle_cuda());
        ConvolutionBackwardFilter::Param param;
        param.sparse = Convolution::Param::Sparse::GROUP;
        param.pad_h = PH;
        param.pad_w = PW;
        param.stride_h = SH;
        param.stride_w = SW;
        auto ICg = IC / group;
        auto OCg = OC / group;
        checker.set_param(param).exec({{N, IC, IH, IW},
                {N, OC, OH, OW}, {group, OCg, ICg, FH, FW}});
    };
    // normal case
    run(2, 64, 7, 7,
            3, 3,
            32, 5, 5,
            0, 0,
            1, 1,
            2);
    // padded case
    run(2, 32, 7, 7,
            3, 3,
            64, 7, 7,
            1, 1,
            1, 1,
            4);
    // strided case
    run(2, 32, 7, 7,
            3, 3,
            64, 3, 3,
            0, 0,
            2, 2,
            8);
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
