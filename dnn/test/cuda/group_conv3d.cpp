/**
 * \file dnn/test/cuda/group_conv3d.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs/nn.h"

#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/convolution3d.h"
#include "test/cuda/fixture.h"

#include "src/cuda/utils.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, GROUP_CONVOLUTION3D_FORWARD) {
    bool is_int_available = cuda::is_compute_capability_required(6, 1);
    static_cast<void>(is_int_available);
    auto run = [&](size_t N, size_t IC, size_t ID, size_t IH, size_t IW,
                   size_t FD, size_t FH, size_t FW, size_t OC, size_t PD,
                   size_t PH, size_t PW, size_t SD, size_t SH, size_t SW,
                   size_t DD, size_t DH, size_t DW, size_t group) {
        {
            // float case
            Checker<Convolution3D> checker(handle_cuda());
            Convolution3D::Param param;
            param.sparse = Convolution3D::Param::Sparse::GROUP;
            param.pad_d = PD;
            param.pad_h = PH;
            param.pad_w = PW;
            param.stride_d = SD;
            param.stride_h = SH;
            param.stride_w = SW;
            param.dilate_d = DD;
            param.dilate_h = DH;
            param.dilate_w = DW;
            auto ICpg = IC / group;
            auto OCpg = OC / group;
            checker.set_param(param).exec(
                    {{N, IC, ID, IH, IW}, {group, OCpg, ICpg, FD, FH, FW}, {}});
        }
    };
    // normal case
    run(2, 64, 7, 7, 7, 1, 1, 1, 32, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2);
    run(1, 2, 2, 2, 2, 1, 1, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2);
    run(2, 64, 7, 7, 7, 3, 3, 3, 32, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2);
    // padded case
    run(2, 32, 7, 7, 7, 3, 3, 3, 64, 2, 2, 2, 1, 1, 1, 1, 1, 1, 4);
    // strided case
    run(2, 32, 7, 7, 7, 3, 3, 3, 64, 0, 0, 0, 2, 2, 2, 1, 1, 1, 8);
    // dilated case
#if CUDNN_MAJOR >= 6
    run(10, 4, 64, 64, 12, 3, 2, 2, 64, 0, 0, 0, 1, 1, 1, 3, 4, 2, 4);
#else
#endif
}

TEST_F(CUDA, GROUP_CONVOLUTION3D_FORWARD_1x1x1) {
    auto run = [&](size_t N, size_t IC, size_t ID, size_t IH, size_t IW,
                   size_t FD, size_t FH, size_t FW, size_t OC, size_t group) {
        Checker<Convolution3D> checker(handle_cuda());
#if CUDNN_MAJOR <= 6
        bool require_algo = true;
        checker.set_before_exec_callback(
                AlgoChecker<Convolution3DForward>{
                        "group_conv3d:1x1x1", &require_algo});
#endif
        Convolution3D::Param param;
        param.sparse = Convolution3D::Param::Sparse::GROUP;
        auto ICg = IC / group;
        auto OCg = OC / group;
        checker.set_param(param).exec(
                {{N, IC, ID, IH, IW}, {group, OCg, ICg, FD, FH, FW}, {}});
    };
    size_t ic = 192;
    for (size_t g = 2; g <= 4; g += 1) {
        for (size_t id = 4; id <= 16; id *= 2) {
            size_t iw = id, ih = id;
            run(2, ic, id, ih, iw, 1, 1, 1, ic / g, g);
            run(2, ic, id + 1, ih + 1, iw + 1, 1, 1, 1, ic / g, g);
        }
    }
}

TEST_F(CUDA, GROUP_CONVOLUTION3D_BACKWARD_DATA) {
    auto run = [&](size_t N, size_t IC, size_t ID, size_t IH, size_t IW,
                   size_t FD, size_t FH, size_t FW, size_t OC, size_t OD,
                   size_t OH, size_t OW, size_t PD, size_t PH, size_t PW,
                   size_t SD, size_t SH, size_t SW, size_t group) {
        Checker<Convolution3DBackwardData> checker(handle_cuda());
        Convolution3DBackwardData::Param param;
        param.sparse = Convolution3D::Param::Sparse::GROUP;
        param.pad_d = PD;
        param.pad_h = PH;
        param.pad_w = PW;
        param.stride_d = SD;
        param.stride_h = SH;
        param.stride_w = SW;
        auto ICg = IC / group;
        auto OCg = OC / group;
        checker.set_param(param).exec({{group, OCg, ICg, FD, FH, FW},
                                       {N, OC, OD, OH, OW},
                                       {N, IC, ID, IH, IW}});
    };    
    // bug case in prev ver 
    
    run(1, 2, 1, 1, 1,  1, 1, 1, 2,  1, 1, 3, 0, 0, 1, 1, 1, 1, 2); 
    run(1, 2, 1, 1, 1,  1, 1, 1, 2,  1, 1, 2, 0, 0, 1, 1, 1, 2, 2); 
    run(1, 2, 1, 1, 1,  1, 1, 1, 2,  1, 2, 1, 0, 1, 0, 1, 2, 1, 2); 
    run(1, 2, 1, 1, 1,  1, 1, 1, 2,  2, 1, 1, 1, 0, 0, 2, 1, 1, 2); 
    // normal case
    run(2, 64, 7, 7, 7, 3, 3, 3, 32, 5, 5, 5, 0, 0, 0, 1, 1, 1, 2);
    // padded case
    run(2, 32, 7, 7, 7, 3, 3, 3, 64, 7, 7, 7, 1, 1, 1, 1, 1, 1, 4);
    // strided case
    run(2, 32, 7, 7, 7, 3, 3, 3, 64, 3, 3, 3, 0, 0, 0, 2, 2, 2, 8);
    // bigger case
    run(2, 32, 64, 64, 64, 3, 3, 3, 32, 62, 62, 62, 0, 0, 0, 1, 1, 1, 4);
}

TEST_F(CUDA, GROUP_CONVOLUTION3D_BACKWARD_FILTER) {
    auto run = [&](size_t N, size_t IC, size_t ID, size_t IH, size_t IW,
                   size_t FD, size_t FH, size_t FW, size_t OC, size_t OD,
                   size_t OH, size_t OW, size_t PD, size_t PH, size_t PW,
                   size_t SD, size_t SH, size_t SW, size_t group) {
        Checker<Convolution3DBackwardFilter> checker(handle_cuda());
        Convolution3DBackwardFilter::Param param;
        param.sparse = Convolution3D::Param::Sparse::GROUP;
        param.pad_d = PD;
        param.pad_h = PH;
        param.pad_w = PW;
        param.stride_d = SD;
        param.stride_h = SH;
        param.stride_w = SW;
        auto ICg = IC / group;
        auto OCg = OC / group;
        checker.set_param(param).exec({{N, IC, ID, IH, IW},
                                       {N, OC, OD, OH, OW},
                                       {group, OCg, ICg, FD, FH, FW}});
    };
    // normal case
    run(2, 64, 7, 7, 7, 3, 3, 3, 32, 5, 5, 5, 0, 0, 0, 1, 1, 1, 2);
    // padded case
    run(2, 32, 7, 7, 7, 3, 3, 3, 64, 7, 7, 7, 1, 1, 1, 1, 1, 1, 4);
    // strided case
    run(2, 32, 7, 7, 7, 3, 3, 3, 64, 3, 3, 3, 0, 0, 0, 2, 2, 2, 8);
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
