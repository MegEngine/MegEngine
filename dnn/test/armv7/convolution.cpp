/**
 * \file dnn/test/armv7/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/armv7/fixture.h"

#include "test/common/convolution.h"
#include "test/common/checker.h"
#include "test/common/benchmarker.h"

#include "test/common/rng.h"

using namespace megdnn;
using namespace test;

#if MEGDNN_WITH_BENCHMARK
TEST_F(ARMV7, BENCHMARK_CONVOLUTION_STRIDE2)
{
    using Param = param::Convolution;
    auto run = [&](const TensorShapeArray& shapes, Param param) {
        Benchmarker<Convolution> benchmarker_float(handle());
        size_t RUN = 100;
        auto tfloat = benchmarker_float.set_display(false)
                              .set_times(RUN)
                              .set_param(param)
                              .exec(shapes);
        size_t IC = shapes[1][1];
        size_t FH = shapes[1][2];
        size_t FW = shapes[1][3];
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::Float32()},
                           {shapes[1], dtype::Float32()}, dst_layout);
        printf("flops: %.3f mflops\n",
               (IC * dst_layout.total_nr_elems() * FH * FW * 2) /
                       (tfloat / RUN * 1000));
    };

    auto profile = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                       size_t stride) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = kernel / 2;
        param.pad_w = kernel / 2;
        printf("oc: %zd ic: %zd w: %zd h: %zd stride: %zd kernel_size: %zd\n",
               oc, ic, w, h, stride, kernel);

        run({{1, ic, h, w}, {oc, ic, kernel, kernel}, {}},
            param);

    };

    for (size_t kernel : {2, 3, 5, 7}) {
        for (size_t ic : {3, 6, 12, 24}) {
            for (size_t oc : {3, 6, 12, 24}) {
                for (size_t size : {4, 7, 8, 14, 16, 17, 28, 32, 34, 64, 112}) {
                    profile(oc, ic, size, size, kernel, 2);
                }
            }
        }
    }
}
#endif

TEST_F(ARMV7, BENCHMARK_CONVOLUTION_1X1)
{
    int exec_times = 50;
    Benchmarker<MatrixMul> benchmarker_gemm(handle());
    benchmarker_gemm.set_times(exec_times);

    Benchmarker<Convolution> benchmarker(handle());
    benchmarker.set_times(exec_times);

    float mod = 1000 * exec_times  / 1e9;
    auto run = [&](size_t IC, size_t OC, size_t H, size_t W) {
        float time = 1.f, perf = 1.f;

        std::cout<<std::endl;
        std::cout<< "CONV: IC " << IC << ", OC " << OC <<
                    ", H " << H << ", W " << W <<std::endl;
        time = benchmarker.exec({{1, IC, H, W}, {OC, IC, 1, 1}, {1, OC, H, W}});
        perf = OC * (2 * H * W - 1) * IC / time * mod;
        std::cout<<"Performance is " << perf <<" Gflops" <<std::endl;

        std::cout<<"GEMM: (" << OC <<", "<< H*W << ", " <<IC <<")"<<std::endl;
        //time = benchmarker_gemm.exec({{OC, H*W}, {H*W, IC}, {}});
        //perf = OC * (2 * H * W - 1) * IC / time * mod;
        time = benchmarker_gemm.exec({{OC, IC}, {IC, H*W}, {}});
        perf = OC * (2 * IC -1) * H * W / time * mod;
        std::cout<<"Performance is " << perf <<" Gflops" <<std::endl;

    };

    //run(32, 32, 64, 64);
    //run(8, 8, 32, 32);
    //run(32, 32, 128, 128);
    //run(32, 32, 512, 512);
    //run(10,10,2,5);
    //run(100,100,2,50);

    run(16,4,240,135);
    run(8,32,120,67);
    run(16,64,60,33);

    run(1,1,28,28);
    run(8,1,28,28);
    run(2,2,28,28);
    run(8,2,28,28);
    run(4,4,28,28);
    run(16,4,28,28);
}

TEST_F(ARMV7, BENCHMARK_GROUP_CONVOLUTION_1X1) {
    int exec_times = 50;
    Benchmarker<Convolution> benchmarker_gconv1x1(handle());
    benchmarker_gconv1x1.set_times(exec_times);

    float mod = 1000 * exec_times / 1e9;
    auto run = [&](size_t IC, size_t OC,  size_t H, size_t W, size_t group){
        float time = 1.f, perf = 1.f;

        std::cout<<std::endl;
        std::cout<< "GCONV: IC " << IC << ", OC " << OC <<
                ", H " << H << ", W " << W <<", GROUP "<<group << std::endl;

        auto ICg = IC / group;
        auto OCg = OC / group;
        param::Convolution param;
        param.sparse = param::Convolution::Sparse::GROUP;
        time = benchmarker_gconv1x1.set_param(param).exec({{1, IC, H, W},
                {group, OCg, ICg, 1, 1},{}});
        perf = group * OCg * ICg * H * W / time * mod;
        std::cout<<"Performance is " << perf <<" Gflops" <<std::endl;
    };
    run(8*4, 1*4, 28, 28, 4);
    run(2*4, 2*4, 28, 28, 4);
    run(8*4, 2*4, 28, 28, 4);
    run(4*4, 4*4, 28, 28, 4);
    run(16*4, 4*4, 28, 28, 4);

}


// vim: syntax=cpp.doxygen
