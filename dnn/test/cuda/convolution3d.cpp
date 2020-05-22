/**
 * \file dnn/test/cuda/convolution3d.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/convolution3d.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "src/cuda/utils.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
#include "test/cuda/fixture.h"

namespace megdnn {
namespace test {

#if 0
TEST_F(CUDA, CONVOLUTION3D_8X8X32) {
    if (!cuda::is_compute_capability_required(6, 1)) {
        printf("Skip CUDA.CONVOLUTION_8X8X32 test as current device"
               "doesn't support\n");
        return;
    }
    using namespace convolution3d;
    std::vector<TestArg> args;
    {
        auto v = get_args();
        for (auto&& a : v) {
            args.push_back(std::move(a));
        }
    }
    /*
    {
        auto v = get_dilated_args();
        for (auto &&a: v) {
            args.push_back(std::move(a));
        }
    }
    {
        auto v = get_chanwise_args();
        for (auto &&a: v) {
            args.push_back(std::move(a));
        }
    }
    */
    Checker<Convolution3DForward> checker(handle_cuda());
    UniformIntRNG rng(-4, 4);
    UniformIntRNG rng_same(1, 1);
    for (auto arg : args) {
        arg.param.format = param::Convolution3D::Format::NDHWC;
        arg.param.data_type = param::Convolution3D::DataType::INT8x8x32;
        arg.src = cvt_src_or_dst_ncdhw2ndhwc(arg.src);
        arg.filter = cvt_filter_ncdhw2ndhwc(arg.filter);
        checker.set_dtype(0, dtype::Int8())
                .set_dtype(1, dtype::Int8())
                .set_dtype(2, dtype::Int32())
                .set_param(arg.param)
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .execs({arg.src, arg.filter, {}});
    }
}
#endif

TEST_F(CUDA, CONVOLUTION3D_FORWARD) {
    using namespace convolution3d;
    std::vector<TestArg> args = get_args();
    /*
    {
        auto v = get_chanwise_args();
        for (auto&& a : v) {
            args.push_back(std::move(a));
        }
    }
    {
        auto v = get_dilated_args();
        for (auto&& a : v) {
            args.push_back(std::move(a));
        }
    }
    */
    bool fp16_checked = false;
    Checker<Convolution3DForward> checker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 1.0f / sqrt(arg.filter[1] * arg.filter[2] *
                                  arg.filter[3] * arg.filter[4]);
        UniformFloatRNG rng(scale, 2 * scale);
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
        if (!fp16_checked || arg.src.total_nr_elems() >= 1000)
            continue;
        checker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
    }
}

TEST_F(CUDA, CONVOLUTION3D_1X1X1_FORWARD) {
    using namespace convolution3d;
    std::vector<TestArg> args = get_1x1x1_args();
    Checker<Convolution3DForward> checker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 1.0f / sqrt(arg.filter[1] * arg.filter[2] *
                                  arg.filter[3] * arg.filter[4]);
        UniformFloatRNG rng(scale, 2 * scale);
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
    }
}

TEST_F(CUDA, CONVOLUTION3D_MATMUL_FORWARD) {
    using namespace convolution3d;
    std::vector<TestArg> args = get_args();
    Checker<Convolution3DForward> checker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 1.0f / sqrt(arg.filter[1] * arg.filter[2] *
                                  arg.filter[3] * arg.filter[4]);
        UniformFloatRNG rng(scale, 2 * scale);
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_param(arg.param).
                execs({arg.src, arg.filter, {}});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_CONVOLUTION3D_MATMUL_BACKWARD_FILTER) {
    using namespace convolution3d;
    std::vector<TestArg> args = get_speed_test_args();
    Benchmarker<Convolution3DBackwardFilter> marker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 1.0f / sqrt(arg.filter[1] * arg.filter[2] *
                                  arg.filter[3] * arg.filter[4]);
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        auto opr = handle_cuda()->create_operator<Convolution3D>();
        opr->param() = arg.param;
        opr->deduce_layout(src, filter, dst);
        UniformFloatRNG rng(scale, 2 * scale);
        marker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_param(arg.param)
                .execs({src, dst, filter});
    }
}

TEST_F(CUDA, BENCHMARK_CONVOLUTION3D_MATMUL_FORWARD) {
    using namespace convolution3d;
    std::vector<TestArg> args = get_speed_test_args();
    Benchmarker<Convolution3DForward> marker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 1.0f / sqrt(arg.filter[1] * arg.filter[2] *
                                  arg.filter[3] * arg.filter[4]);
        UniformFloatRNG rng(scale, 2 * scale);
        marker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                . //set_param(arg.param).
                execs({arg.src, arg.filter, {}});
    }
}

TEST_F(CUDA, BENCHMARK_CONVOLUTION3D_1X1X1_FORWARD) {
    using namespace convolution3d;
    std::vector<TestArg> args = get_1x1x1_args();
    Benchmarker<Convolution3DForward> marker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 1.0f / sqrt(arg.filter[1] * arg.filter[2] *
                                  arg.filter[3] * arg.filter[4]);
        UniformFloatRNG rng(scale, 2 * scale);
        marker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .
                //      set_param(arg.param).
                execs({arg.src, arg.filter, {}});
    }
}

TEST_F(CUDA, BENCHMARK_CONVOLUTION3D_FORWARD) {
    using namespace convolution3d;
    std::vector<TestArg> args = get_args();
    {
        auto v = get_chanwise_args();
        for (auto&& a : v)
            args.push_back(std::move(a));
    }
    {
        auto v = get_1x1x1_args();
        for (auto&& a : v)
            args.push_back(std::move(a));
    }
    {
        auto v = get_dilated_args();
        for (auto&& a : v)
            args.push_back(std::move(a));
    }
    Benchmarker<Convolution3DForward> marker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 1.0f / sqrt(arg.filter[1] * arg.filter[2] *
                                  arg.filter[3] * arg.filter[4]);
        UniformFloatRNG rng(scale, 2 * scale);
        marker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
        marker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
    }
}

#endif


TEST_F(CUDA, CONVOLUTION3D_BACKWARD_DATA) {
    using namespace convolution3d;
    std::vector<TestArg> args = get_args();
    Checker<Convolution3DBackwardData> checker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 1.0f / sqrt(arg.filter[0] * arg.filter[2] *
                                  arg.filter[3] * arg.filter[4]);
        UniformFloatRNG rng(scale, 2 * scale);
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_cuda()->create_operator<Convolution3D>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
        src.dtype = dst.dtype = filter.dtype = dtype::Float16();
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
    }
}

TEST_F(CUDA, CONVOLUTION3D_BACKWARD_FILTER) {
    using namespace convolution3d;
    std::vector<TestArg> args = get_args();
    Checker<Convolution3DBackwardFilter> checker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_cuda()->create_operator<Convolution3D>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        float scale = 1.0f / sqrt(dst[0] * dst[2] * dst[3] * dst[4]);
        UniformFloatRNG rng(scale, 2 * scale);
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .exec(TensorLayoutArray{src, dst, filter});

        if (dst.total_nr_elems() >= 1000)
            continue;
        src.dtype = dst.dtype = filter.dtype = dtype::Float16();
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .exec(TensorLayoutArray{src, dst, filter});
    }
}

TEST_F(CUDA, CONVOLUTION3D_MATMUL_BACKWARD_FILTER) {
    using namespace convolution3d;
    std::vector<TestArg> args = get_args();
    Checker<Convolution3DBackwardFilter> checker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 1.0f / sqrt(arg.filter[1] * arg.filter[2] *
                                  arg.filter[3] * arg.filter[4]);
        UniformFloatRNG rng(scale, 2 * scale);
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        auto opr = handle_cuda()->create_operator<Convolution3D>();
        opr->param() = arg.param;
        opr->deduce_layout(src, filter, dst);
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_param(arg.param)
                .exec(TensorLayoutArray{src, dst, filter});
    }
}

/*
TEST_F(CUDA, CONV_CONFIG_COMBINATIONS) {
    auto eps_getter = [](bool f16, int stage, const char *name) -> float {
        if (f16) {
            return stage == 2 ? 0.9 : 0.7;
        }
        if (strstr(name, "WINOGRAD_NONFUSED"))
            return 0.3;
        return 1e-3;
    };
    convolution3d::test_conv_config_combinations(handle_cuda(), false, true,
true, eps_getter);
}
*/

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
