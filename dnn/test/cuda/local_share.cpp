/**
 * \file dnn/test/cuda/local_share.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/nn.h"

#include "src/common/utils.h"
#include "test/common/checker.h"
#include "test/common/convolution.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"
#include "test/cuda/utils.h"

using namespace megdnn;
using namespace test;

namespace {
struct LocalShareArgs {
    size_t b, c, f, p, s, h, w, sg;
};

std::vector<LocalShareArgs> get_local_share_conv_1x1_args_lar_bs() {
    std::vector<LocalShareArgs> ret;
    // clang-format off
    for (size_t b : {32, 64}) {
    for (size_t c : {32, 16, 8}) {
    for (size_t f : {1}) {
    for (int p : {0}) {
    for (size_t s : {1, 2}) {
    for (size_t h : {8, 16}) {
    for (size_t w : {2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 24, 32, 33}) {
    for (size_t sg : {3, 2}) {
        size_t ho = infer_conv_shape(h, f, s, p);
        size_t wo = infer_conv_shape(w, f, s, p);
        if (ho % sg != 0 || wo % sg != 0)
            continue;
        ret.emplace_back(LocalShareArgs{b, c, f, static_cast<size_t>(p),
                                        s, h, w, sg});
    } } } } } } } }
    // clang-format on
    return ret;
}

std::vector<LocalShareArgs> get_local_share_conv_3x3_args_lar_bs() {
    std::vector<LocalShareArgs> ret;
    // clang-format off
    for (size_t b : {32, 64}) {
    for (size_t c : {32, 16, 8}) {
    for (size_t f : {3}) {
    for (int p : {static_cast<int>(f / 2), 0}) {
    for (size_t s : {1, 2}) {
    for (size_t h : {8, 16}) {
    for (size_t w : {3, 4, 5, 6, 7, 8, 9, 10, 16, 24, 32, 33}) {
    for (size_t sg : {3, 2}) {
        size_t ho = infer_conv_shape(h, f, s, p);
        size_t wo = infer_conv_shape(w, f, s, p);
        if (ho % sg != 0 || wo % sg != 0)
            continue;
        ret.emplace_back(LocalShareArgs{b, c, f, static_cast<size_t>(p),
                                        s, h, w, sg});
    } } } } } } } }
    // clang-format on
    return ret;
}

std::vector<LocalShareArgs> get_local_share_conv_5x5_args_lar_bs() {
    std::vector<LocalShareArgs> ret;
    // clang-format off
    for (size_t b : {32, 64}) {
    for (size_t c : {32, 16, 8}) {
    for (size_t f : {5}) {
    for (int p : {static_cast<int>(f / 2), 0}) {
    for (size_t s : {1, 2}) {
    for (size_t h : {8, 16}) {
    for (size_t w : {8, 9, 10, 16, 24, 32, 33}) {
    for (size_t sg : {3, 2}) {
        size_t ho = infer_conv_shape(h, f, s, p);
        size_t wo = infer_conv_shape(w, f, s, p);
        if (ho % sg != 0 || wo % sg != 0)
            continue;
        ret.emplace_back(LocalShareArgs{b, c, f, static_cast<size_t>(p), s,
                                        h, w, sg});
    } } } } } } } }
    // clang-format on
    return ret;
}

std::vector<LocalShareArgs> get_local_share_conv_7x7_args_lar_bs() {
    std::vector<LocalShareArgs> ret;
    // clang-format off
    for (size_t b : {32, 64}) {
    for (size_t c : {32, 16, 8}) {
    for (size_t f : {7}) {
    for (int p : {static_cast<int>(f / 2), 0}) {
    for (size_t s : {1, 2}) {
    for (size_t h : {8, 16}) {
    for (size_t w : {8, 9, 10, 16, 24, 32, 33}) {
    for (size_t sg : {3, 2}) {
        size_t ho = infer_conv_shape(h, f, s, p);
        size_t wo = infer_conv_shape(w, f, s, p);
        if (ho % sg != 0 || wo % sg != 0)
            continue;
        ret.emplace_back(LocalShareArgs{b, c, f, static_cast<size_t>(p), s,
                                        h, w, sg});
    } } } } } } } }
    // clang-format on
    return ret;
}

std::vector<LocalShareArgs> get_local_share_conv_small_image(size_t kernel_size) {
    size_t f = kernel_size;
    std::vector<LocalShareArgs> ret;
    // clang-format off
    for (size_t b : {8, 16, 32, 48, 64}) {
    for (size_t c : {8, 16, 32, 48, 64, 96, 128}) {
    for (int p : {static_cast<int>(f / 2), 0}) {
    for (size_t s : {1, 2}) {
    for (size_t h : {12}) {
    for (size_t w : {12}) {
    for (size_t sg : {3, 2}) {
        size_t ho = infer_conv_shape(h, f, s, p);
        size_t wo = infer_conv_shape(w, f, s, p);
        if (ho % sg != 0 || wo % sg != 0)
            continue;
        ret.emplace_back(LocalShareArgs{b, c, f, static_cast<size_t>(p), s,
                                        h, w, sg});
    } } } } } } }
    // clang-format on
    return ret;
}

std::vector<LocalShareArgs> get_local_share_conv_small_image() {
    std::vector<LocalShareArgs> ret = get_local_share_conv_small_image(3);
    auto ret1 = get_local_share_conv_small_image(5);
    auto ret2 = get_local_share_conv_small_image(7);
    ret.insert(ret.begin(), ret1.begin(), ret1.end());
    ret.insert(ret.begin(), ret2.begin(), ret2.end());
    return ret;
}

void test_local_share_bwd_data_implicit_gemm(size_t kernel_size,
                                             Handle* handle) {
    Checker<LocalShareBackwardData> checker(handle);
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShareBackwardData>(
            "LOCAL_SHARE_IMPLICIT_GEMM", &require_algo));
    using Param = LocalShare::Param;
    ConstValue const_0{0};
    auto args = get_local_share_conv_small_image(kernel_size);
    for (auto&& arg : args) {
        static_cast<void>(arg);
        size_t b = arg.b, c = arg.c, f = arg.f, p = arg.p, s = arg.s, h = arg.h,
               w = arg.w, sg = arg.sg;
        size_t ho = infer_conv_shape(h, f, s, p),
               wo = infer_conv_shape(w, f, s, p);
        Param param;
        param.stride_h = param.stride_w = s;
        param.pad_h = param.pad_w = p;
        param.spatial_groups_h = param.spatial_groups_w = sg;
        checker.set_param(param);
        checker.set_rng(2, &const_0);
        TensorShape diff{b, c, ho, wo}, filter{sg, sg, 4, f, f, c},
                grad{b, 4, h, w};
        checker.execs({filter, diff, grad});
        diff = TensorShape{b, c, ho, wo},
        filter = TensorShape{sg, sg, 8, f, f, c};
        grad = {b, 8, h, w};
        checker.exec({filter, diff, grad});
    }
}
}  // namespace

TEST_F(CUDA, LOCAL_SHARE_FORWARD_1x1_LAR_BS) {
    require_compute_capability(6, 0);
    Checker<LocalShare> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShare>(
            "LOCAL_SHARE_CHWN_BATCH_SIZE_AWARE", &require_algo));
    using Param = LocalShare::Param;
    auto args = get_local_share_conv_1x1_args_lar_bs();
    for (auto&& arg : args) {
        size_t b = arg.b, c = arg.c, f = arg.f, p = arg.p, s = arg.s, h = arg.h,
               w = arg.w, sg = arg.sg;
        Param param;
        param.stride_h = param.stride_w = s;
        param.pad_h = param.pad_w = p;
        param.spatial_groups_h = param.spatial_groups_w = sg;
        checker.set_param(param);
        TensorShape src{b, 4, h, w}, filter{sg, sg, 4, f, f, c};
        checker.execs({src, filter, {}});
        src = TensorShape{b, 8, h, w}, filter = TensorShape{sg, sg, 8, f, f, c};
        checker.exec({src, filter, {}});
    }
}

TEST_F(CUDA, LOCAL_SHARE_FORWARD_3x3_LAR_BS) {
    require_compute_capability(6, 0);
    Checker<LocalShare> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShare>(
            "LOCAL_SHARE_CHWN_BATCH_SIZE_AWARE", &require_algo));
    using Param = LocalShare::Param;
    auto args = get_local_share_conv_3x3_args_lar_bs();
    ConstValue const_1{1};
    for (auto&& arg : args) {
        size_t b = arg.b, c = arg.c, f = arg.f, p = arg.p, s = arg.s, h = arg.h,
               w = arg.w, sg = arg.sg;
        Param param;
        param.stride_h = param.stride_w = s;
        param.pad_h = param.pad_w = p;
        param.spatial_groups_h = param.spatial_groups_w = sg;
        checker.set_param(param);
        TensorShape src{b, 4, h, w}, filter{sg, sg, 4, f, f, c};
        checker.execs({src, filter, {}});
        src = TensorShape{b, 8, h, w}, filter = TensorShape{sg, sg, 8, f, f, c};
        checker.exec({src, filter, {}});
    }
}

TEST_F(CUDA, LOCAL_SHARE_FORWARD_5x5_LAR_BS) {
    require_compute_capability(6, 0);
    Checker<LocalShare> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShare>(
            "LOCAL_SHARE_CHWN_BATCH_SIZE_AWARE", &require_algo));
    using Param = LocalShare::Param;
    auto args = get_local_share_conv_5x5_args_lar_bs();
    for (auto&& arg : args) {
        size_t b = arg.b, c = arg.c, f = arg.f, p = arg.p, s = arg.s, h = arg.h,
               w = arg.w, sg = arg.sg;
        Param param;
        param.stride_h = param.stride_w = s;
        param.pad_h = param.pad_w = p;
        param.spatial_groups_h = param.spatial_groups_w = sg;
        checker.set_param(param);
        TensorShape src{b, 4, h, w}, filter{sg, sg, 4, f, f, c};
        checker.execs({src, filter, {}});
        src = TensorShape{b, 8, h, w}, filter = TensorShape{sg, sg, 8, f, f, c};
        checker.exec({src, filter, {}});
    }
}

TEST_F(CUDA, LOCAL_SHARE_FORWARD_7x7_LAR_BS) {
    require_compute_capability(6, 0);
    Checker<LocalShare> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShare>(
            "LOCAL_SHARE_CHWN_BATCH_SIZE_AWARE", &require_algo));
    using Param = LocalShare::Param;
    auto args = get_local_share_conv_7x7_args_lar_bs();
    for (auto&& arg : args) {
        size_t b = arg.b, c = arg.c, f = arg.f, p = arg.p, s = arg.s, h = arg.h,
               w = arg.w, sg = arg.sg;
        Param param;
        param.stride_h = param.stride_w = s;
        param.pad_h = param.pad_w = p;
        param.spatial_groups_h = param.spatial_groups_w = sg;
        checker.set_param(param);
        TensorShape src{b, 4, h, w}, filter{sg, sg, 4, f, f, c};
        checker.execs({src, filter, {}});
        src = TensorShape{b, 8, h, w}, filter = TensorShape{sg, sg, 8, f, f, c};
        checker.exec({src, filter, {}});
    }
}

TEST_F(CUDA, LOCAL_SHARE_BATCHED_MATMUL) {
    Checker<LocalShare> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShare>(
            "LOCAL_SHARE_BATCHED_MATMUL", &require_algo));
    using Param = LocalShare::Param;
    auto args = convolution::get_args();
    for (size_t sg : {2, 3}) {
        for (auto&& arg : args) {
            if (arg.param.sparse != LocalShare::Param::Sparse::DENSE)
                continue;
            if (arg.param.format != LocalShare::Param::Format::NCHW)
                continue;
            if (arg.param.dilate_h != 1 || arg.param.dilate_w != 1)
                continue;
            Param param;
            param.sparse = arg.param.sparse;
            param.stride_h = arg.param.stride_h,
            param.stride_w = arg.param.stride_w;
            param.pad_h = arg.param.pad_h, param.pad_w = arg.param.pad_w;
            param.dilate_h = arg.param.dilate_h,
            param.dilate_w = arg.param.dilate_w;
            param.spatial_groups_h = param.spatial_groups_w = sg;
            size_t ho = infer_conv_shape(arg.src[2], arg.filter[2],
                                         param.stride_h, param.pad_h);
            size_t wo = infer_conv_shape(arg.src[3], arg.filter[3],
                                         param.stride_w, param.pad_w);
            if (ho % sg != 0 || wo % sg != 0)
                continue;
            TensorShape filter{sg,
                               sg,
                               arg.filter[1],
                               arg.filter[2],
                               arg.filter[3],
                               arg.filter[0]};
            checker.set_param(param);
            checker.exec({arg.src, filter, {}});
        }
    }
}

TEST_F(CUDA, GROUP_LOCAL_SHARE_BATCHED_MATMUL) {
    Checker<LocalShare> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShare>(
            "LOCAL_SHARE_BATCHED_MATMUL", &require_algo));
    using Param = LocalShare::Param;
    auto args = convolution::get_args();
    for (size_t sg : {2, 3}) {
        for (auto&& arg : args) {
            if (arg.param.sparse != LocalShare::Param::Sparse::DENSE)
                continue;
            if (arg.param.format != LocalShare::Param::Format::NCHW)
                continue;
            if (arg.param.dilate_h != 1 || arg.param.dilate_w != 1)
                continue;
            if (arg.filter.ndim != 4)
                continue;
            Param param;
            param.sparse = Param::Sparse::GROUP;
            param.stride_h = arg.param.stride_h,
            param.stride_w = arg.param.stride_w;
            param.pad_h = arg.param.pad_h, param.pad_w = arg.param.pad_w;
            param.dilate_h = arg.param.dilate_h,
            param.dilate_w = arg.param.dilate_w;
            param.spatial_groups_h = param.spatial_groups_w = sg;
            size_t ho = infer_conv_shape(arg.src[2], arg.filter[2],
                                         param.stride_h, param.pad_h);
            size_t wo = infer_conv_shape(arg.src[3], arg.filter[3],
                                         param.stride_w, param.pad_w);
            if (ho % sg != 0 || wo % sg != 0)
                continue;
            size_t nr_groups = 3;
            TensorShape filter{nr_groups,
                               sg,
                               sg,
                               arg.filter[1],
                               arg.filter[2],
                               arg.filter[3],
                               arg.filter[0]};
            TensorShape src{arg.src[0], arg.src[1] * nr_groups, arg.src[2],
                            arg.src[3]};
            checker.set_param(param);
            checker.exec({src, filter, {}});
        }
    }
}

TEST_F(CUDA, LOCAL_SHARE_FORWARD_SMALL_IMAGE_GENERAL) {
    require_compute_capability(6, 0);
    Checker<LocalShare> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShare>(
            "LOCAL_SHARE_CHWN_BATCH_SIZE_AWARE_SMALL_IMAGE", &require_algo));
    using Param = LocalShare::Param;
    auto args = convolution::get_args();
    for (size_t sg : {2, 3}) {
        for (auto&& arg : args) {
            if (arg.param.sparse != LocalShare::Param::Sparse::DENSE)
                continue;
            if (arg.param.format != LocalShare::Param::Format::NCHW)
                continue;
            if (arg.param.dilate_h != 1 || arg.param.dilate_w != 1)
                continue;
            Param param;
            param.stride_h = arg.param.stride_h,
            param.stride_w = arg.param.stride_w;
            param.pad_h = arg.param.pad_h, param.pad_w = arg.param.pad_w;
            param.dilate_h = arg.param.dilate_h,
            param.dilate_w = arg.param.dilate_w;
            param.spatial_groups_h = param.spatial_groups_w = sg;
            size_t ho = infer_conv_shape(arg.src[2], arg.filter[2],
                                         param.stride_h, param.pad_h);
            size_t wo = infer_conv_shape(arg.src[3], arg.filter[3],
                                         param.stride_w, param.pad_w);
            if (ho % sg != 0 || wo % sg != 0)
                continue;
            arg.filter[1] = arg.filter[1] + (4 - arg.filter[1] % 4);
            arg.src[1] = arg.filter[1];
            TensorShape filter{sg,
                               sg,
                               arg.filter[1],
                               arg.filter[2],
                               arg.filter[3],
                               arg.filter[0]};
            checker.set_param(param);
            checker.exec({arg.src, filter, {}});
        }
    }
}

TEST_F(CUDA, LOCAL_SHARE_FORWARD_SMALL_IMAGE_SPECIAL) {
    require_compute_capability(6, 0);
    Checker<LocalShare> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShare>(
            "LOCAL_SHARE_CHWN_BATCH_SIZE_AWARE_SMALL_IMAGE", &require_algo));
    using Param = LocalShare::Param;
    auto args = get_local_share_conv_small_image();
    for (auto&& arg : args) {
        size_t b = arg.b, c = arg.c, f = arg.f, p = arg.p, s = arg.s, h = arg.h,
               w = arg.w, sg = arg.sg;
        Param param;
        param.stride_h = param.stride_w = s;
        param.pad_h = param.pad_w = p;
        param.spatial_groups_h = param.spatial_groups_w = sg;
        checker.set_param(param);
        TensorShape src{b, 4, h, w}, filter{sg, sg, 4, f, f, c};
        checker.execs({src, filter, {}});
        src = TensorShape{b, 8, h, w}, filter = TensorShape{sg, sg, 8, f, f, c};
        checker.exec({src, filter, {}});
    }
}

TEST_F(CUDA, LOCAL_SHARE_BWD_DATA_IMPLICIT_GEMM_GENERAL) {
    require_compute_capability(6, 0);
    Checker<LocalShareBackwardData> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShareBackwardData>(
            "LOCAL_SHARE_IMPLICIT_GEMM", &require_algo));
    using Param = LocalShare::Param;
    auto args = convolution::get_args();
    ConstValue const_0{0};
    for (size_t sg : {2, 3}) {
        for (auto&& arg : args) {
            if (arg.param.sparse != LocalShare::Param::Sparse::DENSE)
                continue;
            if (arg.param.format != LocalShare::Param::Format::NCHW)
                continue;
            if (arg.param.dilate_h != 1 || arg.param.dilate_w != 1)
                continue;
            Param param;
            param.stride_h = arg.param.stride_h,
            param.stride_w = arg.param.stride_w;
            param.pad_h = arg.param.pad_h, param.pad_w = arg.param.pad_w;
            param.dilate_h = arg.param.dilate_h,
            param.dilate_w = arg.param.dilate_w;
            param.spatial_groups_h = param.spatial_groups_w = sg;
            size_t ho = infer_conv_shape(arg.src[2], arg.filter[2],
                                         param.stride_h, param.pad_h);
            size_t wo = infer_conv_shape(arg.src[3], arg.filter[3],
                                         param.stride_w, param.pad_w);
            if (ho % sg != 0 || wo % sg != 0)
                continue;
            arg.filter[0] = arg.filter[0] + (4 - arg.filter[0] % 4);
            TensorShape filter{sg,
                               sg,
                               arg.filter[1],
                               arg.filter[2],
                               arg.filter[3],
                               arg.filter[0]};
            TensorShape diff{arg.src[0], arg.filter[0], ho, wo};
            checker.set_param(param);
            checker.set_rng(2, &const_0);
            checker.exec({filter, diff, arg.src});
        }
    }
}

TEST_F(CUDA, LOCAL_SHARE_BWD_DATA_IMPLICIT_GEMM_SPECIAL_PART1) {
    require_compute_capability(6, 0);
    test_local_share_bwd_data_implicit_gemm(3, handle_cuda());
}

TEST_F(CUDA, LOCAL_SHARE_BWD_DATA_IMPLICIT_GEMM_SPECIAL_PART2) {
    require_compute_capability(6, 0);
    test_local_share_bwd_data_implicit_gemm(5, handle_cuda());
}

TEST_F(CUDA, LOCAL_SHARE_BWD_DATA_IMPLICIT_GEMM_SPECIAL_PART3) {
    require_compute_capability(6, 0);
    test_local_share_bwd_data_implicit_gemm(7, handle_cuda());
}

TEST_F(CUDA, LOCAL_SHARE_BWD_DATA_BATCHED_MATMUL) {
    Checker<LocalShareBackwardData> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShareBackwardData>(
            "LOCAL_SHARE_BATCHED_MATMUL", &require_algo));
    using Param = LocalShare::Param;
    auto args = convolution::get_args();
    ConstValue const_0{0};
    for (size_t sg : {2, 3}) {
        for (auto&& arg : args) {
            if (arg.param.sparse != LocalShare::Param::Sparse::DENSE)
                continue;
            if (arg.param.format != LocalShare::Param::Format::NCHW)
                continue;
            if (arg.param.dilate_h != 1 || arg.param.dilate_w != 1)
                continue;
            Param param;
            param.stride_h = arg.param.stride_h,
            param.stride_w = arg.param.stride_w;
            param.pad_h = arg.param.pad_h, param.pad_w = arg.param.pad_w;
            param.dilate_h = arg.param.dilate_h,
            param.dilate_w = arg.param.dilate_w;
            param.spatial_groups_h = param.spatial_groups_w = sg;
            size_t ho = infer_conv_shape(arg.src[2], arg.filter[2],
                                         param.stride_h, param.pad_h);
            size_t wo = infer_conv_shape(arg.src[3], arg.filter[3],
                                         param.stride_w, param.pad_w);
            if (ho % sg != 0 || wo % sg != 0)
                continue;
            TensorShape filter{sg,
                               sg,
                               arg.filter[1],
                               arg.filter[2],
                               arg.filter[3],
                               arg.filter[0]};
            TensorShape diff{arg.src[0], arg.filter[0], ho, wo};
            checker.set_rng(2, &const_0);
            checker.set_param(param);
            checker.exec({filter, diff, arg.src});
        }
    }
}

TEST_F(CUDA, GROUP_LOCAL_SHARE_BWD_DATA_BATCHED_MATMUL) {
    Checker<LocalShareBackwardData> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShareBackwardData>(
            "LOCAL_SHARE_BATCHED_MATMUL", &require_algo));
    using Param = LocalShare::Param;
    auto args = convolution::get_args();
    ConstValue const_0{0};
    for (size_t sg : {2, 3}) {
        for (auto&& arg : args) {
            if (arg.param.sparse != LocalShare::Param::Sparse::DENSE)
                continue;
            if (arg.param.format != LocalShare::Param::Format::NCHW)
                continue;
            if (arg.param.dilate_h != 1 || arg.param.dilate_w != 1)
                continue;
            Param param;
            param.sparse = Param::Sparse::GROUP;
            param.stride_h = arg.param.stride_h,
            param.stride_w = arg.param.stride_w;
            param.pad_h = arg.param.pad_h, param.pad_w = arg.param.pad_w;
            param.dilate_h = arg.param.dilate_h,
            param.dilate_w = arg.param.dilate_w;
            param.spatial_groups_h = param.spatial_groups_w = sg;
            size_t ho = infer_conv_shape(arg.src[2], arg.filter[2],
                                         param.stride_h, param.pad_h);
            size_t wo = infer_conv_shape(arg.src[3], arg.filter[3],
                                         param.stride_w, param.pad_w);
            if (ho % sg != 0 || wo % sg != 0)
                continue;
            size_t nr_groups = 3;
            TensorShape filter{nr_groups,
                               sg,
                               sg,
                               arg.filter[1],
                               arg.filter[2],
                               arg.filter[3],
                               arg.filter[0]};
            TensorShape diff{arg.src[0], arg.filter[0] * nr_groups, ho, wo};
            TensorShape grad{arg.src[0], arg.src[1] * nr_groups, arg.src[2],
                             arg.src[3]};
            checker.set_rng(2, &const_0);
            checker.set_param(param);
            checker.exec({filter, diff, grad});
        }
    }
}

TEST_F(CUDA, LOCAL_SHARE_BWD_FILTER_IMPLICIT_GEMM_GENERAL) {
    require_compute_capability(6, 0);
    Checker<LocalShareBackwardFilter> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShareBackwardFilter>(
            "LOCAL_SHARE_IMPLICIT_GEMM", &require_algo));
    using Param = LocalShare::Param;
    auto args = convolution::get_args();
    ConstValue const_0{0};
    for (size_t sg : {2, 3}) {
        for (auto&& arg : args) {
            if (arg.param.sparse != LocalShare::Param::Sparse::DENSE)
                continue;
            if (arg.param.format != LocalShare::Param::Format::NCHW)
                continue;
            if (arg.param.dilate_h != 1 || arg.param.dilate_w != 1)
                continue;
            Param param;
            param.stride_h = arg.param.stride_h,
            param.stride_w = arg.param.stride_w;
            param.pad_h = arg.param.pad_h, param.pad_w = arg.param.pad_w;
            param.dilate_h = arg.param.dilate_h,
            param.dilate_w = arg.param.dilate_w;
            param.spatial_groups_h = param.spatial_groups_w = sg;
            size_t ho = infer_conv_shape(arg.src[2], arg.filter[2],
                                         param.stride_h, param.pad_h);
            size_t wo = infer_conv_shape(arg.src[3], arg.filter[3],
                                         param.stride_w, param.pad_w);
            if (ho % sg != 0 || wo % sg != 0)
                continue;
            arg.src[0] = arg.src[0] + (4 - arg.src[0] % 4);
            TensorShape grad{sg,
                             sg,
                             arg.filter[1],
                             arg.filter[2],
                             arg.filter[3],
                             arg.filter[0]};
            TensorShape diff{arg.src[0], arg.filter[0], ho, wo};
            checker.set_param(param);
            checker.set_rng(2, &const_0);
            checker.exec({arg.src, diff, grad});
        }
    }
}

TEST_F(CUDA, LOCAL_SHARE_BWD_FILTER_IMPLICIT_GEMM_SPECIAL) {
    require_compute_capability(6, 0);
    Checker<LocalShareBackwardFilter> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShareBackwardFilter>(
            "LOCAL_SHARE_IMPLICIT_GEMM", &require_algo));
    using Param = LocalShare::Param;
    ConstValue const_0{0};
    auto args = get_local_share_conv_small_image();
    for (auto&& arg : args) {
        static_cast<void>(arg);
        size_t b = arg.b, c = arg.c, f = arg.f, p = arg.p, s = arg.s, h = arg.h,
               w = arg.w, sg = arg.sg;
        size_t ho = infer_conv_shape(h, f, s, p),
               wo = infer_conv_shape(w, f, s, p);
        Param param;
        param.stride_h = param.stride_w = s;
        param.pad_h = param.pad_w = p;
        param.spatial_groups_h = param.spatial_groups_w = sg;
        checker.set_param(param);
        checker.set_rng(2, &const_0);
        TensorShape diff{b, c, ho, wo}, grad{sg, sg, 4, f, f, c},
                src{b, 4, h, w};
        checker.execs({src, diff, grad});
        src = {b, 8, h, w};
        diff = TensorShape{b, c, ho, wo},
        grad = TensorShape{sg, sg, 8, f, f, c};
        checker.exec({src, diff, grad});
    }
}

TEST_F(CUDA, LOCAL_SHARE_BWD_FILTER_BATCHED_MATMUL) {
    Checker<LocalShareBackwardFilter> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShareBackwardFilter>(
            "LOCAL_SHARE_BATCHED_MATMUL", &require_algo));
    using Param = LocalShare::Param;
    auto args = convolution::get_args();
    ConstValue const_0{0};
    for (size_t sg : {2, 3}) {
        for (auto&& arg : args) {
            if (arg.param.sparse != LocalShare::Param::Sparse::DENSE)
                continue;
            if (arg.param.format != LocalShare::Param::Format::NCHW)
                continue;
            if (arg.param.dilate_h != 1 || arg.param.dilate_w != 1)
                continue;
            Param param;
            param.stride_h = arg.param.stride_h,
            param.stride_w = arg.param.stride_w;
            param.pad_h = arg.param.pad_h, param.pad_w = arg.param.pad_w;
            param.dilate_h = arg.param.dilate_h,
            param.dilate_w = arg.param.dilate_w;
            param.spatial_groups_h = param.spatial_groups_w = sg;
            size_t ho = infer_conv_shape(arg.src[2], arg.filter[2],
                                         param.stride_h, param.pad_h);
            size_t wo = infer_conv_shape(arg.src[3], arg.filter[3],
                                         param.stride_w, param.pad_w);
            if (ho % sg != 0 || wo % sg != 0)
                continue;
            TensorShape grad{sg,
                             sg,
                             arg.filter[1],
                             arg.filter[2],
                             arg.filter[3],
                             arg.filter[0]};
            TensorShape diff{arg.src[0], arg.filter[0], ho, wo};
            checker.set_rng(2, &const_0);
            checker.set_param(param);
            checker.exec({arg.src, diff, grad});
        }
    }
}

TEST_F(CUDA, GROUP_LOCAL_SHARE_BWD_FILTER_BATCHED_MATMUL) {
    Checker<LocalShareBackwardFilter> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<LocalShareBackwardFilter>(
            "LOCAL_SHARE_BATCHED_MATMUL", &require_algo));
    using Param = LocalShare::Param;
    auto args = convolution::get_args();
    ConstValue const_0{0};
    for (size_t sg : {2, 3}) {
        for (auto&& arg : args) {
            if (arg.param.sparse != LocalShare::Param::Sparse::DENSE)
                continue;
            if (arg.param.format != LocalShare::Param::Format::NCHW)
                continue;
            if (arg.param.dilate_h != 1 || arg.param.dilate_w != 1)
                continue;
            Param param;
            param.sparse = Param::Sparse::GROUP;
            param.stride_h = arg.param.stride_h,
            param.stride_w = arg.param.stride_w;
            param.pad_h = arg.param.pad_h, param.pad_w = arg.param.pad_w;
            param.dilate_h = arg.param.dilate_h,
            param.dilate_w = arg.param.dilate_w;
            param.spatial_groups_h = param.spatial_groups_w = sg;
            size_t ho = infer_conv_shape(arg.src[2], arg.filter[2],
                                         param.stride_h, param.pad_h);
            size_t wo = infer_conv_shape(arg.src[3], arg.filter[3],
                                         param.stride_w, param.pad_w);
            if (ho % sg != 0 || wo % sg != 0)
                continue;
            size_t nr_groups = 3;
            TensorShape grad{nr_groups,
                             sg,
                             sg,
                             arg.filter[1],
                             arg.filter[2],
                             arg.filter[3],
                             arg.filter[0]};
            TensorShape diff{arg.src[0], arg.filter[0] * nr_groups, ho, wo};
            TensorShape src{arg.src[0], arg.src[1] * nr_groups, arg.src[2],
                            arg.src[3]};
            checker.set_rng(2, &const_0);
            checker.set_param(param);
            checker.exec({src, diff, grad});
        }
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_LOCAL_SHARE_BWD_FILTER) {
    CUBenchmarker<LocalShareBackwardFilter> bencher(handle_cuda());
    size_t RUNS = 1000;
    bencher.set_display(false).set_times(RUNS);
    std::unique_ptr<OprProxy<LocalShareBackwardFilter>> proxy{
            new OprProxy<LocalShareBackwardFilter>{true}};
    bencher.set_proxy(proxy);

    LocalShare::Param param;
    NormalRNG rng;

    auto run = [&](size_t batch, size_t ic, size_t ih, size_t iw, size_t oc,
                   size_t f, size_t s, size_t sg) {
        param.pad_h = f / 2;
        param.pad_w = f / 2;
        param.stride_h = s;
        param.stride_w = s;
        param.spatial_groups_h = sg;
        param.spatial_groups_w = sg;

        TensorShape src = {batch, ic, ih, iw}, grad = {sg, sg, ic, f, f, oc};
        size_t ho = infer_conv_shape(ih, f, s, f / 2);
        size_t wo = infer_conv_shape(iw, f, s, f / 2);
        TensorShape diff = {batch, oc, ho, wo};

        float flo = 2.0 * batch * oc * ho * wo * ic * f * f / (1e12);

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        bencher.proxy()->target_algo_info.reset();
        auto time_in_ms = bencher.execs({src, diff, grad}) / RUNS;

        printf("src=%s, diff=%s, grad=%s, float32: %.2fms "
               "%.2fTFlops\n",
               src.to_string().c_str(), diff.to_string().c_str(),
               grad.to_string().c_str(), time_in_ms,
               (flo / (time_in_ms * 1e-3)));

    };
    // stride = 1
    run(32, 128, 24, 24, 128, 1, 1, 3);
    run(32, 256, 12, 12, 256, 1, 1, 3);

    // stride = 2
    run(32, 256, 12, 12, 512, 1, 2, 3);
    run(32, 512, 6, 6, 1024, 1, 2, 3);

    // stride = 1
    run(32, 128, 24, 24, 128, 3, 1, 3);
    run(32, 256, 12, 12, 256, 3, 1, 3);

    // stride = 2
    run(32, 128, 24, 24, 256, 3, 2, 3);
    run(32, 256, 12, 12, 512, 3, 2, 3);

    // stride = 1
    run(64, 128, 24, 24, 128, 1, 1, 3);
    run(64, 256, 12, 12, 256, 1, 1, 3);

    // stride = 2
    run(64, 256, 12, 12, 512, 1, 2, 3);
    run(64, 512, 6, 6, 1024, 1, 2, 3);

    // stride = 1
    run(64, 128, 24, 24, 128, 3, 1, 3);
    run(64, 256, 12, 12, 256, 3, 1, 3);

    // stride = 2
    run(64, 128, 24, 24, 256, 3, 2, 3);
    run(64, 256, 12, 12, 512, 3, 2, 3);
}


TEST_F(CUDA, BENCHMARK_GROUP_LOCAL_SHARE_FORWARD) {
    CUBenchmarker<LocalShare> bencher(handle_cuda());
    size_t RUNS = 1000;
    bencher.set_display(false).set_times(RUNS);
    std::unique_ptr<OprProxy<LocalShareForward>> proxy{
            new OprProxy<LocalShareForward>{true}};
    bencher.set_proxy(proxy);

    LocalShare::Param param;
    NormalRNG rng;

    auto run = [&](size_t batch, size_t ic, size_t ih, size_t iw, size_t oc,
                   size_t f, size_t s, size_t sg) {
        param.pad_h = f / 2;
        param.pad_w = f / 2;
        param.stride_h = s;
        param.stride_w = s;
        param.spatial_groups_h = sg;
        param.spatial_groups_w = sg;
        param.sparse = LocalShare::Param::Sparse::GROUP;

        TensorShape src = {1, batch * ic, ih, iw},
                    filter = {batch, sg, sg, ic, f, f, oc};
        size_t ho = infer_conv_shape(ih, f, s, f / 2);
        size_t wo = infer_conv_shape(iw, f, s, f / 2);

        float flo = 2.0 * batch * oc * ho * wo * ic * f * f / (1e12);

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        bencher.proxy()->target_algo_info.reset();
        auto time_in_ms = bencher.execs({src, filter, {}}) / RUNS;
        ;

        printf("src=%s, filter=%s, float32: %.2fms %.2fTFlops\n",
               src.to_string().c_str(), filter.to_string().c_str(), time_in_ms,
               (flo / (time_in_ms * 1e-3)));

    };
    // stride = 1
    run(32, 128, 24, 24, 128, 1, 1, 3);
    run(32, 256, 12, 12, 256, 1, 1, 3);

    // stride = 2
    run(32, 256, 12, 12, 512, 1, 2, 3);
    run(32, 512, 6, 6, 1024, 1, 2, 3);

    // stride = 1
    run(64, 128, 24, 24, 128, 1, 1, 3);
    run(64, 256, 12, 12, 256, 1, 1, 3);

    // stride = 2
    run(64, 256, 12, 12, 512, 1, 2, 3);
    run(64, 512, 6, 6, 1024, 1, 2, 3);
}

TEST_F(CUDA, BENCHMARK_LOCAL_SHARE_BWD_DATA) {
    CUBenchmarker<LocalShareBackwardData> bencher(handle_cuda());
    size_t RUNS = 1000;
    bencher.set_display(false).set_times(RUNS);
    std::unique_ptr<OprProxy<LocalShareBackwardData>> proxy{
            new OprProxy<LocalShareBackwardData>{true}};
    bencher.set_proxy(proxy);

    LocalShare::Param param;
    NormalRNG rng;

    auto run = [&](size_t batch, size_t ic, size_t ih, size_t iw, size_t oc,
                   size_t f, size_t s, size_t sg) {
        param.pad_h = f / 2;
        param.pad_w = f / 2;
        param.stride_h = s;
        param.stride_w = s;
        param.spatial_groups_h = sg;
        param.spatial_groups_w = sg;

        TensorShape grad = {batch, ic, ih, iw}, filter = {sg, sg, ic, f, f, oc};
        size_t ho = infer_conv_shape(ih, f, s, f / 2);
        size_t wo = infer_conv_shape(iw, f, s, f / 2);
        TensorShape diff = {batch, oc, ho, wo};

        float flo = 2.0 * batch * oc * ho * wo * ic * f * f / (1e12);

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        bencher.proxy()->target_algo_info.reset();
        auto time_in_ms = bencher.execs({filter, diff, grad}) / RUNS;

        printf("filter=%s, diff=%s, grad=%s, float32: %.2fms "
               "%.2fTFlops\n",
               filter.to_string().c_str(), diff.to_string().c_str(),
               grad.to_string().c_str(), time_in_ms,
               (flo / (time_in_ms * 1e-3)));

    };
    // stride = 1
    run(32, 128, 24, 24, 128, 1, 1, 3);
    run(32, 256, 12, 12, 256, 1, 1, 3);

    // stride = 2
    run(32, 256, 12, 12, 512, 1, 2, 3);
    run(32, 512, 6, 6, 1024, 1, 2, 3);

    // stride = 1
    run(32, 128, 24, 24, 128, 3, 1, 3);
    run(32, 256, 12, 12, 256, 3, 1, 3);

    // stride = 2
    run(32, 128, 24, 24, 256, 3, 2, 3);
    run(32, 256, 12, 12, 512, 3, 2, 3);

    // stride = 1
    run(64, 128, 24, 24, 128, 1, 1, 3);
    run(64, 256, 12, 12, 256, 1, 1, 3);

    // stride = 2
    run(64, 256, 12, 12, 512, 1, 2, 3);
    run(64, 512, 6, 6, 1024, 1, 2, 3);

    // stride = 1
    run(64, 128, 24, 24, 128, 3, 1, 3);
    run(64, 256, 12, 12, 256, 3, 1, 3);

    // stride = 2
    run(64, 128, 24, 24, 256, 3, 2, 3);
    run(64, 256, 12, 12, 512, 3, 2, 3);
}

TEST_F(CUDA, BENCHMARK_LOCAL_SHARE_FORWARD_BOTTLENECK) {
    CUBenchmarker<LocalShare> bencher(handle_cuda());
    CUBenchmarker<Convolution> bencher_conv(handle_cuda());
    size_t RUNS = 1000;
    bencher.set_display(false).set_times(RUNS);
    std::unique_ptr<OprProxy<LocalShareForward>> proxy{
            new OprProxy<LocalShareForward>{true}};
    bencher.set_proxy(proxy);

    bencher_conv.set_display(false).set_times(RUNS);
    std::unique_ptr<OprProxy<Convolution>> conv_proxy{
            new OprProxy<Convolution>{true}};
    bencher_conv.set_proxy(conv_proxy);

    LocalShare::Param param;
    Convolution::Param conv_param;
    NormalRNG rng;

    auto run = [&](size_t batch, size_t ic, size_t ih, size_t iw, size_t oc,
                   size_t f, size_t s, size_t sg) {
        param.pad_h = f / 2;
        param.pad_w = f / 2;
        param.stride_h = s;
        param.stride_w = s;
        param.spatial_groups_h = sg;
        param.spatial_groups_w = sg;

        conv_param.pad_h = f / 2;
        conv_param.pad_w = f / 2;
        conv_param.stride_h = s;
        conv_param.stride_w = s;

        TensorShape src = {batch, ic, ih, iw}, filter = {sg, sg, ic, f, f, oc};
        size_t ho = infer_conv_shape(ih, f, s, f / 2);
        size_t wo = infer_conv_shape(iw, f, s, f / 2);

        float flo = 2.0 * batch * oc * ho * wo * ic * f * f / (1e12);

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        bencher.proxy()->target_algo_info.reset();
        auto time_in_ms = bencher.execs({src, filter, {}}) / RUNS;

        bencher_conv.set_param(conv_param);
        bencher_conv.proxy()->target_algo_info.reset();
        auto time_in_ms_conv =
                bencher_conv.execs({src, {oc, ic, f, f}, {}}) / RUNS;

        printf("src=%s, filter=%s, float32: %.2fms %.2fTFlops, "
               "conv(float32): %.2fms %.2fTFlops, local_share/conv=%.2f\n",
               src.to_string().c_str(), filter.to_string().c_str(), time_in_ms,
               (flo / (time_in_ms * 1e-3)), time_in_ms_conv,
               (flo / (time_in_ms_conv * 1e-3)), time_in_ms / time_in_ms_conv);

    };
    // stride = 1
    run(32, 128, 24, 24, 128, 1, 1, 3);
    run(32, 256, 12, 12, 256, 1, 1, 3);

    // stride = 2
    run(32, 256, 12, 12, 512, 1, 2, 3);
    run(32, 512, 6, 6, 1024, 1, 2, 3);

    // stride = 1
    run(32, 128, 24, 24, 128, 3, 1, 3);
    run(32, 256, 12, 12, 256, 3, 1, 3);

    // stride = 2
    run(32, 128, 24, 24, 256, 3, 2, 3);
    run(32, 256, 12, 12, 512, 3, 2, 3);

    // stride = 1
    run(64, 128, 24, 24, 128, 1, 1, 3);
    run(64, 256, 12, 12, 256, 1, 1, 3);

    // stride = 2
    run(64, 256, 12, 12, 512, 1, 2, 3);
    run(64, 512, 6, 6, 1024, 1, 2, 3);

    // stride = 1
    run(64, 128, 24, 24, 128, 3, 1, 3);
    run(64, 256, 12, 12, 256, 3, 1, 3);

    // stride = 2
    run(64, 128, 24, 24, 256, 3, 2, 3);
    run(64, 256, 12, 12, 512, 3, 2, 3);
}

TEST_F(CUDA, BENCHMARK_LOCAL_SHARE_FORWARD_FROM_RESEARCH) {
    CUBenchmarker<LocalShare> bencher(handle_cuda());
    CUBenchmarker<Convolution> bencher_conv(handle_cuda());
    size_t RUNS = 1000;
    bencher.set_display(false).set_times(RUNS);
    std::unique_ptr<OprProxy<LocalShareForward>> proxy{
            new OprProxy<LocalShareForward>{true}};
    bencher.set_proxy(proxy);

    bencher_conv.set_display(false).set_times(RUNS);
    std::unique_ptr<OprProxy<Convolution>> conv_proxy{
            new OprProxy<Convolution>{true}};
    bencher_conv.set_proxy(conv_proxy);

    LocalShare::Param param;
    Convolution::Param conv_param;
    NormalRNG rng;

    auto run = [&](size_t batch, size_t ic, size_t ih, size_t iw, size_t oc,
                   size_t f, size_t s, size_t sg) {
        param.pad_h = f / 2;
        param.pad_w = f / 2;
        param.stride_h = s;
        param.stride_w = s;
        param.spatial_groups_h = sg;
        param.spatial_groups_w = sg;

        conv_param.pad_h = f / 2;
        conv_param.pad_w = f / 2;
        conv_param.stride_h = s;
        conv_param.stride_w = s;

        TensorShape src = {batch, ic, ih, iw}, filter = {sg, sg, ic, f, f, oc};
        size_t ho = infer_conv_shape(ih, f, s, f / 2);
        size_t wo = infer_conv_shape(iw, f, s, f / 2);

        float flo = 2.0 * batch * oc * ho * wo * ic * f * f / (1e12);

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        bencher.proxy()->target_algo_info.reset();
        auto time_in_ms = bencher.execs({src, filter, {}}) / RUNS;

        bencher_conv.set_param(conv_param);
        bencher_conv.proxy()->target_algo_info.reset();
        auto time_in_ms_conv =
                bencher_conv.execs({src, {oc, ic, f, f}, {}}) / RUNS;

        printf("src=%s, filter=%s, float32: %.2fms %.2fTFlops, "
               "conv(float32): %.2fms %.2fTFlops, local_share/conv=%.2f\n",
               src.to_string().c_str(), filter.to_string().c_str(), time_in_ms,
               (flo / (time_in_ms * 1e-3)), time_in_ms_conv,
               (flo / (time_in_ms_conv * 1e-3)), time_in_ms / time_in_ms_conv);

    };
    // stride = 1
    run(64, 128, 24, 24, 128, 1, 1, 3);
    run(64, 256, 12, 12, 256, 1, 1, 3);
    run(64, 512, 6, 6, 512, 1, 1, 3);
    run(64, 1024, 3, 3, 1024, 1, 1, 3);

    // stride = 2
    run(64, 128, 24, 24, 256, 1, 2, 3);
    run(64, 256, 12, 12, 512, 1, 2, 3);
    run(64, 512, 6, 6, 1024, 1, 2, 3);

    // stride = 1
    run(64, 128, 24, 24, 128, 3, 1, 3);
    run(64, 256, 12, 12, 256, 3, 1, 3);
    run(64, 512, 6, 6, 512, 3, 1, 3);
    run(64, 1024, 3, 3, 1024, 3, 1, 3);

    // stride = 2
    run(64, 128, 24, 24, 256, 3, 2, 3);
    run(64, 256, 12, 12, 512, 3, 2, 3);
    run(64, 512, 6, 6, 1024, 3, 2, 3);
}

TEST_F(CUDA, BENCHMARK_LOCAL_SHARE_FORWARD) {
    require_compute_capability(6, 0);
    CUBenchmarker<LocalShare> bencher(handle_cuda());
    CUBenchmarker<Convolution> bencher_conv(handle_cuda());
    size_t RUNS = 200;
    bencher.set_display(false).set_times(RUNS);
    std::unique_ptr<OprProxy<LocalShareForward>> proxy{
            new OprProxy<LocalShareForward>{true}};
    bencher.set_proxy(proxy);

    bencher_conv.set_display(false).set_times(RUNS);
    std::unique_ptr<OprProxy<Convolution>> conv_proxy{
            new OprProxy<Convolution>{true}};
    bencher_conv.set_proxy(conv_proxy);

    LocalShare::Param param;
    Convolution::Param conv_param;
    NormalRNG rng;

    auto run = [&](size_t batch, size_t ic, size_t ih, size_t iw, size_t oc,
                   size_t f, size_t s, size_t sg) {
        param.pad_h = f / 2;
        param.pad_w = f / 2;
        param.stride_h = s;
        param.stride_w = s;
        param.spatial_groups_h = sg;
        param.spatial_groups_w = sg;

        conv_param.pad_h = f / 2;
        conv_param.pad_w = f / 2;
        conv_param.stride_h = s;
        conv_param.stride_w = s;

        TensorShape src = {batch, ic, ih, iw}, filter = {sg, sg, ic, f, f, oc};
        size_t ho = infer_conv_shape(ih, f, s, f / 2);
        size_t wo = infer_conv_shape(iw, f, s, f / 2);

        float flo = 2.0 * batch * oc * ho * wo * ic * f * f / (1e12);

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        bencher.proxy()->target_algo_info.reset();
        auto time_in_ms = bencher.execs({src, filter, {}}) / RUNS;

        bencher_conv.set_param(conv_param);
        bencher_conv.proxy()->target_algo_info.reset();
        auto time_in_ms_conv =
                bencher_conv.execs({src, {oc, ic, f, f}, {}}) / RUNS;

        printf("src=%s, filter=%s, float32: %.2fms %.2fTFlops, "
               "conv(float32): %.2fms %.2fTFlops, local_share/conv=%.2f\n",
               src.to_string().c_str(), filter.to_string().c_str(), time_in_ms,
               (flo / (time_in_ms * 1e-3)), time_in_ms_conv,
               (flo / (time_in_ms_conv * 1e-3)), time_in_ms / time_in_ms_conv);

    };
    run(64, 256, 48, 48, 256, 7, 1, 3);
    run(64, 128, 24, 24, 128, 7, 1, 3);
    run(64, 256, 12, 12, 256, 7, 1, 3);
    run(64, 512, 6, 6, 512, 7, 1, 3);

    run(64, 256, 48, 48, 256, 5, 1, 3);
    run(64, 128, 24, 24, 128, 5, 1, 3);
    run(64, 256, 12, 12, 256, 5, 1, 3);
    run(64, 512, 6, 6, 512, 5, 1, 3);

    run(32, 64, 96, 96, 256, 7, 2, 3);
    run(32, 128, 24, 24, 128, 7, 2, 3);
    run(32, 256, 12, 12, 256, 7, 2, 3);

    run(32, 64, 96, 96, 256, 5, 2, 3);
    run(32, 128, 24, 24, 128, 5, 2, 3);
    run(32, 256, 12, 12, 256, 5, 2, 3);
}
#endif

// vim: syntax=cpp.doxygen
