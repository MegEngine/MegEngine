/**
 * \file dnn/test/cpu/mask_conv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/cpu/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/mask_conv.h"
#include "test/common/rng.h"
#include "test/common/utils.h"

using namespace megdnn;
using namespace test;

TEST_F(CPU, MASK_CONV) {
    mask_conv_test(handle());
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CPU, MASK_CONV_BENCHMARK) {
    mask_conv_benchmark(handle());
}
#endif

TEST_F(CPU, MASK_PROPAGATE) {
    param::MaskPropagate mask_param;
    auto mask_check = [&](const TensorNDArray& tensors) {
        auto mask_src = tensors[0];
        auto mask_dst = tensors[1];

        auto src_ptr = static_cast<float*>(megdnn_malloc(
                handle(), mask_src.layout.total_nr_elems() * sizeof(float)));
        auto src = TensorND{
                src_ptr,
                TensorLayout{mask_src.layout.reshape({1, 1, mask_src.layout[0],
                                                      mask_src.layout[1]}),
                             dtype::Float32()}};
        for (size_t i = 0; i < src.layout.total_nr_elems(); ++i) {
            src_ptr[i] = static_cast<float>(mask_src.ptr<int>()[i]);
        }

        auto filter_ptr = static_cast<float*>(megdnn_malloc(
                handle(),
                mask_param.kernel_h * mask_param.kernel_w * sizeof(float)));
        auto filter = TensorND{
                static_cast<void*>(filter_ptr),
                TensorLayout{{1, 1, mask_param.kernel_h, mask_param.kernel_w},
                             dtype::Float32()}};
        for (size_t i = 0; i < mask_param.kernel_h * mask_param.kernel_w; ++i) {
            filter_ptr[i] = 1.0;
        }

        TensorLayout dst_layout{dtype::Float32()};

        param::Convolution conv_param{
                param::Convolution::Mode::CROSS_CORRELATION,
                mask_param.pad_h,
                mask_param.pad_w,
                mask_param.stride_h,
                mask_param.stride_w,
                mask_param.dilate_h,
                mask_param.dilate_w};
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = conv_param;
        opr->deduce_layout(src.layout, filter.layout, dst_layout);
        auto dst_ptr = static_cast<float*>(megdnn_malloc(
                handle(), mask_dst.layout.total_nr_elems() * sizeof(float)));
        auto dst = TensorND{dst_ptr, dst_layout};
        WorkspaceWrapper workspace{
                handle(), opr->get_workspace_in_bytes(src.layout, filter.layout,
                                                      dst.layout, nullptr)};
        opr->exec(src, filter, dst, nullptr, workspace.workspace());
        for (size_t i = 0; i < dst.layout.total_nr_elems(); ++i) {
            mask_dst.ptr<int>()[i] = dst_ptr[i] > 0;
        }
        delete dst_ptr;
        delete filter_ptr;
        delete src_ptr;
    };

    Checker<MaskPropagate> checker(handle());
    auto rng = std::make_unique<BernoulliRNG>(0.5);
    checker.set_extra_opr_impl(mask_check)
            .set_dtype(0, dtype::Int32())
            .set_rng(0, rng.get());

    auto run = [&](size_t IH, size_t IW, size_t FH, size_t FW, size_t SH = 1,
                   size_t SW = 1, size_t PH = 0, size_t PW = 0, size_t DH = 1,
                   size_t DW = 1) {
        mask_param.kernel_h = FH;
        mask_param.kernel_w = FW;
        mask_param.pad_h = PH;
        mask_param.pad_w = PW;
        mask_param.stride_h = SH;
        mask_param.stride_w = SW;
        mask_param.dilate_h = DH;
        mask_param.dilate_w = DW;
        checker.set_param(mask_param);

        TensorShape src_shape{IH, IW}, dst_shape{};

        checker.execs({src_shape, dst_shape});
    };
    run(5, 5, 3, 2);
    run(5, 5, 2, 3, 2, 2);
    run(5, 5, 3, 3, 2, 2, 1, 2);
    run(5, 5, 3, 3, 2, 1, 1, 2);
    run(5, 5, 3, 3, 1, 2, 2, 2);
    run(24, 23, 4, 4, 1, 1, 3, 2);
    run(24, 23, 4, 4, 1, 1, 3, 2, 2, 2);
    run(24, 23, 4, 4, 1, 1, 3, 2, 2, 3);
    run(24, 23, 4, 4, 1, 1, 3, 2, 3, 3);
}

// vim: syntax=cpp.doxygen
