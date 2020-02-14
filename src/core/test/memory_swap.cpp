/**
 * \file src/core/test/memory_swap.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/io.h"

using namespace mgb;

using Elemwise = opr::Elemwise;
using Mode = Elemwise::Mode;
#if MGB_ENABLE_MEMORY_SWAP
auto run = [](const int flag) {
    auto KEY = "MGB_MEMORY_SWAP_PARAM_BUCKET_IMPLEMENT";
    auto old_value = getenv(KEY);
    if (flag)
        setenv(KEY, "1", 1);
    else
        setenv(KEY, "0", 1);

    HostTensorGenerator<> gen_;

    auto gen = [&](const TensorShape& shp) { return gen_(shp, "gpu0"); };
    constexpr size_t batch_size = 5, C = 8, H = 100, W = 128;
    constexpr size_t limit = 200;
    auto host_data = gen({batch_size, C, H, W});
    auto graph = ComputingGraph::make();

    SymbolVarArray kernels;
    SymbolVarArray conv_res;
    auto data = opr::Host2DeviceCopy::make(*graph, host_data).rename("data");
    conv_res.push_back(data);

    size_t out_chl = host_data->shape(1), layer_count = 0;

    auto add_layer = [&](size_t oc, size_t kernal_shape, size_t padding) {
        gen_.std(sqrt(2.0 / (out_chl * kernal_shape * kernal_shape)));
        auto host_kern = gen({oc, out_chl, kernal_shape, kernal_shape});
        auto dev_kern = std::make_shared<DeviceTensorND>();
        dev_kern->copy_from(*host_kern);
        auto current_param = opr::Convolution::Param();
        kernels.emplace_back(opr::SharedDeviceTensor::make(*graph, dev_kern));
        current_param.pad_h = current_param.pad_w = padding;
        conv_res.push_back(opr::relu(opr::Convolution::make(
                conv_res[layer_count],
                kernels.back().rename(ssprintf("param%zu", layer_count)),
                current_param)));
        layer_count++;
        out_chl = oc;
    };

    for (size_t i = 1; i <= limit; ++i)
        add_layer(30, 5, 2);

    auto loss = opr::Dot::make(conv_res[limit].flatten(),
                               conv_res[limit].flatten());
    std::vector<HostTensorND> grad_kernels_get(kernels.size());
    ComputingGraph::OutputSpec out_spec;
    for (size_t i = 0; i < kernels.size(); ++i) {
        out_spec.emplace_back(make_callback_copy(cg::grad(loss, kernels[i]),
                                                 grad_kernels_get[i]));
    }
    std::vector<HostTensorND> grad_kernels_expect(grad_kernels_get.size());
    for (bool swap : {false, true}) {
        graph->options().enable_memory_swap = swap;
        auto func = graph->compile(out_spec);
        func->execute();
        if (!swap) {
            for (size_t i = 0; i < grad_kernels_get.size(); ++i)
                grad_kernels_expect[i].copy_from(grad_kernels_get[i]);
        }
    }

    for (size_t i = 0; i < grad_kernels_get.size(); ++i)
        MGB_ASSERT_TENSOR_NEAR(grad_kernels_get[i], grad_kernels_expect[i],
                               1e-3);
    if (old_value) {
        setenv(KEY, old_value, 1);
    } else {
        unsetenv(KEY);
    }
};

TEST(TestMemorySwap, FullConvSerial) {
    REQUIRE_GPU(1);
    run(0);
}

TEST(TestMemorySwap, FullConvParallel) {
    REQUIRE_GPU(1);
    run(0);
}

#endif  // MGB_ENABLE_MEMORY_SWAP

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
