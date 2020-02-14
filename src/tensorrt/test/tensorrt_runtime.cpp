/**
 * \file src/tensorrt/test/tensorrt_runtime.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/comp_node_env.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/megdnn_helper.h"
#include "megbrain/utils/debug.h"

#if MGB_ENABLE_TENSOR_RT

#include "megbrain/tensorrt/tensorrt_opr.h"
#include "megbrain/tensorrt/tensorrt_runtime_opr.h"
#include "make_trt_net.h"

#include <random>

using namespace mgb;
using namespace nvinfer1;

template <typename T>
using TensorRTUniquePtr = intl::TensorRTUniquePtr<T>;



TEST(TestOprTensorRT, RuntimeBasic) {
    REQUIRE_GPU(1);
    intl::SimpleTensorRTNetwork net;
    auto make_trt = [&net]() {
        auto p = net.create_trt_network(false);
        TensorRTUniquePtr<INetworkDefinition> trt_net{p.second, {}};
        TensorRTUniquePtr<IBuilder> builder{p.first, {}};
        builder->setMaxBatchSize(5);
#if NV_TENSOR_RT_VERSION >= 6001
        TensorRTUniquePtr<IBuilderConfig> build_config{
                builder->createBuilderConfig()};
        TensorRTUniquePtr<ICudaEngine> cuda_engine{
                builder->buildEngineWithConfig(*trt_net, *build_config)};
#else
        TensorRTUniquePtr<ICudaEngine> cuda_engine{
                builder->buildCudaEngine(*trt_net)};
#endif
        TensorRTUniquePtr<IHostMemory> mem{cuda_engine->serialize(), {}};
        return TensorRTRuntimeOpr::make(mem->data(), mem->size(), {net.x})[0];
    };
    auto y2 = make_trt();

    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = net.graph->compile({make_callback_copy(net.y, host_z1),
                                    make_callback_copy(y2, host_z2)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);
}



TEST(TestOprTensorRT, ConcatRuntimeBasic) {
    REQUIRE_GPU(1);
    intl::ConcatConvTensorRTNetwork net;

    SymbolVar y2;
    {
        auto p = net.create_trt_network(false);
        TensorRTUniquePtr<INetworkDefinition> trt_net{p.second, {}};
        TensorRTUniquePtr<IBuilder> builder{p.first, {}};
        builder->setMaxBatchSize(5);
#if NV_TENSOR_RT_VERSION >= 6001
        TensorRTUniquePtr<IBuilderConfig> build_config{
                builder->createBuilderConfig()};
        auto cuda_engine =
                builder->buildEngineWithConfig(*trt_net, *build_config);
#else
        auto cuda_engine = builder->buildCudaEngine(*trt_net);
#endif
        TensorRTUniquePtr<IHostMemory> mem{cuda_engine->serialize(), {}};

        FILE* fout = fopen(output_file("trt_cuda_engine").c_str(), "wb");
        auto wr = fwrite(mem->data(), 1, mem->size(), fout);
        mgb_assert(wr == mem->size());
        fclose(fout);

        y2 = TensorRTRuntimeOpr::make(
                TensorRTRuntimeOpr::to_shared_ptr_engine(cuda_engine), {},
                {net.x0, net.x1})[0];
    }

    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = net.graph->compile({make_callback_copy(net.y, host_z1),
                                    make_callback_copy(y2, host_z2)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);
}

TEST(TestOprTensorRT, RuntimeProfile) {
    REQUIRE_GPU(1);
    intl::ConcatConvTensorRTNetwork net;
    SymbolVar y2;
    {
        auto p = net.create_trt_network(false);
        TensorRTUniquePtr<INetworkDefinition> trt_net{p.second, {}};
        TensorRTUniquePtr<IBuilder> builder{p.first, {}};
        builder->setMaxBatchSize(5);
#if NV_TENSOR_RT_VERSION >= 6001
        TensorRTUniquePtr<IBuilderConfig> build_config{
                builder->createBuilderConfig()};
        auto cuda_engine =
                builder->buildEngineWithConfig(*trt_net, *build_config);
#else
        auto cuda_engine = builder->buildCudaEngine(*trt_net);
#endif
        TensorRTUniquePtr<IHostMemory> mem{cuda_engine->serialize(), {}};

        FILE* fout = fopen(output_file("trt_cuda_engine").c_str(), "wb");
        auto wr = fwrite(mem->data(), 1, mem->size(), fout);
        mgb_assert(wr == mem->size());
        fclose(fout);

        y2 = TensorRTRuntimeOpr::make(
                TensorRTRuntimeOpr::to_shared_ptr_engine(cuda_engine), {},
                {net.x0, net.x1})[0];
    }

    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = net.graph->compile({make_callback_copy(net.y, host_z1),
                                    make_callback_copy(y2, host_z2)});

    {
        mgb::GraphProfiler profiler(net.graph.get());

        func->execute();

        profiler.to_json()->writeto_fpath(output_file(
                "TestOprTensorRT.RuntimeProfile.FromProfiler.json"));

        auto prof_obj = *static_cast<json::Object*>(profiler.to_json().get());
        auto record_obj =
                *static_cast<json::Object*>(prof_obj["opr_internal_pf"].get());
        auto opr_prof_arr = *static_cast<json::Array*>(
                record_obj[y2.node()->owner_opr()->id_str()].get());
        for (auto item_arr : opr_prof_arr.get_impl()) {
            auto layer_info_arr = *static_cast<json::Array*>(item_arr.get());
            auto layer_time =
                    *static_cast<json::Number*>(layer_info_arr[1].get());

            mgb_assert(layer_time.get_impl() > 0, "Error occured in json.");
        }

        MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);
    }
    // Run it again after profiler is not in existance.
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);
}

TEST(TestOprTensorRT, RuntimeChangeBatchSize) {
    REQUIRE_GPU(1);
    intl::SimpleTensorRTNetwork net;
    auto make_trt = [&net]() {
        auto p = net.create_trt_network(false);
        TensorRTUniquePtr<INetworkDefinition> trt_net{p.second, {}};
        TensorRTUniquePtr<IBuilder> builder{p.first, {}};
        builder->setMaxBatchSize(10);
#if NV_TENSOR_RT_VERSION >= 6001
        TensorRTUniquePtr<IBuilderConfig> build_config{
                builder->createBuilderConfig()};
        TensorRTUniquePtr<ICudaEngine> cuda_engine{
                builder->buildEngineWithConfig(*trt_net, *build_config)};
#else
        TensorRTUniquePtr<ICudaEngine> cuda_engine{
                builder->buildCudaEngine(*trt_net)};
#endif
        TensorRTUniquePtr<IHostMemory> mem{cuda_engine->serialize(), {}};
        return TensorRTRuntimeOpr::make(mem->data(), mem->size(), {net.x})[0];
    };
    auto y2 = make_trt();

    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = net.graph->compile({make_callback_copy(net.y, host_z1),
                                    make_callback_copy(y2, host_z2)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);
    *net.host_x = *net.gen({1, 23, 28, 28});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);
    *net.host_x = *net.gen({10, 23, 28, 28});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);
}

#endif  // MGB_ENABLE_TENSOR_RT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
