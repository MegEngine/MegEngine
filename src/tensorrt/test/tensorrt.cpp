/**
 * \file src/tensorrt/test/tensorrt.cpp
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
#include "megbrain/opr/basic_arith.h"

#if MGB_ENABLE_TENSOR_RT

#include "megbrain/tensorrt/tensorrt_opr.h"
#include "make_trt_net.h"

#include <random>

using namespace mgb;
using namespace nvinfer1;
using namespace opr;

TEST(TestOprTensorRT, Profile) {
    REQUIRE_GPU(1);
    intl::ConcatConvTensorRTNetwork net;

    auto p = net.create_trt_network(true);

    auto y2 = TensorRTOpr::make(TensorRTOpr::to_shared_ptr_builder(p.first),
                                TensorRTOpr::to_shared_ptr_network(p.second),
                                intl::TensorRTGraphFeatureBits::NCHW_FLOAT, {},
                                {net.x0, net.x1})[0];

    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = net.graph->compile({make_callback_copy(net.y, host_z1),
                                    make_callback_copy(y2, host_z2)});
    {
        mgb::GraphProfiler profiler(net.graph.get());

        func->execute();

        profiler.to_json()->writeto_fpath(
                output_file("TestOprTensorRT.Profile.FromProfiler.json"));
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

TEST(TestOprTensorRT, Basic) {
    REQUIRE_GPU(1);
    intl::SimpleTensorRTNetwork net;

    auto p = net.create_trt_network(true);
    auto trt_net =
            TensorRTOpr::to_shared_ptr_network(p.second);
    auto y2 = TensorRTOpr::make(
            TensorRTOpr::to_shared_ptr_builder(p.first), trt_net,
            intl::TensorRTGraphFeatureBits::NCHW_FLOAT, {}, {net.x})[0];

    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = net.graph->compile({make_callback_copy(net.y, host_z1),
                                    make_callback_copy(y2, host_z2)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);

    auto&& host_x = net.host_x;
    auto&& gen = net.gen;

    *host_x = *gen({1, 23, 43, 43});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);
    *host_x = *gen({10, 23, 12, 12});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-3);

    *host_x = *gen({10, 23, 12, 12});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-3);

    // write to file so in python test we can have an engine file
    TensorRTUniquePtr<IBuilder> builder{
            createInferBuilder(TensorRTOpr::Logger::instance()), {}};
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
    FILE* fout = fopen(output_file("trt_cuda_engine_test").c_str(), "wb");
    auto wr = fwrite(mem->data(), 1, mem->size(), fout);
    mgb_assert(wr == mem->size());
    fclose(fout);
    debug::write_to_file(output_file("trt_cuda_engine_test.input").c_str(),
                         debug::dump_tensor(*host_x, "x"));
    debug::write_to_file(output_file("trt_cuda_engine_test.output").c_str(),
                         debug::dump_tensor(host_z1, "x"));
}

TEST(TestOprTensorRT, QuantizedBasic) {
    REQUIRE_GPU(1);
    intl::SimpleQuantizedTensorRTNetwork net;
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    auto p = net.create_trt_network(true);
    auto trt_net =
            TensorRTOpr::to_shared_ptr_network(p.second);

    auto y2 = TensorRTOpr::make(
            TensorRTOpr::to_shared_ptr_builder(p.first), trt_net,
            intl::TensorRTGraphFeatureBits::NCHW4_QINT8, {}, {net.x})[0];
    y2 = opr::TypeCvt::make(y2, dtype::QuantizedS8(1.1f));
    y2 = opr::TypeCvt::make(y2, dtype::Float32());

    HostTensorND host_z_mgb_fp32;
    HostTensorND host_z_mgb_qint8;
    HostTensorND host_z_trt;
    auto func = net.graph->compile(
            {make_callback_copy(net.quantized_y, host_z_mgb_qint8),
             make_callback_copy(net.y, host_z_mgb_fp32),
             make_callback_copy(y2, host_z_trt)});

    mgb::GraphProfiler profiler(net.graph.get());

    func->execute();

    profiler.to_json()->writeto_fpath(
            output_file("TestOprTensorRT.QuantizedBasic.json"));

    MGB_ASSERT_TENSOR_NEAR(host_z_mgb_qint8, host_z_trt, 1e-5);
}


TEST(TestOprTensorRT, ConcatBasic) {
    REQUIRE_GPU(1);
    intl::ConcatConvTensorRTNetwork net;

    auto p = net.create_trt_network(true);
    auto y2 = TensorRTOpr::make(TensorRTOpr::to_shared_ptr_builder(p.first),
                                TensorRTOpr::to_shared_ptr_network(p.second),
                                intl::TensorRTGraphFeatureBits::NCHW_FLOAT, {},
                                {net.x0, net.x1})[0];

    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = net.graph->compile({make_callback_copy(net.y, host_z1),
                                    make_callback_copy(y2, host_z2)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);

    auto&& host_x0 = net.host_x0;
    auto&& host_x1 = net.host_x1;
    auto&& gen = net.gen;

    *host_x0 = *gen({5, 23, 18, 28});
    *host_x1 = *gen({5, 23, 18, 28});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);
}



#endif  // MGB_ENABLE_TENSOR_RT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
