/**
 * \file src/opr/test/atlas_runtime_op.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/tensor_manip.h"
#include "megdnn/dtype.h"
#if MGB_ATLAS

#include "megbrain/comp_node_env.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/test/helper.h"

#include "megbrain/opr/atlas_runtime_op.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/plugin/profiler.h"

#include <random>
#include <vector>
#include <stdio.h>

#include "./atlas_models.h"

using namespace mgb;
using namespace opr;
using namespace serialization;

TEST(TestOprAtlas, Basic) {
    HostTensorGenerator<> gen;
    const auto& graph = ComputingGraph::make();
    const auto& host_x = gen({4, 3, 16, 16});

    //! run om model
    const auto& om_buffer = ATLAS_MODEL.at("model_om");
    auto cn = CompNode::load("atlas0");
    auto x = Host2DeviceCopy::make(*graph, host_x, cn);
    auto y = opr::AtlasRuntimeOpr::make(om_buffer.first, om_buffer.second,
                                        {x})[0];
    HostTensorND host_om;
    auto om_func = graph->compile({make_callback_copy(y, host_om, true)});
    om_func->execute().wait();

    //! run mdl model
    const auto& mdl_buffer = ATLAS_MODEL.at("model_mdl");
    auto loader = GraphLoader::make(
            InputFile::make_mem_proxy(mdl_buffer.first, mdl_buffer.second));
    auto rst = loader->load();
    auto input = rst.tensor_map.at("d");
    input->copy_from(*host_x).sync();
    HostTensorND host_mdl;
    auto mgb_func = rst.graph_compile(
            {make_callback_copy(rst.output_var_list[0], host_mdl)});
    mgb_func->execute().wait();

    //! In atlas, the inner compute is fp16
    MGB_ASSERT_TENSOR_NEAR(host_mdl, host_om, 1e-3);
}

TEST(TestOprAtlas, DynamicBatch) {
    for (size_t batch : {1, 6, 20}) {
        HostTensorGenerator<> gen;
        const auto& graph = ComputingGraph::make();
        const auto& host_x = gen({batch, 3, 16, 16});

        //! run om model
        const auto& om_buffer = ATLAS_MODEL.at("model_dyn_om");
        auto cn = CompNode::load("atlas0");
        auto x = Host2DeviceCopy::make(*graph, host_x, cn);
        auto y = opr::AtlasRuntimeOpr::make(om_buffer.first, om_buffer.second,
                                            {x})[0];
        HostTensorND host_om;
        auto om_func = graph->compile({make_callback_copy(y, host_om, true)});
        om_func->execute().wait();

        //! run mdl model
        const auto& mdl_buffer = ATLAS_MODEL.at("model_mdl");
        auto loader = GraphLoader::make(
                InputFile::make_mem_proxy(mdl_buffer.first, mdl_buffer.second));
        auto rst = loader->load();
        auto input = rst.tensor_map.at("d");
        input->copy_from(*host_x).sync();
        HostTensorND host_mdl;
        auto mgb_func = rst.graph_compile(
                {make_callback_copy(rst.output_var_list[0], host_mdl)});
        mgb_func->execute().wait();

        //! In atlas, the inner compute is fp16
        MGB_ASSERT_TENSOR_NEAR(host_mdl, host_om, 1e-3);
    }
}

TEST(TestOprAtlas, Rgb888) {
    HostTensorGenerator<dtype::Uint8, RandomDistribution::UNIFORM> gen;
    const auto& graph = ComputingGraph::make();
    const auto &host_x = gen({1, 3, 16, 16});

    //! run om model
    const auto& om_buffer = ATLAS_MODEL.at("model_rgb_om");
    auto x = Host2DeviceCopy::make(*graph, host_x);
    x = opr::Dimshuffle::make(x, {0, 2, 3, 1});
    auto cn = CompNode::load("atlas0");
    auto atlas_x = Copy::make(x, {cn});
    auto y = opr::AtlasRuntimeOpr::make(om_buffer.first, om_buffer.second,
                                        {atlas_x})[0];
    HostTensorND host_om;
    auto om_func = graph->compile({make_callback_copy(y, host_om, true)});
    om_func->execute().wait();

    //! run mdl model
    const auto& mdl_buffer = ATLAS_MODEL.at("model_aipp_mdl");
    auto loader = GraphLoader::make(
            InputFile::make_mem_proxy(mdl_buffer.first, mdl_buffer.second));
    auto rst = loader->load();
    auto input = rst.tensor_map.at("d");
    input->copy_from(*host_x).sync();
    HostTensorND host_mdl;
    auto mgb_func = rst.graph_compile(
            {make_callback_copy(rst.output_var_list[0], host_mdl)});
    mgb_func->execute().wait();

    //! In atlas, the inner compute is fp16
    MGB_ASSERT_TENSOR_NEAR(host_mdl,
                           host_om, 1e-3);
}

TEST(TestOprAtlas, Yuv) {
    //! As YUV420SP depends on the input processed by AIPP, so here we just
    //! check if the shape satisfy.
    HostTensorGenerator<dtype::Uint8, RandomDistribution::UNIFORM> gen;
    const auto& graph = ComputingGraph::make();
    const auto &host_x = gen({1, 24, 16, 1});

    //! run om model
    const auto& om_buffer = ATLAS_MODEL.at("model_yuv_om");
    auto cn = CompNode::load("atlas0");
    auto x = Host2DeviceCopy::make(*graph, host_x, cn);
    auto y = opr::AtlasRuntimeOpr::make(om_buffer.first, om_buffer.second,
                                        {x})[0];
    HostTensorND host_om;
    auto om_func = graph->compile({make_callback_copy(y, host_om, true)});
    om_func->execute().wait();
}

TEST(TestOprAtlas, Serialization) {
    using namespace serialization;

    HostTensorGenerator<> gen;
    const auto& graph = ComputingGraph::make();
    const auto& host_x = gen({4, 3, 16, 16});

    const auto& om_buffer = ATLAS_MODEL.at("model_om");
    auto cn = CompNode::load("atlas0");
    auto x = Host2DeviceCopy::make(*graph, host_x, cn);
    auto y = opr::AtlasRuntimeOpr::make(om_buffer.first, om_buffer.second,
                                        {x})[0];

    auto fname = output_file("AtlasRuntimeOprTest");
    auto dump = [&]() {
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()));
        auto rst = dumper->dump({y});
        ASSERT_EQ(rst.outputs.size(), 1u);
    };
    auto load = [&]() {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()));
        auto rst = loader->load();
        ASSERT_EQ(rst.output_var_list.size(), 1u);
    };
    dump();
    load();
}

TEST(TestOprAtlas, Profiling) {
    HostTensorGenerator<> gen;
    const auto& graph = ComputingGraph::make();
    GraphProfiler profiler{graph.get()};
    const auto& host_x = gen({1, 3, 16, 16});

    //! run om model
    const auto& om_buffer = ATLAS_MODEL.at("model_dyn_om");
    auto cn = CompNode::load("atlas0");
    auto x = Host2DeviceCopy::make(*graph, host_x, cn);
    auto y = opr::AtlasRuntimeOpr::make(om_buffer.first, om_buffer.second,
                                        {x})[0];
    HostTensorND host_om;
    auto om_func = graph->compile({make_callback_copy(y, host_om, true)});
    om_func->execute().wait();

    profiler.to_json_full(om_func.get())
            ->writeto_fpath(output_file("atlas_runtime_opr_profile.json"));
}

#endif  // MGB_ATLAS

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
