/**
 * \file src/core/test/comp_node_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./comp_node_helper.h"

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/serializer.h"

using namespace mgb;
using namespace comp_node_test;

namespace {

void run_comp_seq_rec_basic(CompNode cn, bool fake_first) {
    using ConvParam = opr::Convolution::Param;
    ConvParam param;
    param.sparse = ConvParam::Sparse::GROUP;
    HostTensorGenerator<> gen;
    auto host_x = gen({3, 4, 10, 8}, cn), host_y = gen({2, 3, 2, 3, 3}, cn);

    int iter = 0;
    std::vector<int> executed;

    HostTensorND host_z;
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         z = opr::CallbackInjector::make(
                 opr::Convolution::make(x, y, param),
                 [&](DeviceTensorND&dv) { executed.push_back(iter); });
    graph->options().comp_node_seq_record_level = 1;
    if (fake_first) {
        graph->options().fake_next_exec = true;
        graph->options().var_sanity_check_first_run = false;
    }
    auto func = graph->compile({make_callback_copy(z, host_z)});
    if (fake_first) {
        func->execute();  // first exec
    }
    int change = 5;
    for (; iter < 10; ++iter) {
        if (iter == change) {
            *host_x = *gen({2, 4, 15, 13}, cn);
        }
        host_x->copy_from_fixlayout(*gen(host_x->shape(), cn));
        func->execute();
        auto expect = eval_conv_cpu<opr::Convolution>(*host_x, *host_y, param);
        MGB_ASSERT_TENSOR_NEAR(expect, host_z, 1e-3) << "iter " << iter;
    }
    ASSERT_EQ(executed.size(), 4u);

    // if fake_first, both warmup exec and exec with recorder will perform in
    // iter0 else, normal exec will perform in iter0 and exec with recorder in
    // iter1
    ASSERT_EQ(executed[0], 0);
    ASSERT_EQ(executed[1], fake_first ? 0 : 1);

    // recorder would be reset, normal exec
    ASSERT_EQ(executed[2], change);
    // create new recorder, exec with recorder
    ASSERT_EQ(executed[3], change + 1);
}

void run_comp_seq_rec_basic_level2(CompNode cn) {
    using ConvParam = opr::ConvBias::Param;
    ConvParam param;
    param.sparse = ConvParam::Sparse::GROUP;
    HostTensorGenerator<> gen;
    auto host_x = gen({3, 4, 10, 8}, cn), host_y = gen({2, 3, 2, 3, 3}, cn);

    int iter = 0;
    std::vector<int> executed;

    HostTensorND host_z;
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         z = opr::CallbackInjector::make(
                 opr::ConvBias::make(x, y, param),
                 [&](DeviceTensorND&dv) { executed.push_back(iter); });
    graph->options().comp_node_seq_record_level = 2;
    graph->options().var_sanity_check_first_run = false;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    ComputingGraph::assert_destroy(graph);
    for (; iter < 10; ++iter) {
        host_x->copy_from_fixlayout(*gen(host_x->shape(), cn));
        func->execute();
        auto expect = eval_conv_cpu<opr::ConvBias>(*host_x, *host_y, param);
        MGB_ASSERT_TENSOR_NEAR(expect, host_z, 1e-3) << "iter " << iter;
    }
    ASSERT_EQ(executed.size(), 2u);

    //! test default_cpu with record2
    {
        HostTensorND hz;
        graph = ComputingGraph::make();
        x = opr::Host2DeviceCopy::make(*graph, host_x);
        y = opr::Host2DeviceCopy::make(*graph, host_y);
        z = opr::ConvBias::make(x, y, param);
        z = opr::GetVarShape::make(z);
        graph->options().comp_node_seq_record_level = 2;
        graph->options().var_sanity_check_first_run = false;
        auto func = graph->compile({make_callback_copy(z, hz, true)});
        ComputingGraph::assert_destroy(graph);
        func->execute();
        ASSERT_TRUE(hz.comp_node() == cn);
        ASSERT_EQ(hz.ptr<int>()[0], 3);
        ASSERT_EQ(hz.ptr<int>()[1], 6);
        ASSERT_EQ(hz.ptr<int>()[2], 8);
        ASSERT_EQ(hz.ptr<int>()[3], 6);
    }
}

void run_comp_seq_rec_dyn_elemwise(CompNode cn, bool fake_first) {
    // dynamic memory is allocated in elemwise
    HostTensorGenerator<> gen;
    auto host_x = gen({3, 3}, cn), host_y = gen({1, 3}, cn), host_z = gen({3, 1}, cn);

    auto check = [&]() {
        HostTensorND ret(CompNode::load("cpux"), host_x->shape());
        auto px = host_x->ptr<float>(), py = host_y->ptr<float>(),
             pz = host_z->ptr<float>(), pw = ret.ptr<float>();
        auto sz0 = host_x->shape()[0], sz1 = host_x->shape()[1];
        for (size_t i = 0; i < sz0; ++i) {
            for (size_t j = 0; j < sz1; ++j) {
                pw[i * sz1 + j] = px[i * sz1 + j] * py[j] + pz[i];
            }
        }
        return ret;
    };

    auto graph = ComputingGraph::make();
    // test record on first run
    graph->options().var_sanity_check_first_run = false;
    graph->options().graph_opt_level = 0;
    graph->options().comp_node_seq_record_level = 1;
    if (fake_first) {
        graph->options().fake_next_exec = true;
    }
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         z = opr::Host2DeviceCopy::make(*graph, host_z),
         w = opr::Elemwise::make({x, y, z}, opr::Elemwise::Mode::FUSE_MUL_ADD3);

    HostTensorND host_w;
    auto func = graph->compile({make_callback_copy(w, host_w)});
    if (fake_first) {
        func->execute();
    }
    for (int i = 0; i < 10; ++i) {
        if (i == 5) {
            *host_x = *gen({10, 8}, cn);
            *host_y = *gen({1, 8}, cn);
            *host_z = *gen({10, 1}, cn);
        }
        host_x->copy_from(*gen(host_x->shape(), cn));
        func->execute();
        auto expect = check();
        MGB_ASSERT_TENSOR_EQ(expect, host_w) << "iter " << i;
    }
}

void run_level2(CompNode cn, bool use_multi_holder) {
    HostTensorGenerator<> gen;
    auto host_x = gen({4, 3, 6, 7}, cn), host_w = gen({2, 3, 2, 3}, cn),
         host_y = gen({1, 25}, cn), host_z = gen({8, 1}, cn),
         host_large = gen({8, 25}, cn);
    auto make_func = [&](bool enable) -> thin_function<const HostTensorND&()> {
        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = 0;
        if (enable) {
            graph->options().var_sanity_check_first_run = false;
            graph->options().comp_node_seq_record_level = 2;
        }
        auto repeat2 = [](SymbolVar x) { return opr::Concat::make({x, x}, 0); };
        SymbolVar w;
        auto dev_w = std::make_shared<DeviceTensorND>();
        // test shared dev tensor with 1 refcnt
        if (use_multi_holder) {
            dev_w->copy_from(*host_w).sync();
            w = opr::MultipleDeviceTensorHolder::make(*graph, {dev_w})[0];
        } else {
            w = opr::SharedDeviceTensor::make(*graph, *host_w);
        }

        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             // test shared dev tensor with 1 refcnt
                c = opr::Convolution::make(x, w).reshape({8, 25}),
             y = opr::Host2DeviceCopy::make(*graph, host_y),
             large = opr::ImmutableTensor::make(*graph, *host_large),
             z = opr::Host2DeviceCopy::make(*graph, host_z),
             // elemwise with larger tmp storage
                t0 = opr::Elemwise::make(
                             {c, y, z}, opr::Elemwise::Mode::FUSE_MUL_ADD3) +
                     large,
             // t1 shape is {8, 1}
                t1 = opr::reduce_sum(t0, z.symshape()),
             t2 = opr::Elemwise::make(
                     {repeat2(c), y, repeat2(t1)}, opr::Elemwise::Mode::FUSE_MUL_ADD3),
             large1 = opr::ImmutableTensor::make(*graph, *host_large);
        t2 * 2;  // unused opr

        // used large static infer
        graph->static_infer_manager().infer_value(large.node());

        // unused large static infer
        graph->static_infer_manager().infer_value(large1.node());

        // static infer value
        graph->static_infer_manager().infer_value((t1.symshape() + 1).node());

        auto result = std::make_shared<HostTensorND>();
        auto func = graph->compile({make_callback_copy(t2, *result)});
        std::shared_ptr<cg::AsyncExecutable> sh_func(func.release());
        if (enable) {
            ComputingGraph::assert_destroy(graph);
        }
        auto exec = [result, sh_func]() -> const HostTensorND& {
            sh_func->execute();
            return *result;
        };
        return exec;
    };

    auto f0 = make_func(false), f1 = make_func(true);
    for (int i = 0; i < 3; ++i) {
        host_x->copy_from(*gen(host_x->shape(), cn));
        host_y->copy_from(*gen(host_y->shape(), cn));
        host_z->copy_from(*gen(host_z->shape(), cn));
        auto&& expect = f0();
        auto&& get = f1();
        MGB_ASSERT_TENSOR_EQ(expect, get);
    }

    host_x->resize({1});
    ASSERT_THROW(f1(), MegBrainError);
}

}  // anonymous namespace

namespace mgb {
namespace comp_node_test {
namespace seq_rec {

template <>
void run<basic>(CompNode cn) {
    run_comp_seq_rec_basic(cn, false);
}

template <>
void run<basic_level2>(CompNode cn) {
    run_comp_seq_rec_basic_level2(cn);
}

template <>
void run<basic_fake_exec>(CompNode cn) {
    run_comp_seq_rec_basic(cn, true);
}

template <>
void run<dyn_elemwise>(CompNode cn) {
    run_comp_seq_rec_dyn_elemwise(cn, false);
}

template <>
void run<dyn_elemwise_fake_exec>(CompNode cn) {
    run_comp_seq_rec_dyn_elemwise(cn, true);
}

template <>
void run<level2>(CompNode cn) {
    run_level2(cn, false);
}

template <>
void run<level2_multi_holder>(CompNode cn) {
    run_level2(cn, true);
}

template <>
void run<level2_share_storage>(CompNode cn) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1}, cn), host_y = gen({1}, cn), host_z = gen({10}, cn);
    auto make_func =
            [&](bool enable) -> thin_function<std::array<const HostTensorND*, 2>()> {
        auto g0 = ComputingGraph::make(), g1 = ComputingGraph::make();
        if (enable) {
            g0->options().var_sanity_check_first_run = false;
            g0->options().comp_node_seq_record_level = 2;
            g1->options().var_sanity_check_first_run = false;
            g1->options().comp_node_seq_record_level = 2;
            g0->share_device_memory_with(*g1);
        }
        auto x0 = opr::Host2DeviceCopy::make(*g0, host_x),
             x1 = opr::Host2DeviceCopy::make(*g1, host_x),
             y = opr::Host2DeviceCopy::make(*g0, host_y),
             z = opr::Host2DeviceCopy::make(*g1, host_z);
        auto t0 = x0 + y, t1 = x1 + z;

        auto host_t0 = std::make_shared<HostTensorND>(),
             host_t1 = std::make_shared<HostTensorND>();
        auto f0 = g0->compile({make_callback_copy(t0, *host_t0)});
        auto f1 = g1->compile({make_callback_copy(t1, *host_t1)});
        std::shared_ptr<cg::AsyncExecutable> sh_f0(f0.release()), sh_f1(f1.release());
        if (enable) {
            ComputingGraph::assert_destroy(g0);
            ComputingGraph::assert_destroy(g1);
        }
        auto exec = [host_t0, host_t1, sh_f0,
                     sh_f1]() -> std::array<const HostTensorND*, 2> {
            sh_f0->execute();
            sh_f1->execute();
            return {host_t0.get(), host_t1.get()};
        };
        return exec;
    };

    auto f0 = make_func(false), f1 = make_func(true);
    for (int i = 0; i < 3; ++i) {
        host_x->copy_from(*gen(host_x->shape(), cn));
        host_y->copy_from(*gen(host_y->shape(), cn));
        host_z->copy_from(*gen(host_z->shape(), cn));
        auto&& expect = f0();
        auto&& get = f1();
        MGB_ASSERT_TENSOR_EQ(*expect[0], *get[0]);
        MGB_ASSERT_TENSOR_EQ(*expect[1], *get[1]);
    }
}

template <>
void run<level2_exec_check>(CompNode cn) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1}, cn);
    for (int testcase = 0; testcase < 3; ++testcase) {
        host_x->copy_from(*gen({1}));
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x), y = x * 2;
        HostTensorND host_y;
        graph->options().var_sanity_check_first_run = false;
        graph->options().comp_node_seq_record_level = 2;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        ASSERT_EQ(host_y.shape(), host_x->shape());
        auto expect = host_x->ptr<float>()[0] * 2;
        ASSERT_NE(expect, host_y.ptr<float>()[0]);

        if (testcase == 0) {
            ComputingGraph::assert_destroy(graph);
            func->execute();
            ASSERT_EQ(expect, host_y.ptr<float>()[0]);
        } else if (testcase == 1) {
            ASSERT_THROW(func->execute(), MegBrainError);
        } else {
            // it should be OK to destroy func and then graph
            func.reset();
            graph.reset();
        }
    };
}

template <>
void run<sync_from_func>(CompNode cn) {
    REQUIRE_THREAD();
    HostTensorGenerator<> gen;
    auto host_x = gen({1}, cn);
    for (int level : {1, 2}) {
        for (bool sync : {false, true}) {
            auto graph = ComputingGraph::make();
            auto x = opr::Host2DeviceCopy::make(*graph, host_x),
                 y = opr::Sleep::make(x, 0.15) * 2;
            HostTensorND host_y;
            graph->options().var_sanity_check_first_run = false;
            graph->options().comp_node_seq_record_level = level;
            auto cb = [&](const DeviceTensorND& dv) {
                host_y.copy_from(dv);
                if (sync) {
                    host_y.sync();
                }
            };
            auto func = graph->compile({{y, cb}});
            if (level == 2) {
                ComputingGraph::assert_destroy(graph);
            }
            for (int i = 0; i < 3; ++i) {
                host_x->ptr<float>()[0] = i + 0.3;
                func->execute();
                if (!sync) {
                    func->wait();
                }
                auto got = host_y.ptr<float>()[0];
                MGB_ASSERT_FLOAT_EQ((i + 0.3) * 2, got)
                        << "level=" << level << " i=" << i;
            }
        }
    }
}

template <>
void run<cb_non_contig>(CompNode cn) {
    REQUIRE_THREAD();
    HostTensorGenerator<> gen;
    auto host_x = gen({4, 5}, cn);
    for (int level : {1, 2}) {
        for (bool sync : {false, true}) {
            auto graph = ComputingGraph::make();
            auto x = opr::Host2DeviceCopy::make(*graph, host_x),
                 y = opr::Dimshuffle::make(x, {1, 0});
            HostTensorND host_y;
            graph->options().var_sanity_check_first_run = false;
            graph->options().comp_node_seq_record_level = level;
            auto cb = [&](const DeviceTensorND& dv) {
                host_y.copy_from(dv);
                if (sync) {
                    host_y.sync();
                }
            };
            auto func = graph->compile({{y, cb}});
            if (level == 2) {
                ComputingGraph::assert_destroy(graph);
            }
            for (int i = 0; i < 3; ++i) {
                host_x->copy_from(*gen(host_x->shape()));
                HostTensorND expect{host_x->comp_node(), {5, 4}};
                auto px = host_x->ptr<float>(), py = expect.ptr<float>();
                for (int i = 0; i < 5; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        py[i * 4 + j] = px[j * 5 + i];
                    }
                }
                func->execute();
                if (!sync) {
                    func->wait();
                }
                MGB_ASSERT_TENSOR_EQ(expect, host_y);
            }
        }
    }
}

template <>
void run<shape_dep_const_shape>(CompNode cn) {
    // load model using const var shape to work around shape dependencies
    using namespace serialization;
    HostTensorGenerator<> gen;
    auto host_x = gen({4, 5}, cn);
    auto fname = output_file("test_comp_node_record_shape_dep_const_shape");

    HostTensorND y_expect;
    {
        // dump graph
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, OperatorNodeConfig{"x"}),
             y = x.flatten() +
                 opr::reduce_sum(opr::GetVarShape::make(x), x.make_scalar(1));

        graph->compile({make_callback_copy(y, y_expect)})->execute();

        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()));
        dumper->dump({y});
    }

    HostTensorND host_y;
    {
        GraphLoadConfig config;
        config.const_var_shape = true;
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()));
        auto load_rst = loader->load(config);
        load_rst.graph->options().comp_node_seq_record_level = 2;
        load_rst.graph->options().var_sanity_check_first_run = false;
        auto x_inp = load_rst.tensor_map.at("x");
        auto y = load_rst.output_var_list.at(0);
        auto func = load_rst.graph_compile({make_callback_copy(y, host_y)});

        x_inp->copy_from(*host_x);
        func->execute();
    }

    MGB_ASSERT_TENSOR_EQ(y_expect, host_y);
}

//! single thread multi recorder run interleave
template <>
void run<multi_recorder_run>(CompNode cn) {
    using ConvParam = opr::Convolution::Param;
    ConvParam param;
    param.sparse = ConvParam::Sparse::GROUP;
    HostTensorGenerator<> gen;
    std::vector<HostTensorND> host_z_v(2, HostTensorND());
    std::vector<std::unique_ptr<mgb::cg::AsyncExecutable>> funcs;
    auto host_x = gen({3, 4, 10, 8}, cn), host_y = gen({2, 3, 2, 3, 3}, cn);
    auto gen_graph = [&](int graph_id) -> std::unique_ptr<mgb::cg::AsyncExecutable> {
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             y = opr::Host2DeviceCopy::make(*graph, host_y),
             z = opr::Convolution::make(x, y, param);
        graph->options().comp_node_seq_record_level = 1;
        return graph->compile({make_callback_copy(z, host_z_v[graph_id])});
    };
    funcs.push_back(gen_graph(0));
    funcs.push_back(gen_graph(1));
    for (int iter = 0; iter < 10; ++iter) {
        host_x->copy_from_fixlayout(*gen(host_x->shape(), cn));
        funcs[0]->execute();
        funcs[1]->execute();
        auto expect = eval_conv_cpu<opr::Convolution>(*host_x, *host_y, param);
        MGB_ASSERT_TENSOR_NEAR(expect, host_z_v[0], 1e-3) << "iter " << iter;
        MGB_ASSERT_TENSOR_NEAR(expect, host_z_v[1], 1e-3) << "iter " << iter;
    }
}

template <>
void run<void>(CompNode) {}

}  // namespace seq_rec
}  // namespace comp_node_test
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
